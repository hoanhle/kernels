# Matmul SM120

## Setup

Fixed M=N=K=4096. RTX 5090 has peak 209.5 BF16 TFLOPS and global memory bandwidth of 1792 GB/s

A is row-major [M, K]. B is logical [K, N] with column-major storage.


| Kernel                                          |  TFLOPS | Performance relative to cuBLAS |
|:------------------------------------------------|--------:|:------------------------------|
| CuBLAS 12.8.4.1 via PyTorch 2.9.1 CUDA 12.8     |  182.68 | 100%                          |
| v0 (tiled, block2d)                             |   29.76 | 16.29%                        |
| v1 (ldmatrix, mma_tiled)                        |  72.90  | 39.90%                        |
| v2 (cp_async, double-buffered, swizzled)        |  180.40 | 98.75%                        |

## v0 

The v0 kernels are CUDA-core baselines. They do not use tensor cores.

| Kernel | TFLOPS | Performance relative to cuBLAS |
|:-------|-------:|:-------------------------------|
| matmul_v0_naive |   1.51 | 0.83%                 |
| matmul_v0_tiled |   7.39 | 4.05%                 |
| matmul_v0_block1d |  16.70 | 9.14%               |
| matmul_v0_block2d |  29.76 | 16.29%              |

## v1

The v1 kernels use tensor cores. Threads first stage A and B tiles in shared memory, then use `ldmatrix` to load warp-level fragments into registers and `mma` to accumulate the matrix product.

| Kernel | TFLOPS | Performance relative to cuBLAS |
|:-------|-------:|:-------------------------------|
| matmul_v1_mma_tiled |  72.90 | 39.90%            |

## v2

v2 replaces v1's scalar global-to-shared-memory copies with 16-byte `cp.async` copies.

The `cp.async` fast path requires `K` to be divisible by 8. Each copy moves eight BF16 elements, and `K % 8 == 0` keeps the start of every physical A and B row 16-byte aligned. Other K sizes fall back to v1 rather than mixing asynchronous and scalar copies.

| Kernel | TFLOPS | Performance relative to cuBLAS |
|:-------|-------:|:-------------------------------|
| matmul_v2_cp_async |  117.80 | 64.49%            |
| matmul_v2_cp_async_double_buffered | 162.10 | 88.73%           |
| matmul_v2_cp_async_double_buffered_swizzled | 180.40 | 98.75% |

### Shared-memory swizzling

The swizzled kernel rearranges 16-byte chunks within each shared-memory row to reduce bank conflicts:

- [CUDA Shared Memory Bank](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/)
- [CUDA Shared Memory Swizzling](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)

### Double buffering and occupancy

Each shared-memory stage holds one A tile and one B tile:

```cpp
smem_size = (BM + BN) * BK * sizeof(nv_bfloat16) * NUM_STAGES;
```

For `BM=128`, `BN=128`, `BK=32`, BF16 elements, and two stages:

```text
A tile    = 128 * 32 * 2 bytes = 8,192 bytes
B tile    = 128 * 32 * 2 bytes = 8,192 bytes
One stage = 16,384 bytes        = 16 KiB
Two stages                        = 32,768 bytes = 32 KiB/block
```

The launch allocates this dynamic shared memory through its third argument:

```cpp
kernel<<<blocks, threads, smem_size>>>(...);
```

Using `BK=64` required 64 KiB/block, limited each SM to one resident block, and reached only 108.15 TFLOPS. Reducing `BK` to 32 requires 32 KiB/block, allowing two resident blocks after register limits are considered, and reaches 161.39 TFLOPS. Double buffering helped only after its shared-memory footprint preserved enough occupancy to hide latency.

### Pipeline schedule

`load_stage()` issues and commits asynchronous copies but does not wait for them to finish. With two stages, tiles alternate between shared-memory buffers:

```text
tile 0 -> stage 0
tile 1 -> stage 1
tile 2 -> stage 0
tile 3 -> stage 1
```

The kernel preloads tile 0, then overlaps each subsequent load with computation:

```text
Issue load 0

Issue load 1
Wait for load 0
Compute 0  || load 1 continues

Issue load 2
Wait for load 1
Compute 1  || load 2 continues

Issue load 3
Wait for load 2
Compute 2  || load 3 continues
```

`cp_async_wait_group<1>()` allows the newest copy group to remain in flight while requiring older groups issued by the calling thread to complete. The first `__syncthreads()` ensures every thread's portion of the current stage is ready before the block computes. The second ensures every thread has finished reading that stage before it is reused.

## Notes

| Metric | Value |
|:-------|:------|
| Name | NVIDIA GeForce RTX 5090 |
| Compute capability | 12.0 |
| Max threads per block | 1024 |
| Max threads per multiprocessor | 1536 |
| Tensor Cores / SM | 4 |
| Tensor Cores | 680 |
| Threads per warp | 32 |
| Max registers per block | 65536 |
| Total global memory | 32101 MiB / 33660534784 B |
| Max shared memory per block, default | 49152 B |
| Shared memory per multiprocessor | 102400 B |
| Multiprocessor count | 170 |
| Max warps per multiprocessor | 48, computed from 1536 / 32 |

```zsh
Device 0: "NVIDIA GeForce RTX 5090"
  CUDA Driver Version / Runtime Version          13.0 / 12.8
  CUDA Capability Major/Minor version number:    12.0
  Total amount of global memory:                 32101 MBytes (33660534784 bytes)
  (170) Multiprocessors, (128) CUDA Cores/MP:    21760 CUDA Cores
  GPU Max Clock rate:                            2407 MHz (2.41 GHz)
  Memory Clock rate:                             14001 Mhz
  Memory Bus Width:                              512-bit
  L2 Cache Size:                                 100663296 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 2 / 0
```

## Resources

- https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://www.aleksagordic.com/blog/matmul
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
- https://github.com/gau-nernst/learn-cuda/tree/main/02c_matmul_sm120
