# Matmul SM120

## Resources

- https://www.aleksagordic.com/blog/matmul
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
- https://github.com/gau-nernst/learn-cuda/tree/main/02c_matmul_sm120

## Setup

Fixed M=N=K=4096. RTX 5090 has peak 209.5 BF16 TFLOPS and global memory bandwidth of 1792 GB/s

A is row-major [M, K]. B is logical [K, N] with column-major storage.


| Kernel                                          |  TFLOPS | Performance relative to cuBLAS |
|:------------------------------------------------|--------:|:------------------------------|
| CuBLAS 12.8.4.1 via PyTorch 2.9.1 CUDA 12.8     |  182.68 | 100%                          |

## v0 

The v0 kernels are CUDA-core baselines. They do not use tensor cores.

| Kernel | TFLOPS | Performance relative to cuBLAS |
|:-------|-------:|:-------------------------------|
| naive  |   1.51 | 0.83%                          |


## Notes

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