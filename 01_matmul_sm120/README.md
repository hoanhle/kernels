# Matmul SM120

## Resources

- https://www.aleksagordic.com/blog/matmul
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
- https://github.com/gau-nernst/learn-cuda/tree/main/02c_matmul_sm120

## Setup

Fixed M=N=K=4096. RTX 5090 has peak 209.5 BF16 TFLOPS.

A is row-major [M, K]. B is logical [K, N] with column-major storage.


| Kernel                                          |  TFLOPS | Performance relative to cuBLAS |
|:------------------------------------------------|--------:|:------------------------------|
| CuBLAS 12.8.4.1 via PyTorch 2.9.1 CUDA 12.8     |  182.68 | 100%                          |

## v0 

The v0 kernels are CUDA-core baselines. They do not use tensor cores.

| Kernel | TFLOPS | Performance relative to cuBLAS |
|:-------|-------:|:-------------------------------|
| naive  |   1.51 | 0.83%                          |
