Resources
- https://www.aleksagordic.com/blog/matmul
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
- https://github.com/gau-nernst/learn-cuda/tree/main/02c_matmul_sm120


Fixed M=N=K=4096. 5090 has peak 209.5 BF16 TFLOPS.

Kernel name                                            | TFLOPS          | Performance relative to cuBLAS
-------------------------------------------------------|-----------------|-------------------------------
CuBLAS 12.8.4.1 (via 2.9.1’s CUDA 12.8)                | 182.68 TFLOPS   | 100%


