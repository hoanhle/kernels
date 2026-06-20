#include <cuda_bf16.h>
#include <cuda_runtime.h>

//----------------------------------------------------------------------------
// v0.naive: one CUDA thread computes one C element.

__global__ void matmul_v0_kernel(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
    nv_bfloat16 *C,
    int M,
    int N,
    int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        const nv_bfloat16 prod = A[row * K + k] * B[col * K + k];
        acc += __bfloat162float(prod);
    }

    C[row * N + col] = __float2bfloat16(acc);
}

//----------------------------------------------------------------------------
// Host launcher.

void matmul_v0_bf16(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    const dim3 block_size(16, 16);
    const dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);
    matmul_v0_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}
