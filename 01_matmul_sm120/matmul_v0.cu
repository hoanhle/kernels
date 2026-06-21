#include <cuda_bf16.h>
#include <cuda_runtime.h>

//----------------------------------------------------------------------------
// v0.naive: one CUDA thread computes one C element.

__global__ void matmul_v0_naive_kernel(
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

void matmul_v0_naive_bf16(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    const dim3 block_size(16, 16);
    const dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);
    matmul_v0_naive_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}

//----------------------------------------------------------------------------
// v0.tiled: each CUDA block computes one shared-memory tile of C.

template <int BLOCK_SIZE>
__global__ void matmul_v0_tiled_kernel(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
    nv_bfloat16 *C,
    int M,
    int N,
    int K) {
    __shared__ nv_bfloat16 As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ nv_bfloat16 Bs[BLOCK_SIZE * BLOCK_SIZE];

    const int thread_col = threadIdx.x;
    const int thread_row = threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + thread_col;
    const int row = blockIdx.y * BLOCK_SIZE + thread_row;

    float acc = 0.0f;
    for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
        const int a_col = bk + thread_col;
        const int b_row = bk + thread_row;

        As[thread_row * BLOCK_SIZE + thread_col] =
            (row < M && a_col < K) ? A[row * K + a_col] : __float2bfloat16(0.0f);

        Bs[thread_row * BLOCK_SIZE + thread_col] =
            (b_row < K && col < N) ? B[col * K + b_row] : __float2bfloat16(0.0f);

        __syncthreads();

        for (int dot = 0; dot < BLOCK_SIZE; dot++) {
            const nv_bfloat16 prod =
                As[thread_row * BLOCK_SIZE + dot] * Bs[dot * BLOCK_SIZE + thread_col];
            acc += __bfloat162float(prod);
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = __float2bfloat16(acc);
}

void matmul_v0_tiled_bf16(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    constexpr int block_size = 32;
    const dim3 threads(block_size, block_size);
    const dim3 blocks((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);
    matmul_v0_tiled_kernel<block_size><<<blocks, threads>>>(A, B, C, M, N, K);
}

//----------------------------------------------------------------------------
// v0.block1d: each CUDA thread computes TM output rows for one C column.
// BM/BN/BK are the block tile sizes in M/N/K. TM is the number of M rows each thread computes.

template <int BM, int BN, int BK, int TM>
__global__ void matmul_v0_block1d_kernel(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
    nv_bfloat16 *C,
    int M,
    int N,
    int K) {
    __shared__ nv_bfloat16 As[BM * BK];
    __shared__ nv_bfloat16 Bs[BK * BN];

    const int tid = threadIdx.x;
    const int thread_col = tid % BN;
    const int thread_row = tid / BN;
    const int row_base = blockIdx.y * BM + thread_row * TM;
    const int col = blockIdx.x * BN + thread_col;
    const int inner_col_a = tid % BK;
    const int inner_row_a = tid / BK;
    const int inner_col_b = tid % BN;
    const int inner_row_b = tid / BN;

    float acc[TM] = {0.0f};
    for (int bk = 0; bk < K; bk += BK) {
        // This shape makes each thread load one A element and one B element.
        // Later variants can let each thread load multiple entries to use cache capacity better.
        const int global_row_a = blockIdx.y * BM + inner_row_a;
        const int global_k_a = bk + inner_col_a;
        As[inner_row_a * BK + inner_col_a] =
            (global_row_a < M && global_k_a < K) ? A[global_row_a * K + global_k_a] : __float2bfloat16(0.0f);

        const int global_k_b = bk + inner_row_b;
        const int global_col_b = blockIdx.x * BN + inner_col_b;
        Bs[inner_row_b * BN + inner_col_b] =
            (global_k_b < K && global_col_b < N) ? B[global_col_b * K + global_k_b] : __float2bfloat16(0.0f);

        __syncthreads();

        for (int dot = 0; dot < BK; dot++) {
            const nv_bfloat16 b = Bs[dot * BN + thread_col];
            for (int res = 0; res < TM; res++) {
                const nv_bfloat16 prod = As[(thread_row * TM + res) * BK + dot] * b;
                acc[res] += __bfloat162float(prod);
            }
        }

        __syncthreads();
    }

    if (col < N) {
        for (int res = 0; res < TM; res++) {
            const int row = row_base + res;
            if (row < M)
                C[row * N + col] = __float2bfloat16(acc[res]);
        }
    }
}

void matmul_v0_block1d_bf16(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    const dim3 threads((BM * BN) / TM);
    const dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_v0_block1d_kernel<BM, BN, BK, TM><<<blocks, threads>>>(A, B, C, M, N, K);
}
