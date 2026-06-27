#include "common.cuh"

#include <cuda_runtime.h>

//----------------------------------------------------------------------------
// v1.mma_tiled: one CUDA block computes one BM x BN tile of C using multiple warps.
// BM/BN/BK are the block tile sizes in M/N/K. Each warp computes a WARP_M x WARP_N tile using mma.m16n8k16.

template <int BM, int BN, int BK, int NUM_WARP_M, int NUM_WARP_N>
__global__ void __launch_bounds__(NUM_WARP_M * NUM_WARP_N * 32, 1) matmul_v1_mma_tiled_kernel(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
    nv_bfloat16 *C,
    int M,
    int N,
    int K) {
    constexpr int WARP_SIZE = 32;
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int num_threads = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int WARP_M = BM / NUM_WARP_M;
    constexpr int WARP_N = BN / NUM_WARP_N;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;

    static_assert(BM % NUM_WARP_M == 0);
    static_assert(BN % NUM_WARP_N == 0);
    static_assert(BK % MMA_K == 0);
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);

    __align__(16) __shared__ nv_bfloat16 As[BM * BK];
    __align__(16) __shared__ nv_bfloat16 Bs[BN * BK];

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warp_id_m = warp_id / NUM_WARP_N;
    const int warp_id_n = warp_id % NUM_WARP_N;
    const int offset_m = blockIdx.y * BM;
    const int offset_n = blockIdx.x * BN;

    float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

    for (int bk = 0; bk < K; bk += BK) {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
            const int local_row = idx / BK;
            const int local_k = idx % BK;
            const int row = offset_m + local_row;
            const int k = bk + local_k;
            As[local_row * BK + local_k] =
                (row < M && k < K) ? A[row * K + k] : __float2bfloat16(0.0f);
        }

        for (int idx = tid; idx < BN * BK; idx += num_threads) {
            const int local_col = idx / BK;
            const int local_k = idx % BK;
            const int col = offset_n + local_col;
            const int k = bk + local_k;
            Bs[local_col * BK + local_k] =
                (col < N && k < K) ? B[col * K + k] : __float2bfloat16(0.0f);
        }

        __syncthreads();

        for (int k = 0; k < BK; k += MMA_K) {
            uint32_t B_reg[NUM_MMA_N][2];

            for (int n = 0; n < NUM_MMA_N; n++) {
                const int local_col = warp_id_n * WARP_N + n * MMA_N + lane % 8;
                const nv_bfloat16 *B_ptr = Bs + local_col * BK + k + (lane / 8) * 8;
                ldmatrix_x2(B_reg[n], cvta_shared(B_ptr));
            }

            for (int m = 0; m < NUM_MMA_M; m++) {
                uint32_t A_reg[4];
                const int local_row = warp_id_m * WARP_M + m * MMA_M + lane % 16;
                const nv_bfloat16 *A_ptr = As + local_row * BK + k + (lane / 16) * 8;
                ldmatrix_x4(A_reg, cvta_shared(A_ptr));

                for (int n = 0; n < NUM_MMA_N; n++)
                    mma_m16n8k16(A_reg, B_reg[n], acc[m][n]);
            }
        }

        __syncthreads();
    }

    for (int m = 0; m < NUM_MMA_M; m++) {
        for (int n = 0; n < NUM_MMA_N; n++) {
            // Accumulator fragment layout for mma.m16n8k16:
            // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
            const int row = offset_m + warp_id_m * WARP_M + m * MMA_M + lane / 4;
            const int col = offset_n + warp_id_n * WARP_N + n * MMA_N + (lane % 4) * 2;
            float *regs = acc[m][n];

            if (row < M && col < N)
                C[row * N + col] = __float2bfloat16(regs[0]);
            if (row < M && col + 1 < N)
                C[row * N + col + 1] = __float2bfloat16(regs[1]);
            if (row + 8 < M && col < N)
                C[(row + 8) * N + col] = __float2bfloat16(regs[2]);
            if (row + 8 < M && col + 1 < N)
                C[(row + 8) * N + col + 1] = __float2bfloat16(regs[3]);
        }
    }
}

void matmul_v1_mma_tiled_bf16(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int NUM_WARP_M = 4;
    constexpr int NUM_WARP_N = 2;
    const dim3 threads(NUM_WARP_M * NUM_WARP_N * 32);
    const dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_v1_mma_tiled_kernel<BM, BN, BK, NUM_WARP_M, NUM_WARP_N><<<blocks, threads>>>(A, B, C, M, N, K);
}
