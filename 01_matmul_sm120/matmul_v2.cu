#include "common.cuh"

#include <cuda_runtime.h>

//----------------------------------------------------------------------------
// v2.cp_async: v1 with 16-byte asynchronous copies from global to shared memory.

template <int BM, int BN, int BK, int NUM_WARP_M, int NUM_WARP_N>
__global__ void __launch_bounds__(NUM_WARP_M * NUM_WARP_N * 32, 1) matmul_v2_cp_async_kernel(
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
    constexpr int COPY_BYTES = 16;
    constexpr int COPY_ELEMS = COPY_BYTES / sizeof(nv_bfloat16);
    constexpr int num_threads = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int WARP_M = BM / NUM_WARP_M;
    constexpr int WARP_N = BN / NUM_WARP_N;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;

    static_assert(BM % NUM_WARP_M == 0);
    static_assert(BN % NUM_WARP_N == 0);
    static_assert(BK % MMA_K == 0);
    static_assert(BK % COPY_ELEMS == 0);
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
        for (int idx = tid * COPY_ELEMS; idx < BM * BK; idx += num_threads * COPY_ELEMS) {
            const int local_row = idx / BK;
            const int local_k = idx % BK;
            const int row = offset_m + local_row;
            const int k = bk + local_k;
            const bool valid = row < M && k < K;
            const uint32_t dst = cvta_shared(As + idx);

            if (valid)
                cp_async(dst, A + row * K + k);
            else
                cp_async_zfill(dst, A);
        }

        for (int idx = tid * COPY_ELEMS; idx < BN * BK; idx += num_threads * COPY_ELEMS) {
            const int local_col = idx / BK;
            const int local_k = idx % BK;
            const int col = offset_n + local_col;
            const int k = bk + local_k;
            const bool valid = col < N && k < K;
            const uint32_t dst = cvta_shared(Bs + idx);

            if (valid)
                cp_async(dst, B + col * K + k);
            else
                cp_async_zfill(dst, B);
        }

        cp_async_commit_group();
        cp_async_wait_group<0>();
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

void matmul_v1_mma_tiled_bf16(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
    nv_bfloat16 *C,
    int M,
    int N,
    int K);

//----------------------------------------------------------------------------
// v2.cp_async_double_buffered: load the next K tile while computing the current tile.

template <bool SWIZZLED, int BM, int BN, int BK, int NUM_WARP_M, int NUM_WARP_N>
__global__ void __launch_bounds__(NUM_WARP_M * NUM_WARP_N * 32, 1) matmul_v2_cp_async_double_buffered_kernel(
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
    constexpr int COPY_BYTES = 16;
    constexpr int COPY_ELEMS = COPY_BYTES / sizeof(nv_bfloat16);
    constexpr int NUM_STAGES = 2;
    constexpr int num_threads = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int WARP_M = BM / NUM_WARP_M;
    constexpr int WARP_N = BN / NUM_WARP_N;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int A_STAGE_ELEMS = BM * BK;
    constexpr int B_STAGE_ELEMS = BN * BK;
    constexpr int STAGE_ELEMS = A_STAGE_ELEMS + B_STAGE_ELEMS;

    static_assert(BM % NUM_WARP_M == 0);
    static_assert(BN % NUM_WARP_N == 0);
    static_assert(BK % MMA_K == 0);
    static_assert(BK % COPY_ELEMS == 0);
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);

    extern __shared__ __align__(16) nv_bfloat16 smem[];

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warp_id_m = warp_id / NUM_WARP_N;
    const int warp_id_n = warp_id % NUM_WARP_N;
    const int offset_m = blockIdx.y * BM;
    const int offset_n = blockIdx.x * BN;

    float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

    auto load_stage = [&](int stage, int bk) {
        nv_bfloat16 *As = smem + stage * STAGE_ELEMS;
        nv_bfloat16 *Bs = As + A_STAGE_ELEMS;

        for (int idx = tid * COPY_ELEMS; idx < BM * BK; idx += num_threads * COPY_ELEMS) {
            const int local_row = idx / BK;
            const int local_k = idx % BK;
            const int row = offset_m + local_row;
            const int k = bk + local_k;
            const bool valid = row < M && k < K;
            uint32_t dst = cvta_shared(As + idx);

            if constexpr (SWIZZLED) {
                // Swizzle 16-byte chunks as they enter shared memory:
                // https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/
                constexpr int stride_bytes = BK * sizeof(nv_bfloat16);
                dst = cvta_shared(As) + swizzle_16b_offset<stride_bytes>(local_row, local_k / COPY_ELEMS);
            }

            if (valid)
                cp_async(dst, A + row * K + k);
            else
                cp_async_zfill(dst, A);
        }

        for (int idx = tid * COPY_ELEMS; idx < BN * BK; idx += num_threads * COPY_ELEMS) {
            const int local_col = idx / BK;
            const int local_k = idx % BK;
            const int col = offset_n + local_col;
            const int k = bk + local_k;
            const bool valid = col < N && k < K;
            uint32_t dst = cvta_shared(Bs + idx);

            if constexpr (SWIZZLED) {
                constexpr int stride_bytes = BK * sizeof(nv_bfloat16);
                dst = cvta_shared(Bs) + swizzle_16b_offset<stride_bytes>(local_col, local_k / COPY_ELEMS);
            }

            if (valid)
                cp_async(dst, B + col * K + k);
            else
                cp_async_zfill(dst, B);
        }

        cp_async_commit_group();
    };

    auto compute_stage = [&](int stage) {
        const nv_bfloat16 *As = smem + stage * STAGE_ELEMS;
        const nv_bfloat16 *Bs = As + A_STAGE_ELEMS;

        for (int k = 0; k < BK; k += MMA_K) {
            uint32_t B_reg[NUM_MMA_N][2];

            for (int n = 0; n < NUM_MMA_N; n++) {
                const int local_col = warp_id_n * WARP_N + n * MMA_N + lane % 8;
                const int local_k = k + (lane / 8) * 8;
                uint32_t src = cvta_shared(Bs + local_col * BK + local_k);

                if constexpr (SWIZZLED) {
                    constexpr int stride_bytes = BK * sizeof(nv_bfloat16);
                    src = cvta_shared(Bs) + swizzle_16b_offset<stride_bytes>(local_col, local_k / COPY_ELEMS);
                }

                ldmatrix_x2(B_reg[n], src);
            }

            for (int m = 0; m < NUM_MMA_M; m++) {
                uint32_t A_reg[4];
                const int local_row = warp_id_m * WARP_M + m * MMA_M + lane % 16;
                const int local_k = k + (lane / 16) * 8;
                uint32_t src = cvta_shared(As + local_row * BK + local_k);

                if constexpr (SWIZZLED) {
                    constexpr int stride_bytes = BK * sizeof(nv_bfloat16);
                    src = cvta_shared(As) + swizzle_16b_offset<stride_bytes>(local_row, local_k / COPY_ELEMS);
                }

                ldmatrix_x4(A_reg, src);

                for (int n = 0; n < NUM_MMA_N; n++)
                    mma_m16n8k16(A_reg, B_reg[n], acc[m][n]);
            }
        }
    };

    const int num_k_tiles = (K + BK - 1) / BK;
    load_stage(0, 0);

    for (int tile = 0; tile < num_k_tiles; tile++) {
        const int next_tile = tile + 1;

        if (next_tile < num_k_tiles)
            load_stage(next_tile % NUM_STAGES, next_tile * BK);
        else
            cp_async_commit_group();

        cp_async_wait_group<NUM_STAGES - 1>();
        __syncthreads();
        compute_stage(tile % NUM_STAGES);
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

void matmul_v2_cp_async_bf16(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    if (K % 8 != 0) {
        matmul_v1_mma_tiled_bf16(A, B, C, M, N, K);
        return;
    }

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int NUM_WARP_M = 4;
    constexpr int NUM_WARP_N = 2;
    const dim3 threads(NUM_WARP_M * NUM_WARP_N * 32);
    const dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_v2_cp_async_kernel<BM, BN, BK, NUM_WARP_M, NUM_WARP_N><<<blocks, threads>>>(A, B, C, M, N, K);
}

void matmul_v2_cp_async_double_buffered_bf16(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
    nv_bfloat16 *C,
    int M,
    int N,
    int K) {
    if (K % 8 != 0) {
        matmul_v1_mma_tiled_bf16(A, B, C, M, N, K);
        return;
    }

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int NUM_WARP_M = 4;
    constexpr int NUM_WARP_N = 2;
    constexpr int NUM_STAGES = 2;
    constexpr int smem_size = (BM + BN) * BK * sizeof(nv_bfloat16) * NUM_STAGES;
    const dim3 threads(NUM_WARP_M * NUM_WARP_N * 32);
    const dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    auto kernel = matmul_v2_cp_async_double_buffered_kernel<false, BM, BN, BK, NUM_WARP_M, NUM_WARP_N>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<blocks, threads, smem_size>>>(A, B, C, M, N, K);
}

void matmul_v2_cp_async_double_buffered_swizzled_bf16(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
    nv_bfloat16 *C,
    int M,
    int N,
    int K) {
    if (K % 8 != 0) {
        matmul_v1_mma_tiled_bf16(A, B, C, M, N, K);
        return;
    }

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int NUM_WARP_M = 4;
    constexpr int NUM_WARP_N = 2;
    constexpr int NUM_STAGES = 2;
    constexpr int smem_size = (BM + BN) * BK * sizeof(nv_bfloat16) * NUM_STAGES;
    const dim3 threads(NUM_WARP_M * NUM_WARP_N * 32);
    const dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    auto kernel = matmul_v2_cp_async_double_buffered_kernel<true, BM, BN, BK, NUM_WARP_M, NUM_WARP_N>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<blocks, threads, smem_size>>>(A, B, C, M, N, K);
}
