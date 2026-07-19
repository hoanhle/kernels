#include "common.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

//----------------------------------------------------------------------------
// v3.tma: v2's double-buffered MMA kernel with TMA global-to-shared staging.

template <int BM, int BN, int BK, int NUM_WARP_M, int NUM_WARP_N>
__global__ void __launch_bounds__(NUM_WARP_M * NUM_WARP_N * 32, 1) matmul_v3_tma_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    nv_bfloat16 *C,
    int M,
    int N,
    int K) {
    constexpr int WARP_SIZE = 32;
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int SWIZZLE_CHUNK_BYTES = 16;
    constexpr int SWIZZLE_CHUNK_ELEMS = SWIZZLE_CHUNK_BYTES / sizeof(nv_bfloat16);
    constexpr int NUM_STAGES = 2;
    constexpr int WARP_M = BM / NUM_WARP_M;
    constexpr int WARP_N = BN / NUM_WARP_N;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int A_STAGE_BYTES = BM * BK * sizeof(nv_bfloat16);
    constexpr int B_STAGE_BYTES = BN * BK * sizeof(nv_bfloat16);
    constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;

    static_assert(BM % NUM_WARP_M == 0);
    static_assert(BN % NUM_WARP_N == 0);
    static_assert(BK % MMA_K == 0);
    static_assert(BK % SWIZZLE_CHUNK_ELEMS == 0);
    static_assert(BK * sizeof(nv_bfloat16) == 64);
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);

    extern __shared__ __align__(1024) unsigned char smem[];

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warp_id_m = warp_id / NUM_WARP_N;
    const int warp_id_n = warp_id % NUM_WARP_N;
    const int offset_m = blockIdx.y * BM;
    const int offset_n = blockIdx.x * BN;
    const uint32_t smem_addr = cvta_shared(smem);
    const uint32_t mbarrier_addr = smem_addr + NUM_STAGES * STAGE_BYTES;

    if (tid == 0) {
        for (int stage = 0; stage < NUM_STAGES; stage++)
            mbarrier_init(mbarrier_addr + stage * 8, 1);
        mbarrier_fence_init();
    }
    __syncthreads();

    float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

    auto load_stage = [&](int stage, int bk) {
        if (warp_id == 0 && elect_one_sync()) {
            const uint32_t As = smem_addr + stage * STAGE_BYTES;
            const uint32_t Bs = As + A_STAGE_BYTES;
            const uint32_t mbarrier = mbarrier_addr + stage * 8;

            tma_2d_g2s(As, &A_tmap, bk, offset_m, mbarrier);
            tma_2d_g2s(Bs, &B_tmap, bk, offset_n, mbarrier);
            mbarrier_arrive_expect_tx(mbarrier, STAGE_BYTES);
        }
    };

    auto compute_stage = [&](int stage) {
        const unsigned char *stage_smem = smem + stage * STAGE_BYTES;
        const nv_bfloat16 *As = reinterpret_cast<const nv_bfloat16 *>(stage_smem);
        const nv_bfloat16 *Bs = reinterpret_cast<const nv_bfloat16 *>(stage_smem + A_STAGE_BYTES);

        for (int k = 0; k < BK; k += MMA_K) {
            uint32_t B_reg[NUM_MMA_N][2];

            for (int n = 0; n < NUM_MMA_N; n++) {
                const int local_col = warp_id_n * WARP_N + n * MMA_N + lane % 8;
                const int local_k = k + (lane / 8) * 8;
                constexpr int stride_bytes = BK * sizeof(nv_bfloat16);
                const uint32_t src =
                    cvta_shared(Bs) +
                    swizzle_16b_offset<stride_bytes>(local_col, local_k / SWIZZLE_CHUNK_ELEMS);
                ldmatrix_x2(B_reg[n], src);
            }

            for (int m = 0; m < NUM_MMA_M; m++) {
                uint32_t A_reg[4];
                const int local_row = warp_id_m * WARP_M + m * MMA_M + lane % 16;
                const int local_k = k + (lane / 16) * 8;
                constexpr int stride_bytes = BK * sizeof(nv_bfloat16);
                const uint32_t src =
                    cvta_shared(As) +
                    swizzle_16b_offset<stride_bytes>(local_row, local_k / SWIZZLE_CHUNK_ELEMS);
                ldmatrix_x4(A_reg, src);

                for (int n = 0; n < NUM_MMA_N; n++)
                    mma_m16n8k16(A_reg, B_reg[n], acc[m][n]);
            }
        }
    };

    const int num_k_tiles = (K + BK - 1) / BK;
    int stage = 0;
    int phase = 0;
    load_stage(0, 0);

    for (int tile = 0; tile < num_k_tiles; tile++) {
        const int next_tile = tile + 1;

        if (next_tile < num_k_tiles)
            load_stage(next_tile % NUM_STAGES, next_tile * BK);

        if (warp_id == 0)
            mbarrier_wait(mbarrier_addr + stage * 8, phase);
        __syncthreads();

        compute_stage(stage);
        __syncthreads();

        stage = (stage + 1) % NUM_STAGES;
        if (stage == 0)
            phase ^= 1;
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

void matmul_v2_cp_async_double_buffered_swizzled_bf16(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
    nv_bfloat16 *C,
    int M,
    int N,
    int K);

static void init_tensor_map(
    CUtensorMap *tensor_map,
    const nv_bfloat16 *data,
    uint64_t height,
    uint64_t width,
    uint32_t box_height,
    uint32_t box_width) {
    constexpr uint32_t rank = 2;
    const uint64_t global_dims[rank] = {width, height};
    const uint64_t global_strides[rank - 1] = {width * sizeof(nv_bfloat16)};
    const uint32_t box_dims[rank] = {box_width, box_height};
    const uint32_t element_strides[rank] = {1, 1};

    const CUresult status = cuTensorMapEncodeTiled(
        tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        const_cast<nv_bfloat16 *>(data),
        global_dims,
        global_strides,
        box_dims,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        // TMA applies the same 64-byte shared-memory swizzle consumed by ldmatrix above.
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (status != CUDA_SUCCESS) {
        const char *message = nullptr;
        cuGetErrorString(status, &message);
        throw std::runtime_error("cuTensorMapEncodeTiled failed: " + std::string(message ? message : "unknown error"));
    }
}

void matmul_v3_tma_bf16(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
    nv_bfloat16 *C,
    int M,
    int N,
    int K) {
    if (K % 8 != 0) {
        matmul_v2_cp_async_double_buffered_swizzled_bf16(A, B, C, M, N, K);
        return;
    }

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int NUM_WARP_M = 4;
    constexpr int NUM_WARP_N = 2;
    constexpr int NUM_STAGES = 2;
    constexpr int smem_size =
        (BM + BN) * BK * sizeof(nv_bfloat16) * NUM_STAGES + NUM_STAGES * sizeof(uint64_t);

    CUtensorMap A_tmap;
    CUtensorMap B_tmap;
    init_tensor_map(&A_tmap, A, M, K, BM, BK);
    init_tensor_map(&B_tmap, B, N, K, BN, BK);

    const dim3 threads(NUM_WARP_M * NUM_WARP_N * 32);
    const dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    auto kernel = matmul_v3_tma_kernel<BM, BN, BK, NUM_WARP_M, NUM_WARP_N>;

    kernel<<<blocks, threads, smem_size>>>(A_tmap, B_tmap, C, M, N, K);
}
