#include "common.cuh"

#include <cfloat>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

//----------------------------------------------------------------------------
// Forward attention versions.
// v1.forward: tiled Tensor Core attention with an online softmax.
// v2.forward: v1 with a 128-byte shared-memory swizzle.
// v3.forward: v2 with double-buffered K/V tiles.
// v4.forward: v3 with TMA global-to-shared copies and mbarrier synchronization.
// v5.forward: v4 with a smaller query tile for higher occupancy.
//
// One CUDA block owns a BLOCK_Q tile for one query head. The query tile stays
// in registers while the block walks over K/V tiles. MHA is the special case
// query_heads == kv_heads; GQA maps multiple query heads to one KV head.

template <bool SWIZZLED, int ROWS, int COLS, int NUM_THREADS>
__device__ inline void load_tile_async(nv_bfloat16 *dst, const nv_bfloat16 *src, int tid) {
    constexpr int VECTOR_BYTES = 16;
    constexpr int VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(nv_bfloat16);
    constexpr int NUM_VECTORS = ROWS * COLS / VECTOR_ELEMENTS;
    constexpr int ROW_BYTES = COLS * sizeof(nv_bfloat16);
    constexpr int VECTORS_PER_ROW = COLS / VECTOR_ELEMENTS;

    static_assert(ROWS * COLS % VECTOR_ELEMENTS == 0);

    for (int vector = tid; vector < NUM_VECTORS; vector += NUM_THREADS) {
        const int element = vector * VECTOR_ELEMENTS;
        uint32_t dst_addr = cvta_shared(dst + element);

        if constexpr (SWIZZLED) {
            const int row = vector / VECTORS_PER_ROW;
            const int chunk = vector % VECTORS_PER_ROW;
            dst_addr =
                cvta_shared(dst) + swizzle_128b_panel_offset<ROW_BYTES>(row, chunk);
        }

        cp_async(dst_addr, src + element);
    }
}

template <
    bool SWIZZLED,
    bool PIPELINED,
    bool TMA,
    bool CAUSAL,
    int BLOCK_Q,
    int BLOCK_KV,
    int HEAD_DIM,
    int NUM_WARPS,
    int MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM) attention_fwd_kernel(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap,
    int query_heads,
    int kv_heads,
    int sequence) {
    constexpr int WARP_SIZE = 32;
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;
    constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;
    constexpr int NUM_MMA_Q = WARP_Q / MMA_M;
    constexpr int NUM_MMA_KV = BLOCK_KV / MMA_N;
    constexpr int NUM_MMA_HEAD_K = HEAD_DIM / MMA_K;
    constexpr int NUM_MMA_OUTPUT_N = HEAD_DIM / MMA_N;
    constexpr int NUM_STAGES = 2;
    constexpr int TILE_BYTES = BLOCK_Q * HEAD_DIM * sizeof(nv_bfloat16);
    constexpr int PIPELINE_BYTES =
        NUM_STAGES * 2 * BLOCK_KV * HEAD_DIM * sizeof(nv_bfloat16);
    constexpr int DATA_BYTES =
        PIPELINED && PIPELINE_BYTES > TILE_BYTES ? PIPELINE_BYTES : TILE_BYTES;
    constexpr int MBARRIER_BYTES = TMA ? NUM_STAGES * sizeof(uint64_t) : 0;

    static_assert(WARP_Q % MMA_M == 0);
    static_assert(BLOCK_KV % MMA_K == 0);
    static_assert(HEAD_DIM % MMA_K == 0);
    static_assert(!TMA || (SWIZZLED && PIPELINED));

    __align__(1024) __shared__ unsigned char smem[DATA_BYTES + MBARRIER_BYTES];
    nv_bfloat16 *tile = reinterpret_cast<nv_bfloat16 *>(smem);

    auto tile_addr = [&](const nv_bfloat16 *base, int row, int col) {
        if constexpr (SWIZZLED) {
            constexpr int ROW_BYTES = HEAD_DIM * sizeof(nv_bfloat16);
            constexpr int VECTOR_ELEMENTS = 16 / sizeof(nv_bfloat16);
            return cvta_shared(base) +
                swizzle_128b_panel_offset<ROW_BYTES>(row, col / VECTOR_ELEMENTS);
        } else {
            return cvta_shared(base + row * HEAD_DIM + col);
        }
    };

    auto kv_tile_addr = [&](const nv_bfloat16 *base, int row, int col) {
        if constexpr (TMA) {
            constexpr int PANEL_ELEMENTS = 128 / sizeof(nv_bfloat16);
            constexpr int PANEL_BYTES = BLOCK_KV * 128;
            constexpr int VECTOR_ELEMENTS = 16 / sizeof(nv_bfloat16);
            const int panel = col / PANEL_ELEMENTS;
            const int chunk = (col % PANEL_ELEMENTS) / VECTOR_ELEMENTS;
            return cvta_shared(base) + panel * PANEL_BYTES +
                swizzle_128b_panel_offset<128>(row, chunk);
        } else {
            return tile_addr(base, row, col);
        }
    };

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int query_block = blockIdx.x;
    const int query_head = blockIdx.y;
    const int batch = blockIdx.z;
    const int query_start = query_block * BLOCK_Q;
    const int group_size = query_heads / kv_heads;
    const int kv_head = query_head / group_size;
    const int kv_tensor = batch * kv_heads + kv_head;

    const size_t query_offset =
        (static_cast<size_t>(batch) * query_heads + query_head) * sequence * HEAD_DIM;
    const size_t kv_offset =
        (static_cast<size_t>(batch) * kv_heads + kv_head) * sequence * HEAD_DIM;

    Q += query_offset + static_cast<size_t>(query_start) * HEAD_DIM;
    K += kv_offset;
    V += kv_offset;
    O += query_offset + static_cast<size_t>(query_start) * HEAD_DIM;

    uint32_t Q_reg[NUM_MMA_Q][NUM_MMA_HEAD_K][4];
    float O_reg[NUM_MMA_Q][NUM_MMA_OUTPUT_N][4] = {};
    float row_max[NUM_MMA_Q][2];
    float row_sum[NUM_MMA_Q][2] = {};

    for (int query_mma = 0; query_mma < NUM_MMA_Q; query_mma++) {
        row_max[query_mma][0] = -FLT_MAX;
        row_max[query_mma][1] = -FLT_MAX;
    }

    const uint32_t smem_addr = cvta_shared(smem);
    const uint32_t mbarrier_addr = smem_addr + DATA_BYTES;

    if constexpr (TMA) {
        if (tid == 0) {
            for (int stage = 0; stage < NUM_STAGES; stage++)
                mbarrier_init(mbarrier_addr + stage * sizeof(uint64_t), 1);
            mbarrier_fence_init();
        }
        __syncthreads();
    }

    load_tile_async<SWIZZLED, BLOCK_Q, HEAD_DIM, NUM_THREADS>(tile, Q, tid);
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();

    for (int query_mma = 0; query_mma < NUM_MMA_Q; query_mma++)
        for (int head_k = 0; head_k < NUM_MMA_HEAD_K; head_k++) {
            const int row = warp_id * WARP_Q + query_mma * MMA_M + lane % MMA_M;
            const int col = head_k * MMA_K + (lane / MMA_M) * 8;
            ldmatrix_x4(Q_reg[query_mma][head_k], tile_addr(tile, row, col));
        }
    __syncthreads();

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));
    const int kv_limit = CAUSAL ? query_start + BLOCK_Q : sequence;
    constexpr int KV_TILE_ELEMENTS = BLOCK_KV * HEAD_DIM;
    constexpr int STAGE_ELEMENTS = 2 * KV_TILE_ELEMENTS;
    constexpr int KV_TILE_BYTES = KV_TILE_ELEMENTS * sizeof(nv_bfloat16);
    constexpr int STAGE_BYTES = STAGE_ELEMENTS * sizeof(nv_bfloat16);
    constexpr int PANEL_ELEMENTS = 128 / sizeof(nv_bfloat16);
    constexpr int PANEL_BYTES = BLOCK_KV * 128;

    static_assert(NUM_STAGES * STAGE_ELEMENTS * sizeof(nv_bfloat16) <= DATA_BYTES);

    auto load_tma_stage = [&](int stage, int kv_start) {
        if (warp_id == 0 && elect_one_sync()) {
            const uint32_t K_smem = smem_addr + stage * STAGE_BYTES;
            const uint32_t V_smem = K_smem + KV_TILE_BYTES;
            const uint32_t mbarrier =
                mbarrier_addr + stage * sizeof(uint64_t);

            for (int panel = 0; panel < HEAD_DIM / PANEL_ELEMENTS; panel++) {
                const int head_start = panel * PANEL_ELEMENTS;
                const int panel_offset = panel * PANEL_BYTES;
                tma_3d_g2s(
                    K_smem + panel_offset,
                    &K_tmap,
                    head_start,
                    kv_start,
                    kv_tensor,
                    mbarrier);
                tma_3d_g2s(
                    V_smem + panel_offset,
                    &V_tmap,
                    head_start,
                    kv_start,
                    kv_tensor,
                    mbarrier);
            }
            mbarrier_arrive_expect_tx(mbarrier, STAGE_BYTES);
        }
    };

    if constexpr (TMA) {
        load_tma_stage(0, 0);
    } else if constexpr (PIPELINED) {
        // Q no longer needs shared memory. Reuse its 32 KiB allocation as two
        // stages, each containing one K tile followed by its matching V tile.
        load_tile_async<SWIZZLED, BLOCK_KV, HEAD_DIM, NUM_THREADS>(tile, K, tid);
        load_tile_async<SWIZZLED, BLOCK_KV, HEAD_DIM, NUM_THREADS>(
            tile + KV_TILE_ELEMENTS,
            V,
            tid);
        cp_async_commit_group();
    }

    for (int kv_tile = 0, kv_start = 0;
         kv_start < kv_limit;
         kv_tile++, kv_start += BLOCK_KV) {
        float scores[NUM_MMA_Q][NUM_MMA_KV][4] = {};

        nv_bfloat16 *K_tile = tile;
        nv_bfloat16 *V_tile = tile;

        if constexpr (PIPELINED) {
            const int stage = kv_tile % NUM_STAGES;
            K_tile = tile + stage * STAGE_ELEMENTS;
            V_tile = K_tile + KV_TILE_ELEMENTS;

            const int next_start = kv_start + BLOCK_KV;
            if (next_start < kv_limit) {
                const int next_stage = (kv_tile + 1) % NUM_STAGES;

                // Issue the next stage before waiting for the current stage,
                // allowing its global-to-shared copies to overlap computation.
                if constexpr (TMA) {
                    load_tma_stage(next_stage, next_start);
                } else {
                    nv_bfloat16 *next_K_tile = tile + next_stage * STAGE_ELEMENTS;
                    nv_bfloat16 *next_V_tile = next_K_tile + KV_TILE_ELEMENTS;

                    load_tile_async<SWIZZLED, BLOCK_KV, HEAD_DIM, NUM_THREADS>(
                        next_K_tile,
                        K + static_cast<size_t>(next_start) * HEAD_DIM,
                        tid);
                    load_tile_async<SWIZZLED, BLOCK_KV, HEAD_DIM, NUM_THREADS>(
                        next_V_tile,
                        V + static_cast<size_t>(next_start) * HEAD_DIM,
                        tid);
                    cp_async_commit_group();
                }
            } else if constexpr (!TMA) {
                cp_async_commit_group();
            }

            if constexpr (TMA) {
                if (warp_id == 0)
                    mbarrier_wait(
                        mbarrier_addr + stage * sizeof(uint64_t),
                        (kv_tile / NUM_STAGES) % 2);
            } else {
                cp_async_wait_group<NUM_STAGES - 1>();
            }
            __syncthreads();
        } else {
            load_tile_async<SWIZZLED, BLOCK_KV, HEAD_DIM, NUM_THREADS>(
                K_tile,
                K + static_cast<size_t>(kv_start) * HEAD_DIM,
                tid);
            cp_async_commit_group();
            cp_async_wait_group<0>();
            __syncthreads();
        }

        for (int head_k = 0; head_k < NUM_MMA_HEAD_K; head_k++) {
            uint32_t K_reg[NUM_MMA_KV][2];
            for (int kv_mma = 0; kv_mma < NUM_MMA_KV; kv_mma++)
            {
                const int row = kv_mma * MMA_N + lane % MMA_N;
                const int col = head_k * MMA_K + (lane / MMA_N) * 8;
                ldmatrix_x2(K_reg[kv_mma], kv_tile_addr(K_tile, row, col));
            }

            for (int query_mma = 0; query_mma < NUM_MMA_Q; query_mma++)
                for (int kv_mma = 0; kv_mma < NUM_MMA_KV; kv_mma++)
                    mma_m16n8k16(
                        Q_reg[query_mma][head_k],
                        K_reg[kv_mma],
                        scores[query_mma][kv_mma]);
        }
        __syncthreads();

        uint32_t exp_scores[NUM_MMA_Q][BLOCK_KV / MMA_K][4];

        for (int query_mma = 0; query_mma < NUM_MMA_Q; query_mma++) {
            float tile_max[2] = {-FLT_MAX, -FLT_MAX};

            for (int kv_mma = 0; kv_mma < NUM_MMA_KV; kv_mma++) {
                float *score = scores[query_mma][kv_mma];
                for (int reg = 0; reg < 4; reg++)
                    score[reg] *= softmax_scale;

                if constexpr (CAUSAL) {
                    const int query_row_0 =
                        query_start + warp_id * WARP_Q + query_mma * MMA_M + lane / 4;
                    const int query_row_1 = query_row_0 + 8;
                    const int key_col = kv_start + kv_mma * MMA_N + (lane % 4) * 2;

                    if (key_col > query_row_0)
                        score[0] = -FLT_MAX;
                    if (key_col + 1 > query_row_0)
                        score[1] = -FLT_MAX;
                    if (key_col > query_row_1)
                        score[2] = -FLT_MAX;
                    if (key_col + 1 > query_row_1)
                        score[3] = -FLT_MAX;
                }

                tile_max[0] = fmaxf(tile_max[0], fmaxf(score[0], score[1]));
                tile_max[1] = fmaxf(tile_max[1], fmaxf(score[2], score[3]));
            }

            tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], 1));
            tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], 2));
            tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], 1));
            tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], 2));

            const float new_max_0 = fmaxf(row_max[query_mma][0], tile_max[0]);
            const float new_max_1 = fmaxf(row_max[query_mma][1], tile_max[1]);
            const float correction_0 = __expf(row_max[query_mma][0] - new_max_0);
            const float correction_1 = __expf(row_max[query_mma][1] - new_max_1);

            for (int output_n = 0; output_n < NUM_MMA_OUTPUT_N; output_n++) {
                O_reg[query_mma][output_n][0] *= correction_0;
                O_reg[query_mma][output_n][1] *= correction_0;
                O_reg[query_mma][output_n][2] *= correction_1;
                O_reg[query_mma][output_n][3] *= correction_1;
            }

            row_max[query_mma][0] = new_max_0;
            row_max[query_mma][1] = new_max_1;

            float tile_sum[2] = {};
            for (int kv_mma = 0; kv_mma < NUM_MMA_KV; kv_mma++) {
                float *score = scores[query_mma][kv_mma];
                score[0] = __expf(score[0] - row_max[query_mma][0]);
                score[1] = __expf(score[1] - row_max[query_mma][0]);
                score[2] = __expf(score[2] - row_max[query_mma][1]);
                score[3] = __expf(score[3] - row_max[query_mma][1]);

                tile_sum[0] += score[0] + score[1];
                tile_sum[1] += score[2] + score[3];

                nv_bfloat162 *pairs =
                    reinterpret_cast<nv_bfloat162 *>(exp_scores[query_mma][kv_mma / 2]);
                pairs[(kv_mma % 2) * 2] =
                    __floats2bfloat162_rn(score[0], score[1]);
                pairs[(kv_mma % 2) * 2 + 1] =
                    __floats2bfloat162_rn(score[2], score[3]);
            }

            tile_sum[0] += __shfl_xor_sync(0xffffffff, tile_sum[0], 1);
            tile_sum[0] += __shfl_xor_sync(0xffffffff, tile_sum[0], 2);
            tile_sum[1] += __shfl_xor_sync(0xffffffff, tile_sum[1], 1);
            tile_sum[1] += __shfl_xor_sync(0xffffffff, tile_sum[1], 2);

            row_sum[query_mma][0] =
                row_sum[query_mma][0] * correction_0 + tile_sum[0];
            row_sum[query_mma][1] =
                row_sum[query_mma][1] * correction_1 + tile_sum[1];
        }

        if constexpr (!PIPELINED) {
            load_tile_async<SWIZZLED, BLOCK_KV, HEAD_DIM, NUM_THREADS>(
                V_tile,
                V + static_cast<size_t>(kv_start) * HEAD_DIM,
                tid);
            cp_async_commit_group();
            cp_async_wait_group<0>();
            __syncthreads();
        }

        for (int output_n = 0; output_n < NUM_MMA_OUTPUT_N; output_n++) {
            uint32_t V_reg[BLOCK_KV / MMA_K][2];
            for (int kv_mma = 0; kv_mma < BLOCK_KV / MMA_K; kv_mma++) {
                const int row = kv_mma * MMA_K + lane % MMA_M;
                const int col = output_n * MMA_N + (lane / MMA_M) * 8;
                ldmatrix_x2_trans(
                    V_reg[kv_mma],
                    kv_tile_addr(V_tile, row, col));
            }

            for (int query_mma = 0; query_mma < NUM_MMA_Q; query_mma++)
                for (int kv_mma = 0; kv_mma < BLOCK_KV / MMA_K; kv_mma++)
                    mma_m16n8k16(
                        exp_scores[query_mma][kv_mma],
                        V_reg[kv_mma],
                        O_reg[query_mma][output_n]);
        }
        __syncthreads();
    }

    for (int query_mma = 0; query_mma < NUM_MMA_Q; query_mma++)
        for (int output_n = 0; output_n < NUM_MMA_OUTPUT_N; output_n++) {
            const int row = warp_id * WARP_Q + query_mma * MMA_M + lane / 4;
            const int col = output_n * MMA_N + (lane % 4) * 2;
            float *output = O_reg[query_mma][output_n];

            reinterpret_cast<nv_bfloat162 *>(O + row * HEAD_DIM + col)[0] =
                __floats2bfloat162_rn(
                    output[0] / row_sum[query_mma][0],
                    output[1] / row_sum[query_mma][0]);
            reinterpret_cast<nv_bfloat162 *>(O + (row + 8) * HEAD_DIM + col)[0] =
                __floats2bfloat162_rn(
                    output[2] / row_sum[query_mma][1],
                    output[3] / row_sum[query_mma][1]);
        }
}

template <
    bool SWIZZLED,
    bool PIPELINED,
    bool TMA,
    int BLOCK_Q,
    int BLOCK_KV,
    int NUM_WARPS = 4,
    int MIN_BLOCKS_PER_SM = 1>
void launch_attention_fwd(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int batch,
    int query_heads,
    int kv_heads,
    int sequence,
    int head_dim,
    bool causal,
    const CUtensorMap& K_tmap,
    const CUtensorMap& V_tmap) {
    constexpr int HEAD_DIM = 128;
    const dim3 threads(NUM_WARPS * 32);
    const dim3 blocks(sequence / BLOCK_Q, query_heads, batch);

    if (causal) {
        attention_fwd_kernel<
            SWIZZLED,
            PIPELINED,
            TMA,
            true,
            BLOCK_Q,
            BLOCK_KV,
            HEAD_DIM,
            NUM_WARPS,
            MIN_BLOCKS_PER_SM>
            <<<blocks, threads>>>(
                Q,
                K,
                V,
                O,
                K_tmap,
                V_tmap,
                query_heads,
                kv_heads,
                sequence);
    } else {
        attention_fwd_kernel<
            SWIZZLED,
            PIPELINED,
            TMA,
            false,
            BLOCK_Q,
            BLOCK_KV,
            HEAD_DIM,
            NUM_WARPS,
            MIN_BLOCKS_PER_SM>
            <<<blocks, threads>>>(
                Q,
                K,
                V,
                O,
                K_tmap,
                V_tmap,
                query_heads,
                kv_heads,
                sequence);
    }
}

static CUtensorMap empty_tensor_map() {
    return {};
}

void attention_v1_fwd_bf16(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int batch,
    int query_heads,
    int kv_heads,
    int sequence,
    int head_dim,
    bool causal) {
    launch_attention_fwd<false, false, false, 128, 32>(
        Q,
        K,
        V,
        O,
        batch,
        query_heads,
        kv_heads,
        sequence,
        head_dim,
        causal,
        empty_tensor_map(),
        empty_tensor_map());
}

void attention_v2_fwd_bf16(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int batch,
    int query_heads,
    int kv_heads,
    int sequence,
    int head_dim,
    bool causal) {
    launch_attention_fwd<true, false, false, 128, 32>(
        Q,
        K,
        V,
        O,
        batch,
        query_heads,
        kv_heads,
        sequence,
        head_dim,
        causal,
        empty_tensor_map(),
        empty_tensor_map());
}

void attention_v3_fwd_bf16(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int batch,
    int query_heads,
    int kv_heads,
    int sequence,
    int head_dim,
    bool causal) {
    launch_attention_fwd<true, true, false, 128, 32>(
        Q,
        K,
        V,
        O,
        batch,
        query_heads,
        kv_heads,
        sequence,
        head_dim,
        causal,
        empty_tensor_map(),
        empty_tensor_map());
}

static void init_kv_tensor_map(
    CUtensorMap *tensor_map,
    const nv_bfloat16 *data,
    int tensors,
    int sequence,
    int head_dim) {
    constexpr uint32_t RANK = 3;
    constexpr uint32_t PANEL_ELEMENTS = 128 / sizeof(nv_bfloat16);
    constexpr uint32_t BLOCK_KV = 32;

    const uint64_t global_dims[RANK] = {
        static_cast<uint64_t>(head_dim),
        static_cast<uint64_t>(sequence),
        static_cast<uint64_t>(tensors),
    };
    const uint64_t global_strides[RANK - 1] = {
        static_cast<uint64_t>(head_dim) * sizeof(nv_bfloat16),
        static_cast<uint64_t>(sequence) * head_dim * sizeof(nv_bfloat16),
    };
    const uint32_t box_dims[RANK] = {PANEL_ELEMENTS, BLOCK_KV, 1};
    const uint32_t element_strides[RANK] = {1, 1, 1};

    const CUresult status = cuTensorMapEncodeTiled(
        tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        RANK,
        const_cast<nv_bfloat16 *>(data),
        global_dims,
        global_strides,
        box_dims,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (status != CUDA_SUCCESS) {
        const char *message = nullptr;
        cuGetErrorString(status, &message);
        throw std::runtime_error(
            "cuTensorMapEncodeTiled failed: " +
            std::string(message ? message : "unknown error"));
    }
}

void attention_v4_fwd_bf16(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int batch,
    int query_heads,
    int kv_heads,
    int sequence,
    int head_dim,
    bool causal) {
    CUtensorMap K_tmap;
    CUtensorMap V_tmap;
    init_kv_tensor_map(&K_tmap, K, batch * kv_heads, sequence, head_dim);
    init_kv_tensor_map(&V_tmap, V, batch * kv_heads, sequence, head_dim);

    launch_attention_fwd<true, true, true, 128, 32>(
        Q,
        K,
        V,
        O,
        batch,
        query_heads,
        kv_heads,
        sequence,
        head_dim,
        causal,
        K_tmap,
        V_tmap);
}

void attention_v5_fwd_bf16(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int batch,
    int query_heads,
    int kv_heads,
    int sequence,
    int head_dim,
    bool causal) {
    CUtensorMap K_tmap;
    CUtensorMap V_tmap;
    init_kv_tensor_map(&K_tmap, K, batch * kv_heads, sequence, head_dim);
    init_kv_tensor_map(&V_tmap, V, batch * kv_heads, sequence, head_dim);

    launch_attention_fwd<true, true, true, 64, 32, 4, 3>(
        Q,
        K,
        V,
        O,
        batch,
        query_heads,
        kv_heads,
        sequence,
        head_dim,
        causal,
        K_tmap,
        V_tmap);
}
