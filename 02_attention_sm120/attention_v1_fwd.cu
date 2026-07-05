#include "common.cuh"

#include <cfloat>
#include <cstdint>

#include <cuda_runtime.h>

//----------------------------------------------------------------------------
// v1.forward: tiled Tensor Core attention with an online softmax.
//
// One CUDA block owns a BLOCK_Q tile for one query head. The query tile stays
// in registers while the block walks over K/V tiles. MHA is the special case
// query_heads == kv_heads; GQA maps multiple query heads to one KV head.

template <int ROWS, int COLS, int NUM_THREADS>
__device__ inline void load_tile_async(nv_bfloat16 *dst, const nv_bfloat16 *src, int tid) {
    constexpr int VECTOR_BYTES = 16;
    constexpr int VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(nv_bfloat16);
    constexpr int NUM_VECTORS = ROWS * COLS / VECTOR_ELEMENTS;

    static_assert(ROWS * COLS % VECTOR_ELEMENTS == 0);

    for (int vector = tid; vector < NUM_VECTORS; vector += NUM_THREADS) {
        const int element = vector * VECTOR_ELEMENTS;
        cp_async(cvta_shared(dst + element), src + element);
    }
}

template <bool CAUSAL, int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32, 1) attention_v1_fwd_kernel(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
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

    static_assert(WARP_Q % MMA_M == 0);
    static_assert(BLOCK_KV % MMA_K == 0);
    static_assert(HEAD_DIM % MMA_K == 0);

    __align__(16) __shared__ nv_bfloat16 tile[BLOCK_Q * HEAD_DIM];

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int query_block = blockIdx.x;
    const int query_head = blockIdx.y;
    const int batch = blockIdx.z;
    const int query_start = query_block * BLOCK_Q;
    const int group_size = query_heads / kv_heads;
    const int kv_head = query_head / group_size;

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

    load_tile_async<BLOCK_Q, HEAD_DIM, NUM_THREADS>(tile, Q, tid);
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();

    for (int query_mma = 0; query_mma < NUM_MMA_Q; query_mma++)
        for (int head_k = 0; head_k < NUM_MMA_HEAD_K; head_k++) {
            const int row = warp_id * WARP_Q + query_mma * MMA_M + lane % MMA_M;
            const int col = head_k * MMA_K + (lane / MMA_M) * 8;
            ldmatrix_x4(Q_reg[query_mma][head_k], cvta_shared(tile + row * HEAD_DIM + col));
        }
    __syncthreads();

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));
    const int kv_limit = CAUSAL ? query_start + BLOCK_Q : sequence;

    for (int kv_start = 0; kv_start < kv_limit; kv_start += BLOCK_KV) {
        float scores[NUM_MMA_Q][NUM_MMA_KV][4] = {};

        load_tile_async<BLOCK_KV, HEAD_DIM, NUM_THREADS>(
            tile,
            K + static_cast<size_t>(kv_start) * HEAD_DIM,
            tid);
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();

        for (int head_k = 0; head_k < NUM_MMA_HEAD_K; head_k++) {
            uint32_t K_reg[NUM_MMA_KV][2];
            for (int kv_mma = 0; kv_mma < NUM_MMA_KV; kv_mma++)
            {
                const int row = kv_mma * MMA_N + lane % MMA_N;
                const int col = head_k * MMA_K + (lane / MMA_N) * 8;
                ldmatrix_x2(K_reg[kv_mma], cvta_shared(tile + row * HEAD_DIM + col));
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

        load_tile_async<BLOCK_KV, HEAD_DIM, NUM_THREADS>(
            tile,
            V + static_cast<size_t>(kv_start) * HEAD_DIM,
            tid);
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();

        for (int output_n = 0; output_n < NUM_MMA_OUTPUT_N; output_n++) {
            uint32_t V_reg[BLOCK_KV / MMA_K][2];
            for (int kv_mma = 0; kv_mma < BLOCK_KV / MMA_K; kv_mma++) {
                const int row = kv_mma * MMA_K + lane % MMA_M;
                const int col = output_n * MMA_N + (lane / MMA_M) * 8;
                ldmatrix_x2_trans(
                    V_reg[kv_mma],
                    cvta_shared(tile + row * HEAD_DIM + col));
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

template <int BLOCK_Q, int BLOCK_KV>
void launch_attention_v1(
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
    constexpr int HEAD_DIM = 128;
    constexpr int NUM_WARPS = 4;

    const dim3 threads(NUM_WARPS * 32);
    const dim3 blocks(sequence / BLOCK_Q, query_heads, batch);

    if (causal) {
        attention_v1_fwd_kernel<true, BLOCK_Q, BLOCK_KV, HEAD_DIM, NUM_WARPS>
            <<<blocks, threads>>>(Q, K, V, O, query_heads, kv_heads, sequence);
    } else {
        attention_v1_fwd_kernel<false, BLOCK_Q, BLOCK_KV, HEAD_DIM, NUM_WARPS>
            <<<blocks, threads>>>(Q, K, V, O, query_heads, kv_heads, sequence);
    }
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
    launch_attention_v1<128, 32>(
        Q, K, V, O, batch, query_heads, kv_heads, sequence, head_dim, causal);
}
