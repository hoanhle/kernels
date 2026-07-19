#pragma once

#include <cstdint>

#include <cuda_bf16.h>

//----------------------------------------------------------------------------
// PTX helpers.

__device__ inline uint32_t cvta_shared(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync
__device__ inline bool elect_one_sync() {
    int elected = 0;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, %1;\n"
        "@p mov.s32 %0, 1;\n"
        "}"
        : "+r"(elected)
        : "r"(0xffffffff));
    return elected;
}

// Split rows wider than 128 bytes into independent 128-byte panels, then XOR
// the row within each panel's 16-byte chunk index. Global memory remains
// row-major; cp.async stores and ldmatrix loads both use this physical offset.
template <int ROW_BYTES>
__device__ inline int swizzle_128b_panel_offset(int row, int chunk) {
    constexpr int CHUNK_BYTES = 16;
    constexpr int PANEL_BYTES = 128;
    constexpr int CHUNKS_PER_PANEL = PANEL_BYTES / CHUNK_BYTES;

    static_assert(ROW_BYTES % PANEL_BYTES == 0);

    const int panel = chunk / CHUNKS_PER_PANEL;
    const int chunk_in_panel = chunk % CHUNKS_PER_PANEL;
    const int swizzled_chunk = chunk_in_panel ^ (row % CHUNKS_PER_PANEL);
    return row * ROW_BYTES + panel * PANEL_BYTES + swizzled_chunk * CHUNK_BYTES;
}

// For QK^T, MMA expects column-major B = K^T[dimension, key], which has the
// same physical storage as row-major K[key, dimension], so no transpose is needed.
__device__ inline void ldmatrix_x2(uint32_t reg[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(addr));
}

// For PV, MMA expects column-major B = V[key, dimension], but V is stored
// row-major, so .trans produces the required B fragment during the load.
__device__ inline void ldmatrix_x2_trans(uint32_t reg[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(addr));
}

// For QK^T, MMA expects row-major A = Q[query, dimension]. The four 8x8
// matrices form the 16x16 A fragment consumed by mma.m16n8k16.
__device__ inline void ldmatrix_x4(uint32_t reg[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr));
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma
__device__ inline void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2], float C[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

__device__ inline void cp_async(uint32_t dst, const void *src) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :
        : "r"(dst), "l"(src));
}

__device__ inline void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ inline void cp_async_wait_group() {
    asm volatile(
        "cp.async.wait_group %0;"
        :
        : "n"(N));
}

__device__ inline void mbarrier_init(uint32_t addr, int count) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :
        : "r"(addr), "r"(count));
}

__device__ inline void mbarrier_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;");
}

__device__ inline void mbarrier_arrive_expect_tx(uint32_t addr, int bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :
        : "r"(addr), "r"(bytes)
        : "memory");
}

__device__ inline void mbarrier_wait(uint32_t addr, int phase) {
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "wait:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra.uni wait;\n"
        "}"
        :
        : "r"(addr), "r"(phase)
        : "memory");
}

__device__ inline void tma_3d_g2s(
    uint32_t dst,
    const void *tensor_map,
    int x,
    int y,
    int z,
    uint32_t mbarrier) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];"
        :
        : "r"(dst), "l"(tensor_map), "r"(x), "r"(y), "r"(z), "r"(mbarrier)
        : "memory");
}
