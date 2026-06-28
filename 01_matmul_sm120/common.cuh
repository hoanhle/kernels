#pragma once

#include <cstdint>

#include <cuda_bf16.h>

//----------------------------------------------------------------------------
// PTX helpers.

__device__ inline uint32_t cvta_shared(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <int STRIDE_BYTES>
__device__ inline int swizzle_16b_offset(int row, int chunk) {
    static_assert(STRIDE_BYTES >= 16 && STRIDE_BYTES <= 128);
    static_assert((STRIDE_BYTES & (STRIDE_BYTES - 1)) == 0);
    constexpr int rows_per_xor = 128 / STRIDE_BYTES;
    // Shared-memory bank layout and XOR swizzling:
    // https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/
    // https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/
    const int swizzled_chunk = chunk ^ ((row % 8) / rows_per_xor);
    return row * STRIDE_BYTES + swizzled_chunk * 16;
}

__device__ inline void ldmatrix_x2(uint32_t reg[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(addr));
}

__device__ inline void ldmatrix_x4(uint32_t reg[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr));
}

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

__device__ inline void cp_async_zfill(uint32_t dst, const void *src) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, 0;"
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
