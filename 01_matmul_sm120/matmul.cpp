#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_BF16(x) TORCH_CHECK(x.dtype() == at::kBFloat16, #x " must be a BF16 tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

using MatmulBF16Fn = void(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
    nv_bfloat16 *C,
    int M,
    int N,
    int K);

MatmulBF16Fn matmul_v0_naive_bf16;
MatmulBF16Fn matmul_v0_tiled_bf16;
MatmulBF16Fn matmul_v0_block1d_bf16;
MatmulBF16Fn matmul_v0_block2d_bf16;
MatmulBF16Fn matmul_v1_mma_tiled_bf16;
MatmulBF16Fn matmul_v2_cp_async_bf16;
MatmulBF16Fn matmul_v2_cp_async_double_buffered_bf16;
MatmulBF16Fn matmul_v2_cp_async_double_buffered_swizzled_bf16;

template <MatmulBF16Fn matmul_fn>
at::Tensor matmul_pt(const at::Tensor& A, const at::Tensor& B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_BF16(A);
    CHECK_BF16(B);
    CHECK_CONTIGUOUS(A);
    TORCH_CHECK(B.t().is_contiguous(), "B must be column-major: B.t() must be contiguous");
    TORCH_CHECK(A.size(1) == B.size(0), "A and B shapes are incompatible");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    at::Tensor C = at::empty({M, N}, A.options());

    matmul_fn(
        reinterpret_cast<const nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16 *>(C.data_ptr<at::BFloat16>()),
        M, N, K);

    return C;
}

TORCH_LIBRARY(matmul_sm120, m) {
    m.def("matmul_v0_naive(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v0_naive_bf16>);
    m.def("matmul_v0_tiled(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v0_tiled_bf16>);
    m.def("matmul_v0_block1d(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v0_block1d_bf16>);
    m.def("matmul_v0_block2d(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v0_block2d_bf16>);
    m.def("matmul_v1_mma_tiled(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v1_mma_tiled_bf16>);
    m.def("matmul_v2_cp_async(Tensor A, Tensor B) -> Tensor", &matmul_pt<matmul_v2_cp_async_bf16>);
    m.def(
        "matmul_v2_cp_async_double_buffered(Tensor A, Tensor B) -> Tensor",
        &matmul_pt<matmul_v2_cp_async_double_buffered_bf16>);
    m.def(
        "matmul_v2_cp_async_double_buffered_swizzled(Tensor A, Tensor B) -> Tensor",
        &matmul_pt<matmul_v2_cp_async_double_buffered_swizzled_bf16>);
}
