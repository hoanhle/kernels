#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_BF16(x) TORCH_CHECK(x.dtype() == at::kBFloat16, #x " must be a BF16 tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

using AttentionBF16Fn = void(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int batch,
    int query_heads,
    int kv_heads,
    int sequence,
    int head_dim,
    bool causal);

AttentionBF16Fn attention_v1_fwd_bf16;

template <AttentionBF16Fn attention_fn>
at::Tensor attention_pt(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    bool causal) {
    CHECK_CUDA(Q);
    CHECK_CUDA(K);
    CHECK_CUDA(V);
    CHECK_BF16(Q);
    CHECK_BF16(K);
    CHECK_BF16(V);
    CHECK_CONTIGUOUS(Q);
    CHECK_CONTIGUOUS(K);
    CHECK_CONTIGUOUS(V);
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "Q, K, and V must be rank-4 tensors");
    TORCH_CHECK(K.sizes() == V.sizes(), "K and V must have the same shape");
    TORCH_CHECK(Q.size(0) == K.size(0), "Q, K, and V must have the same batch size");
    TORCH_CHECK(Q.size(2) == K.size(2), "v1 supports self-attention with equal Q and KV sequence lengths");
    TORCH_CHECK(Q.size(3) == K.size(3), "Q, K, and V must have the same head dimension");
    TORCH_CHECK(Q.size(1) % K.size(1) == 0, "The number of query heads must be divisible by the number of KV heads");
    TORCH_CHECK(Q.size(3) == 128, "v1 supports head dimension 128");
    TORCH_CHECK(Q.size(2) % 128 == 0, "v1 requires sequence length to be divisible by 128");

    const int batch = Q.size(0);
    const int query_heads = Q.size(1);
    const int kv_heads = K.size(1);
    const int sequence = Q.size(2);
    const int head_dim = Q.size(3);

    at::Tensor O = at::empty_like(Q);

    attention_fn(
        reinterpret_cast<const nv_bfloat16 *>(Q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16 *>(K.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16 *>(V.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16 *>(O.data_ptr<at::BFloat16>()),
        batch,
        query_heads,
        kv_heads,
        sequence,
        head_dim,
        causal);

    return O;
}

TORCH_LIBRARY(attention_sm120, m) {
    m.def(
        "attention_v1_fwd(Tensor Q, Tensor K, Tensor V, bool causal) -> Tensor",
        &attention_pt<attention_v1_fwd_bf16>);
}
