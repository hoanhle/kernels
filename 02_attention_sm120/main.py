import math
from pathlib import Path

import click
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.testing import do_bench

CURRENT_DIR = Path(__file__).resolve().parent

torch.utils.cpp_extension.load(
    'attention_sm120',
    sources=[
        CURRENT_DIR / 'attention.cpp',
        *sorted(CURRENT_DIR.glob('attention_v*.cu')),
    ],
    extra_cuda_cflags=[
        '-O3',
        '-lineinfo',
        '-Xptxas=-v',
        '-gencode=arch=compute_120a,code=sm_120a',
    ],
    is_python_module=False,
    verbose=True,
)
module = torch.ops.attention_sm120

#----------------------------------------------------------------------------
# PyTorch baselines.

def torch_naive(q, k, v, causal):
    # q: [N, Hq, Q, D], k/v: [N, Hkv, K, D].
    num_kv_heads = k.shape[1]
    group_size = q.shape[1] // num_kv_heads
    grouped_q = q.reshape(q.shape[0], num_kv_heads, group_size, q.shape[2], q.shape[3])

    # QK^T: N*Hq*Q*K dot products, each with D multiplies and D-1 adds
    #        ~= 2*N*Hq*Q*K*D FLOPs.
    scores = torch.einsum('nhgqd,nhkd->nhgqk', grouped_q, k)

    # One scale per score: N*Hq*Q*K FLOPs.
    scores /= math.sqrt(q.shape[-1])

    if causal:
        # Masking is not counted as FLOPs. This naive implementation has
        # already computed the full Q*K score matrix before applying the mask.
        seq_len = q.shape[-2]
        mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).triu(1)
        scores.masked_fill_(mask, -math.inf)

    # Per score, softmax performs approximately one subtraction, one
    # exponential, one reduction add, and one division. The conventional
    # attention FLOP count excludes softmax and counts only the two matmuls.
    weights = scores.float().softmax(dim=-1).to(q.dtype)

    # WV: ~= 2*N*Hq*Q*K*D FLOPs.
    # Total conventional forward count: 4*N*Hq*Q*K*D FLOPs.
    output = torch.einsum('nhgqk,nhkd->nhgqd', weights, v)
    return output.reshape_as(q)


def fa(q, k, v, causal):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            enable_gqa=q.shape[1] != k.shape[1],
        )


def cudnn(q, k, v, causal):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            enable_gqa=q.shape[1] != k.shape[1],
        )


#----------------------------------------------------------------------------
# Custom kernels.

def attention_v1_fwd(q, k, v, causal):
    return module.attention_v1_fwd(q, k, v, causal)


def attention_v2_fwd(q, k, v, causal):
    return module.attention_v2_fwd(q, k, v, causal)


def attention_v3_fwd(q, k, v, causal):
    return module.attention_v3_fwd(q, k, v, causal)

#----------------------------------------------------------------------------
# Utilities.

def parse_shape(shape):
    parts = shape.split('_')
    parsed = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in parsed):
        raise click.ClickException('--shape dimensions must be positive')

    if len(parsed) == 4:
        batch, q_heads, seq_len, head_dim = parsed
        kv_heads = q_heads
    elif len(parsed) == 5:
        batch, q_heads, kv_heads, seq_len, head_dim = parsed
    else:
        raise click.ClickException(
            '--shape must look like B_H_S_D or B_HQ_HKV_S_D, '
            'for example 4_16_4_4096_128'
        )

    if q_heads % kv_heads != 0:
        raise click.ClickException('The number of query heads must be divisible by the number of KV heads')

    return batch, q_heads, kv_heads, seq_len, head_dim


def make_inputs(batch, q_heads, kv_heads, seq_len, head_dim, requires_grad):
    torch.manual_seed(0)
    q_shape = (batch, q_heads, seq_len, head_dim)
    kv_shape = (batch, kv_heads, seq_len, head_dim)
    q = torch.randn(q_shape, device='cuda', dtype=torch.bfloat16, requires_grad=requires_grad)
    k = torch.randn(kv_shape, device='cuda', dtype=torch.bfloat16, requires_grad=requires_grad)
    v = torch.randn(kv_shape, device='cuda', dtype=torch.bfloat16, requires_grad=requires_grad)
    return q, k, v


def get_kernel(name):
    kernels = {
        'attention_v1_fwd': attention_v1_fwd,
        'attention_v2_fwd': attention_v2_fwd,
        'attention_v3_fwd': attention_v3_fwd,
        'cudnn': cudnn,
        'fa': fa,
        'naive': torch_naive,
    }
    try:
        return kernels[name]
    except KeyError as exc:
        raise click.ClickException(f'Unknown kernel "{name}"') from exc


def run_kernel(name, q, k, v, causal, direction, grad_output):
    if direction == 'forward-backward' and name.startswith('attention_v'):
        raise click.ClickException(f'{name} supports forward only')

    output = get_kernel(name)(q, k, v, causal)
    if direction == 'forward-backward':
        output.backward(grad_output)
    return output


def clear_gradients(q, k, v):
    q.grad = None
    k.grad = None
    v.grad = None


def check_correctness(name, q, k, v, causal):
    if name == 'fa':
        return

    output = get_kernel(name)(q, k, v, causal)
    reference = fa(q, k, v, causal)
    # Different fused reduction orders produce small absolute BF16 errors.
    # Early causal rows average few values and need a larger absolute tolerance.
    simple_atol = 3e-2 if causal else 1e-2
    atol = simple_atol if name == 'naive' or name.startswith('attention_v') else 3e-3
    torch.testing.assert_close(output, reference, rtol=1.6e-2, atol=atol)


def bench_kernel(name, q, k, v, causal, direction, grad_output):
    def run():
        clear_gradients(q, k, v)
        run_kernel(name, q, k, v, causal, direction, grad_output)

    return do_bench(run, return_mode='median')


def profile_kernel(name, q, k, v, causal, direction, grad_output):
    clear_gradients(q, k, v)
    with torch.cuda.nvtx.range(f'{name}_{direction}'):
        run_kernel(name, q, k, v, causal, direction, grad_output)
    torch.cuda.synchronize()


def compute_tflops(
    batch,
    q_heads,
    seq_len,
    head_dim,
    causal,
    direction,
    latency_ms,
    recompute_scores,
):
    # Forward: scores = q @ k.T and output = weights @ v (two matmuls).
    # Fused backward recomputes scores = q @ k.T, then calculates
    #           dv = weights.T @ doutput, dweights = doutput @ v.T,
    #           dq = dscores @ k, and dk = dscores.T @ q (four matmuls).
    # Backward is therefore 2.5x forward, and forward-backward is 3.5x forward.
    # Causal attention has S*(S+1)/2 valid query-key pairs instead of S*S,
    # which approaches half as S grows. This is the effective FLOP count;
    # implementations such as torch_naive may still compute the full matrix.
    causal_factor = 0.5 if causal else 1.0
    backward_factor = 2.5 if recompute_scores else 2.0
    direction_factor = 1.0 + backward_factor if direction == 'forward-backward' else 1.0
    flops = 4 * batch * q_heads * seq_len**2 * head_dim * causal_factor * direction_factor
    return flops / latency_ms / 1e9

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option(
    '--shape',
    help='MHA as B_H_S_D or GQA as B_HQ_HKV_S_D',
    metavar='SHAPE',
    type=str,
    default='8_16_4096_128',
    show_default=True,
)
@click.option(
    '--kernel',
    help='Kernel to benchmark',
    metavar='STR',
    type=str,
    multiple=True,
)
@click.option(
    '--profile',
    help='Kernel to run once inside an NVTX range',
    metavar='STR',
    type=str,
    default=None,
)
@click.option(
    '--direction',
    type=click.Choice(['forward', 'forward-backward']),
    default='forward',
    show_default=True,
)
@click.option('--causal/--non-causal', default=True, show_default=True)
def cmdline(shape, kernel, profile, direction, causal):
    """Benchmark BF16 attention implementations for SM120."""
    if not torch.cuda.is_available():
        raise click.ClickException('CUDA is not available')

    batch, q_heads, kv_heads, seq_len, head_dim = parse_shape(shape)
    requires_grad = direction == 'forward-backward'
    q, k, v = make_inputs(batch, q_heads, kv_heads, seq_len, head_dim, requires_grad)
    grad_output = torch.randn_like(q) if requires_grad else None

    if profile is not None:
        profile_kernel(profile, q, k, v, causal, direction, grad_output)
        return

    normalized_shape = f'{batch}_{q_heads}_{kv_heads}_{seq_len}_{head_dim}'
    print(f'shape: {normalized_shape}, causal: {causal}, direction: {direction}')

    default_kernels = ['fa', 'cudnn']
    if direction == 'forward':
        default_kernels.extend(['attention_v1_fwd', 'attention_v2_fwd', 'attention_v3_fwd'])
    kernels = kernel or default_kernels
    for name in kernels:
        check_correctness(name, q, k, v, causal)
        latency_ms = bench_kernel(name, q, k, v, causal, direction, grad_output)
        tflops = compute_tflops(
            batch,
            q_heads,
            seq_len,
            head_dim,
            causal,
            direction,
            latency_ms,
            recompute_scores=name != 'naive',
        )
        print(f'{name}: {latency_ms:.4f} ms, {tflops:.2f} TFLOPS')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    cmdline()

#----------------------------------------------------------------------------
