import math

import click
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.testing import do_bench

#----------------------------------------------------------------------------
# PyTorch baselines.

def torch_naive(q, k, v, causal):
    scores = torch.einsum('nhqd,nhkd->nhqk', q, k / math.sqrt(q.shape[-1]))

    if causal:
        seq_len = q.shape[-2]
        mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).triu(1)
        scores.masked_fill_(mask, -math.inf)

    weights = scores.float().softmax(dim=-1).to(q.dtype)
    return torch.einsum('nhqk,nhkd->nhqd', weights, v)


def fa(q, k, v, causal):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def cudnn(q, k, v, causal):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

#----------------------------------------------------------------------------
# Utilities.

def parse_shape(shape):
    parts = shape.split('_')
    if len(parts) != 4:
        raise click.ClickException('--shape must look like B_H_S_D, for example 1_16_4096_128')

    parsed = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in parsed):
        raise click.ClickException('--shape dimensions must be positive')
    return parsed


def make_inputs(batch, heads, seq_len, head_dim, requires_grad):
    torch.manual_seed(0)
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, device='cuda', dtype=torch.bfloat16, requires_grad=requires_grad)
    k = torch.randn(shape, device='cuda', dtype=torch.bfloat16, requires_grad=requires_grad)
    v = torch.randn(shape, device='cuda', dtype=torch.bfloat16, requires_grad=requires_grad)
    return q, k, v


def get_kernel(name):
    kernels = {
        'cudnn': cudnn,
        'fa': fa,
        'naive': torch_naive,
    }
    try:
        return kernels[name]
    except KeyError as exc:
        raise click.ClickException(f'Unknown kernel "{name}"') from exc


def run_kernel(name, q, k, v, causal, direction, grad_output):
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
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)


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


def compute_tflops(batch, heads, seq_len, head_dim, causal, direction, latency_ms):
    # Forward consists of q @ k.T and softmax(q @ k.T) @ v. Backward adds
    # four matrix multiplications of the same size. Causal attention evaluates
    # approximately half of the full attention matrix.
    causal_factor = 0.5 if causal else 1.0
    direction_factor = 3.0 if direction == 'forward-backward' else 1.0
    flops = 4 * batch * heads * seq_len**2 * head_dim * causal_factor * direction_factor
    return flops / latency_ms / 1e9

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option(
    '--shape',
    help='Attention shape as batch_heads_sequence_head-dimension',
    metavar='B_H_S_D',
    type=str,
    default='1_16_4096_128',
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

    batch, heads, seq_len, head_dim = parse_shape(shape)
    requires_grad = direction == 'forward-backward'
    q, k, v = make_inputs(batch, heads, seq_len, head_dim, requires_grad)
    grad_output = torch.randn_like(q) if requires_grad else None

    if profile is not None:
        profile_kernel(profile, q, k, v, causal, direction, grad_output)
        return

    print(f'shape: {shape}, causal: {causal}, direction: {direction}')

    kernels = kernel or ['fa', 'cudnn']
    for name in kernels:
        check_correctness(name, q, k, v, causal)
        latency_ms = bench_kernel(name, q, k, v, causal, direction, grad_output)
        tflops = compute_tflops(batch, heads, seq_len, head_dim, causal, direction, latency_ms)
        print(f'{name}: {latency_ms:.4f} ms, {tflops:.2f} effective TFLOPS')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    cmdline()

#----------------------------------------------------------------------------
