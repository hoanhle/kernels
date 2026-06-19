import click
import torch
from triton.testing import do_bench

#----------------------------------------------------------------------------
# Utilities.

def parse_shape(shape):
    parts = shape.split('_')
    if len(parts) != 3:
        raise click.ClickException('--shape must look like M_N_K, for example 4096_4096_4096')
    return tuple(int(part) for part in parts)


def make_inputs(M, N, K):
    scale = K**-0.5
    A = torch.randn(M, K, device='cuda').mul(scale).bfloat16()
    B = torch.randn(N, K, device='cuda').mul(scale).bfloat16().T
    return A.contiguous(), B


def get_kernel(name):
    if name == 'cublas':
        return torch.mm
    raise click.ClickException(f'Unknown kernel "{name}"')


def compute_tflops(M, N, K, latency_ms):
    return 2 * M * N * K / latency_ms / 1e9


def bench_triton(name, A, B):
    f = get_kernel(name)
    return do_bench(lambda: f(A, B), return_mode='median')


def profile_kernel(name, A, B):
    f = get_kernel(name)

    for _ in range(5):
        f(A, B)
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range(name):
        f(A, B)
    torch.cuda.synchronize()

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--shape',   help='Matmul shape as M_N_K',                        metavar='M_N_K', type=str, default='4096_4096_4096', show_default=True)
@click.option('--profile', help='Kernel to run once inside an NVTX range',      metavar='STR',   type=str, default=None)
def cmdline(shape, profile):
    """Benchmark BF16 matmul kernels for SM120."""
    M, N, K = parse_shape(shape)
    A, B = make_inputs(M, N, K)

    if profile is not None:
        profile_kernel(profile, A, B)
        return

    kernel = 'cublas'
    latency_ms = bench_triton(kernel, A, B)
    tflops = compute_tflops(M, N, K, latency_ms)

    print(f'shape: {shape}')
    print(f'kernel: {kernel}')
    print(f'latency: {latency_ms:.4f} ms')
    print(f'TFLOPS: {tflops:.2f}')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    cmdline()

#----------------------------------------------------------------------------
