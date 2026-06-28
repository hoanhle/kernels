from pathlib import Path

import click
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

CURRENT_DIR = Path(__file__).resolve().parent

torch.utils.cpp_extension.load(
    'matmul_sm120',
    sources=[
        CURRENT_DIR / 'matmul.cpp',
        *sorted(CURRENT_DIR.glob('matmul_v*.cu')),
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
module = torch.ops.matmul_sm120

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
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise click.ClickException(f'Unknown kernel "{name}"') from exc


def compute_tflops(M, N, K, latency_ms):
    return 2 * M * N * K / latency_ms / 1e9


def check_correctness(name, A, B):
    if name == 'cublas':
        return
    out = get_kernel(name)(A, B)
    ref = torch.mm(A.float(), B.float()).bfloat16()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def bench_triton(name, A, B):
    f = get_kernel(name)
    return do_bench(lambda: f(A, B), return_mode='median')


def profile_kernel(name, A, B):
    f = get_kernel(name)

    with torch.cuda.nvtx.range(name):
        f(A, B)
    torch.cuda.synchronize()

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--shape',   help='Matmul shape as M_N_K',                        metavar='M_N_K', type=str, default='4096_4096_4096', show_default=True)
@click.option('--kernel',  help='Kernel to benchmark',                          metavar='STR',   type=str, multiple=True)
@click.option('--profile', help='Kernel to run once inside an NVTX range',      metavar='STR',   type=str, default=None)
def cmdline(shape, kernel, profile):
    """Benchmark BF16 matmul kernels for SM120."""
    M, N, K = parse_shape(shape)
    A, B = make_inputs(M, N, K)

    if profile is not None:
        profile_kernel(profile, A, B)
        return

    print(f'shape: {shape}')

    kernels = kernel or [
        'cublas',
        'matmul_v0_naive',
        'matmul_v0_tiled',
        'matmul_v0_block1d',
        'matmul_v0_block2d',
        'matmul_v1_mma_tiled',
        'matmul_v2_cp_async',
        'matmul_v2_cp_async_double_buffered',
        'matmul_v2_cp_async_double_buffered_swizzled',
    ]
    for name in kernels:
        check_correctness(name, A, B)
        latency_ms = bench_triton(name, A, B)
        tflops = compute_tflops(M, N, K, latency_ms)
        print(f'{name}: {latency_ms:.4f} ms, {tflops:.2f} TFLOPS')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    cmdline()

#----------------------------------------------------------------------------
