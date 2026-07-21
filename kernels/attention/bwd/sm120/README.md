# Attention Backward SM120

## Setup

The backward baseline uses BF16 MHA with B=8, Hq=Hkv=16, S=4096, and D=128.
PyTorch 2.9.1, CUDA 12.8, and cuDNN 9.10.2.

NVIDIA reports a theoretical dense BF16 Tensor Core peak of 209.5 TFLOPS with
FP32 accumulation at the 2.407 GHz GPU Boost Clock.

## Results

### Causal

Forward graph construction and gradient clearing are excluded from timing.

| Kernel | Latency | TFLOPS | % of theoretical peak |
|:-------|--------:|-------:|----------------------:|
| F.sdpa() (Flash Attention) | 8.3765 ms | 164.08 | 78.32% |
| F.sdpa() (CuDNN) | 9.4198 ms | 145.90 | 69.64% |

## Running

Benchmark only the backward pass from the repository root:

```bash
python kernels/attention/main.py \
    --direction backward \
    --causal
```

This reports the `fa` and `cudnn` baselines. The forward pass builds each
backend's autograd graph before `do_bench`, so only backward is timed.

## Reproducible clocks and power

Lock the SM clock to the 2.407 GHz clock used for the theoretical peak and set
the power limit explicitly to 575 W. The power limit is a ceiling; verify that
the GPU holds the requested clock rather than power-throttling during the run.

The subshell restores the previous power limit and dynamic clocking when the
benchmark finishes or exits:

```bash
sudo -v
(
    set -e
    previous_power_limit="$(nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader,nounits)"
    cleanup() {
        set +e
        sudo -n nvidia-smi -i 0 -rgc
        sudo -n nvidia-smi -i 0 -pl "$previous_power_limit"
    }
    trap cleanup EXIT

    sudo -n nvidia-smi -i 0 -pl 575
    sudo -n nvidia-smi -i 0 -lgc 2407,2407

    python kernels/attention/main.py --direction backward --causal
)
```
