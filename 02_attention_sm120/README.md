# Attention SM120

## Setup

Fixed B=4, H=16, S=4096, D=128 with BF16 inputs.

`q`, `k`, and `v` use the `[batch, heads, sequence, head dimension]` layout.

PyTorch 2.9.1, CUDA 12.8, and cuDNN 9.10.2:

The RTX 5090 has 170 SMs with four Tensor Cores per SM. Its theoretical dense
BF16 Tensor Core peak with FP32 accumulation is:

```text
170 SMs
* 4 Tensor Cores/SM
* 128 BF16 FLOPs/Tensor Core/cycle
* 2.407 billion cycles/second
= 209.5 TFLOPS
```

An FMA counts as two FLOPs. NVIDIA states that its peak rates are based on the
GPU Boost Clock. The measurements below used the default 575 W power limit with
the SM clock locked to the rated 2.407 GHz boost clock.

### Causal

| Kernel | TFLOPS | % of theoretical peak |
|:-------|-------:|----------------------:|
| PyTorch SDPA, FlashAttention backend | 157.78 | 75.31% |
| PyTorch SDPA, cuDNN backend | 179.83 | 85.84% |

### Non-causal

| Kernel | TFLOPS | % of theoretical peak |
|:-------|-------:|----------------------:|
| PyTorch SDPA, FlashAttention backend | 170.06 | 81.17% |
| PyTorch SDPA, cuDNN backend | 192.72 | 91.99% |

## Running

Benchmark the FlashAttention and cuDNN backends from the repository root:

```bash
python 02_attention_sm120/main.py --causal
python 02_attention_sm120/main.py --non-causal
```

Run selected implementations:

```bash
python 02_attention_sm120/main.py \
    --kernel fa \
    --kernel cudnn \
    --non-causal
```

Run the explicit PyTorch implementation:

```bash
python 02_attention_sm120/main.py --kernel naive --non-causal
```

Benchmark the combined forward and backward pass:

```bash
python 02_attention_sm120/main.py \
    --direction forward-backward \
    --non-causal
```

### Reproducible clocks

For more reproducible measurements, lock the SM clock to the 2.407 GHz clock
used by NVIDIA to calculate the rated peak. Keep the default 575 W power limit
so that the GPU is not power-throttled below the requested clock.

The subshell restores dynamic clocking when the benchmark finishes or exits:

```bash
sudo -v
(
    trap 'sudo -n nvidia-smi -i 0 -rgc' EXIT
    sudo -n nvidia-smi -i 0 -lgc 2407,2407

    python 02_attention_sm120/main.py --non-causal
)
```

## Baselines

- `naive` materializes the full attention-score matrix and uses two dense
  `torch.einsum` operations. Its causal path masks scores after computing the
  complete matrix, so it does not skip the upper triangle.
- `fa` uses PyTorch scaled dot-product attention with the FlashAttention
  backend.
- `cudnn` uses PyTorch scaled dot-product attention with the cuDNN backend.

## FLOP accounting

For non-causal forward attention, the conventional count includes the two
matrix multiplications:

```text
scores = q @ k.T       ~= 2 * B * H * S^2 * D FLOPs
output = weights @ v   ~= 2 * B * H * S^2 * D FLOPs
total                  ~= 4 * B * H * S^2 * D FLOPs
```

Causal attention has `S * (S + 1) / 2` valid query-key pairs, which approaches
half of `S^2`. The benchmark therefore uses half the non-causal FLOP count when
reporting causal TFLOPS.

Backward adds four matrix multiplications:

```text
dv       = weights.T @ doutput
dweights = doutput @ v.T
dq       = dscores @ k
dk       = dscores.T @ q
```

Backward is approximately twice the forward FLOP count, so a combined
forward-backward benchmark uses three times the forward count. Softmax,
masking, scaling, and other elementwise operations are executed but excluded
from the reported FLOP count.

## Resources

- [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
