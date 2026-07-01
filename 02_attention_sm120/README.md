# Attention SM120

## Setup

Fixed B=1, H=16, S=4096, D=128 with BF16 inputs. The RTX 5090 has a
theoretical peak of 209.5 BF16 TFLOPS.

`q`, `k`, and `v` use the `[batch, heads, sequence, head dimension]` layout.

PyTorch 2.9.1, CUDA 12.8, and cuDNN 9.10.2:

### Causal

| Kernel | TFLOPS | % of theoretical peak |
|:-------|-------:|----------------------:|
| PyTorch SDPA, FlashAttention backend | 144.01 | 68.74% |
| PyTorch SDPA, cuDNN backend | 167.65 | 80.02% |

### Non-causal

| Kernel | TFLOPS | % of theoretical peak |
|:-------|-------:|----------------------:|
| PyTorch SDPA, FlashAttention backend | 163.78 | 78.18% |
| PyTorch SDPA, cuDNN backend | 197.30 | 94.18% |

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
