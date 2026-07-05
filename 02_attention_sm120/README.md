# Attention SM120

## Setup

The benchmark protocol follows FlashAttention-4: BF16 inputs, 32K total tokens,
and query hidden dimension 2048. Batch size and query-head count are derived
from sequence length and head dimension:

```text
B = 32768 / S
Hq = 2048 / D
```

The default shape is B=8, Hq=16, S=4096, D=128.

The tensor layouts are:

```text
q:   [batch, query heads, sequence, head dimension]
k/v: [batch, KV heads, sequence, head dimension]
```

MHA uses the same number of query and KV heads. GQA uses fewer KV heads and
requires the number of query heads to be divisible by the number of KV heads.

PyTorch 2.9.1, CUDA 12.8, and cuDNN 9.10.2.

NVIDIA reports a theoretical dense BF16 Tensor Core peak of 209.5 TFLOPS with
FP32 accumulation at the 2.407 GHz GPU Boost Clock. Measurements use the
default 575 W power limit with the SM clock locked to 2.407 GHz.

## Results

Forward BF16 MHA with B=8, Hq=Hkv=16, S=4096, and D=128.

### Causal

| Kernel | TFLOPS | % of theoretical peak |
|:-------|-------:|----------------------:|
| F.sdpa() (Flash Attention) | 166.21 | 79.34% |
| F.sdpa() (CuDNN) | 188.06 | 89.77% |

### Non-causal

| Kernel | TFLOPS | % of theoretical peak |
|:-------|-------:|----------------------:|
| F.sdpa() (Flash Attention) | 177.16 | 84.56% |
| F.sdpa() (CuDNN) | 197.27 | 94.16% |

## Running

Benchmark the FlashAttention and cuDNN backends from the repository root:

```bash
python 02_attention_sm120/main.py --causal
python 02_attention_sm120/main.py --non-causal
```

The four-field shape syntax is an MHA shorthand:

```text
B_H_S_D
```

GQA uses separate query and KV head counts:

```text
B_HQ_HKV_S_D
```

For example, this runs 16 query heads grouped over four KV heads:

```bash
python 02_attention_sm120/main.py \
    --shape 8_16_4_4096_128 \
    --non-causal
```

At sequence length 4096, the two FA4-style MHA shapes are:

```text
D=128: 8_16_4096_128
D=64:  8_32_4096_64
```

The suggested GQA shape above keeps the same batch size, query head count,
sequence length, and head dimension while reducing the KV heads.

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
  backend and enables GQA when the query and KV head counts differ.
- `cudnn` uses PyTorch scaled dot-product attention with the cuDNN backend.
  The installed PyTorch 2.9.1 and cuDNN 9.10.2 stack supports the tested GQA
  shapes.

## FLOP accounting

For non-causal forward attention, the conventional count includes the two
matrix multiplications:

```text
scores = q @ k.T       ~= 2 * B * Hq * S^2 * D FLOPs
output = weights @ v   ~= 2 * B * Hq * S^2 * D FLOPs
total                  ~= 4 * B * Hq * S^2 * D FLOPs
```

Causal attention has `S * (S + 1) / 2` valid query-key pairs, which approaches
half of `S^2`. The benchmark therefore uses half the non-causal FLOP count when
reporting causal TFLOPS. GQA changes K/V storage and reuse, but every query head
still performs attention, so the conventional FLOP count depends on `Hq`, not
`Hkv`.

Fused backward performs five matrix multiplications:

```text
scores   = q @ k.T                 # recompute
dv       = weights.T @ doutput
dweights = doutput @ v.T
dq       = dscores @ k
dk       = dscores.T @ q
```

Fused backward recomputes the scores instead of saving the quadratic attention
matrix, so it performs five matmuls: 2.5 times the forward count. A combined
forward-backward benchmark therefore uses 3.5 times the forward count. Softmax,
masking, scaling, and other elementwise operations are executed but excluded
from the reported FLOP count.

## Resources

- [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [FlashAttention-4 paper](https://github.com/Dao-AILab/flash-attention/blob/main/assets/fa4_paper.pdf)
