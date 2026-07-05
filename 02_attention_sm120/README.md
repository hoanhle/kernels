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
| F.sdpa() (Flash Attention) | 166.10 | 79.28% |
| F.sdpa() (CuDNN) | 187.35 | 89.43% |
| v1 (online softmax, tiling, mma, cp.async) | 128.16 | 61.17% |
| v2 (shared-memory swizzling) | 156.80 | 74.84% |
| v3 (two-stage pipeline) | 164.13 | 78.34% |
| v4 (TMA and mbarrier) | 165.70 | 79.09% |

### Non-causal

| Kernel | TFLOPS | % of theoretical peak |
|:-------|-------:|----------------------:|
| F.sdpa() (Flash Attention) | 177.01 | 84.49% |
| F.sdpa() (CuDNN) | 197.42 | 94.23% |
| v1 (online softmax, tiling, mma, cp.async) | 143.61 | 68.55% |
| v2 (shared-memory swizzling) | 170.79 | 81.52% |
| v3 (two-stage pipeline) | 177.98 | 84.95% |
| v4 (TMA and mbarrier) | 182.38 | 87.05% |

## Running

Benchmark the custom kernel and PyTorch baselines from the repository root:

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

Compare the custom kernels:

```bash
python 02_attention_sm120/main.py \
    --kernel attention_v1_fwd \
    --kernel attention_v2_fwd \
    --kernel attention_v3_fwd \
    --kernel attention_v4_fwd \
    --kernel attention_v5_fwd \
    --non-causal
```

Benchmark the combined forward and backward pass:

```bash
python 02_attention_sm120/main.py \
    --direction forward-backward \
    --non-causal
```

### Profiling

See the [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/pdf/ProfilingGuide.pdf)
for NVIDIA's hardware model and metric definitions.

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

## v1

v1 implements forward BF16 attention with `mma.sync`, 16-byte `cp.async`
copies, and an online softmax. Each block owns 128 query rows and walks through
32-row K/V tiles. Scores and output accumulators stay FP32 in registers; scores
are never written to global memory. Exponentiated scores are converted to BF16
for the weight-times-value MMA and normalized after the final K/V tile.

The same kernel supports MHA and GQA without duplicating K/V:
`kv_head = query_head / (Hq / Hkv)`. The causal specialization skips tiles
above the diagonal and masks the diagonal tile.

v1 supports `D=128` and sequence lengths divisible by 128.

## v2

v2 keeps v1's tile sizes and computation but swizzles Q, K, and V in shared
memory. The `cp.async` stores and `ldmatrix` loads use the same swizzled
addresses, eliminating v1's shared-memory bank conflicts.

## v3

v3 adds a two-stage `cp.async` pipeline to v2. Each stage holds one 32-row K
tile and its corresponding V tile. While the block computes attention using
one stage, it loads the next K/V pair into the other stage. The pipeline reuses
the 32 KiB shared-memory allocation used to stage Q after Q has moved into
registers, so the shared-memory footprint remains unchanged.

## v4

v4 replaces v3's per-thread K/V copies with TMA loads issued by one elected
lane. Each 256-byte K/V row is loaded as two 128-byte panels using hardware
swizzling, and one `mbarrier` tracks completion of each 16 KiB pipeline stage.
Q retains its one-time `cp.async` load.

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
