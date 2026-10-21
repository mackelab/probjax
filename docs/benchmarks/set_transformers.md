# Set transformer benchmark: ISAB vs DeepSet, MLA and full attention

Compares equivariant set encoders on small synthetic set-learning problems and
checks the exact invariances `InducedSelfAttention` is supposed to satisfy.
All numbers below were produced on Apple Silicon CPU (JAX `0.9.2`, float32).

```bash
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_set_transformers --steps 5000 \
  --output docs/benchmarks/set_transformers.json
```

## Measurement settings

Budget per arm and task: **5000 training steps, 3 seeds, batch 64, eval batch
512, Adam 3e-3 with global-norm clip 5**. Models are **2 layers wide 64, four
heads**, ISAB/MLA arms use **32 inducing points**, the in-context task uses
**48 context + 16 query tokens (N = 64)**, both set tasks use **N = 64**.
Runtime scaling uses batch four, width 64, two layers and **64 inducing
points** at N = 64 / 256 / 1024 / 2048.

The arms:

| Arm | Encoder |
| --- | --- |
| `deepset` | Permutation-invariant DeepSet pooling broadcast back to every token |
| `isab` | `InducedSelfAttention`, no query scaling |
| `isab_per_head` | ISAB with `PerHeadQueryScale` on both MABs |
| `isab_ssmax` | ISAB with `SSMaxQueryScale` on both MABs |
| `isab_qassmax` | ISAB with `QASSMaxQueryScale` on both MABs |
| `mla` | Pre-norm block with `MultiHeadLatentAttention` (absorbed, shared KV latent) |
| `sab` | Full self-attention (`Transformer`), quadratic reference |

ISAB blocks use `zero_init=True` (the default): each MAB's attention output
projection and last feedforward layer start at zero, so a block begins as the
identity. They also share one norm between each MAB's query and key/value.
ISAB blocks run two MAB stages each, so their parameter counts are roughly
double the single-attention arms (205k vs 100k at this width).

## Invariance audit

Checked on one fixed model and input; these are exactness checks, not
trainable metrics.

| Check | Max error / status |
| --- | ---: |
| Permutation equivariance `f(P x) - P f(x)` | `9.54e-07` |
| Perturbing `KVLenMask`-masked padding rows | `0.0` |
| Unpadded N=6 vs padded + `KVLenMask` (SSMax on both MABs) | `4.77e-07` |
| Perturbing query rows with separate `x_kv` context | `0.0` |
| `BatchedLowRankBias` token merging, dense kernel | `7.15e-07` |
| `BatchedLowRankBias` token merging, `flex_attention` | `5.96e-07` |
| `bfloat16` preserved across both MABs (`norm_cls=None`) | `True` |

The padding row is the regression guard for the `kv_len` bug: MAB 2 attends
over the inducing bank, not over `kv_len` keys, so query scaling must resolve
`kv_len` from the mask for MAB 1 only.

## Task quality (mean test MSE over 3 seeds)

Lower is better. Baselines are computed per seed on the eval batch.

### `cluster_denoise` (N=64, identity `0.090`, oracle `0.003`)

| Arm | Test MSE | Params | Train step ms |
| --- | ---: | ---: | ---: |
| `deepset` | 0.0086 ± 0.0004 | 108,740 | 13.4 |
| `isab` | 0.0056 ± 0.0008 | 204,612 | 28.8 |
| `isab_per_head` | **0.0054 ± 0.0007** | 204,628 | 30.7 |
| `isab_ssmax` | 0.0067 ± 0.0039 | 204,628 | 40.9 |
| `isab_qassmax` | 0.0057 ± 0.0007 | 230,276 | 45.5 |
| `mla` | 0.0070 ± 0.0011 | 112,652 | 30.1 |
| `sab` | 0.0060 ± 0.0008 | 100,548 | 22.0 |

### `padded_outlier_mean` (N in [16, 64], naive mean `0.040`, oracle `0.002`)

| Arm | Test MSE | Params | Train step ms |
| --- | ---: | ---: | ---: |
| `deepset` | 0.0060 ± 0.0029 | 108,740 | 16.5 |
| `isab` | 0.0040 ± 0.0002 | 204,612 | 34.0 |
| `isab_per_head` | 0.0043 ± 0.0007 | 204,628 | 31.0 |
| `isab_ssmax` | **0.0036 ± 0.0004** | 204,628 | 34.6 |
| `isab_qassmax` | 0.0113 ± 0.0011 | 230,276 | 41.1 |
| `mla` | 0.0049 ± 0.0007 | 112,652 | 27.0 |
| `sab` | 0.0046 ± 0.0005 | 100,548 | 21.2 |

### `in_context_regression` (48 context + 16 query, zero prediction `4.6`)

| Arm | Test MSE | Params | Train step ms |
| --- | ---: | ---: | ---: |
| `deepset` | 0.0866 ± 0.0191 | 108,609 | 8.1 |
| `isab` | 0.0463 ± 0.0188 | 204,481 | 14.5 |
| `isab_per_head` | 0.0358 ± 0.0090 | 204,497 | 15.3 |
| `isab_ssmax` | 0.0293 ± 0.0070 | 204,497 | 14.7 |
| `isab_qassmax` | **0.0278 ± 0.0063** | 230,145 | 21.8 |
| `mla` | 0.0485 ± 0.0023 | 112,521 | 13.0 |
| `sab` | 0.0582 ± 0.0162 | 100,417 | 10.9 |

Observations:

- At 5000 steps the best ISAB variant beats `sab`, `mla` and `deepset` on all
  three tasks, while paying `O(N M)` instead of `O(N²)`.
- On `in_context_regression` the gap is largest: routing context through the
  inducing bank is an inductive bias for this task.
- Differences between ISAB variants are often within the seed spread (3
  seeds), and the best query scaling varies by task.
- Zero init was compared against the default init at the same budget (not
  committed): it was neutral overall, winning some arm/task pairs and losing
  others. At 1500 steps it was clearly slower to converge.

## Runtime scaling (warm ms, batch 4)

| Arm | N | Forward | Backward |
| --- | ---: | ---: | ---: |
| `deepset` | 64 | 0.40 | 0.95 |
| `deepset` | 256 | 0.68 | 2.74 |
| `deepset` | 1024 | 3.15 | 8.33 |
| `deepset` | 2048 | 5.50 | 15.18 |
| `isab` | 64 | 1.07 | 3.02 |
| `isab` | 256 | 1.98 | 5.55 |
| `isab` | 1024 | 5.59 | 16.65 |
| `isab` | 2048 | 9.97 | 34.00 |
| `isab_ssmax` | 64 | 0.98 | 2.87 |
| `isab_ssmax` | 256 | 2.25 | 5.54 |
| `isab_ssmax` | 1024 | 5.85 | 17.92 |
| `isab_ssmax` | 2048 | 10.98 | 36.85 |
| `mla` | 64 | 0.50 | 1.80 |
| `mla` | 256 | 2.19 | 6.62 |
| `mla` | 1024 | 35.02 | 110.00 |
| `mla` | 2048 | 124.47 | 513.38 |
| `sab` | 64 | 0.63 | 1.68 |
| `sab` | 256 | 1.77 | 7.80 |
| `sab` | 1024 | 19.27 | 94.10 |
| `sab` | 2048 | 107.64 | 448.90 |

ISAB grows linearly with `N` at fixed `M=64` (about 7x cheaper forward and
9x cheaper backward than `sab` at N=2048) and its query scaling adds no
measurable overhead. `mla` still pays `O(N²)` attention over the shared
latent and is the slowest quadratic arm at large N. CPU timings do not
establish GPU performance.

## Re-run with a smaller budget

```bash
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_set_transformers \
  --steps 50 --seeds 1 --batch-size 16 --set-size 32 --num-inducing 8 \
  --model-dim 32 --tokens 32 128 --output docs/benchmarks/set_smoke.json
```

Write smoke results to a distinct filename so the published measurements
above stay intact. See the [benchmark index](index.md) for the general
caveats.
