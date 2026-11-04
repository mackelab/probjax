# GPU benchmarks for probjax.nn models

End-to-end forward and train-step cost of the `probjax.nn` models before and after the kernel work, plus the new `scan_layers` / `remat_policy` options. The before/after tables run identical code paths of the benchmark on the pre-optimization base and on the merged branch ([`gpu_nn_before.json`](gpu_nn_before.json) → [`gpu_nn.json`](gpu_nn.json)); the scan/remat, memory and CPU tables come from [`kernel_nn_gpu.json`](kernel_nn_gpu.json) and [`kernel_nn_cpu.json`](kernel_nn_cpu.json). Kernel-level numbers are on the [attention](gpu_attention.md) and [kernel](gpu_kernels.md) pages; methodology is on the [overview](gpu_overview.md). Times are medians in milliseconds; a train step is `value_and_grad` of a scalar loss plus an optimizer update. OOM: out of device memory. fail†: XLA could not profile any kernel configuration at this size during autotuning (reported as `NOT_FOUND`), so the case did not run. unsup.: the implementation rejects the shape or backend. —: not measured.

## Transformer and DiffusionTransformer

`Transformer` with the standard profile (model width 256, 8 heads of 32, 4 layers), batch 8, one column per `attention_fn`. Only the default column changed in code: `dot_product_attention` now dispatches to `flex_attention` on NVIDIA GPUs (rules on the [attention page](gpu_attention.md#default-attention_fn-dispatch)). The other columns are controls and should stay put.

| dtype | tokens | op | default `dot_product_attention` | `flex_attention` | `jax.nn` XLA | `jax.nn` cuDNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1024 | forward | 5.60 → 2.25 | 2.21 → 2.22 | 5.63 → 5.60 | 3.08 → 2.82 |
| bf16 | 1024 | train step | 17.8 → 7.55 | 8.18 → 7.59 | 19.0 → 17.7 | 10.5 → 9.73 |
| bf16 | 4096 | forward | 75.7 → 14.8 | 14.8 → 14.8 | 66.6 → 66.6 | 24.1 → 24.0 |
| bf16 | 4096 | train step | OOM → 59.8 | 60.3 → 59.0 | OOM → OOM | 87.1 → 87.4 |
| fp32 | 1024 | forward | 8.25 → 4.18 | 4.21 → 4.18 | 8.19 → 8.20 | — → — |
| fp32 | 1024 | train step | 24.7 → 16.1 | 15.9 → 15.8 | 24.7 → 24.3 | — → — |
| fp32 | 4096 | forward | OOM → 28.0 | 29.6 → 28.5 | OOM → OOM | — → — |
| fp32 | 4096 | train step | OOM → 136 | 134 → 136 | OOM → OOM | — → — |

`DiffusionTransformer`, same profile:

| dtype | tokens | op | default `dot_product_attention` | `flex_attention` | `jax.nn` XLA | `jax.nn` cuDNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1024 | forward | 5.69 → 2.35 | 2.47 → 2.36 | 5.67 → 5.68 | 3.08 → 3.05 |
| bf16 | 1024 | train step | 19.1 → 8.57 | 8.91 → 8.37 | 19.4 → 18.9 | 11.4 → 10.8 |
| bf16 | 4096 | forward | 66.9 → 15.0 | 15.0 → 15.0 | 67.1 → 66.9 | 22.5 → 22.7 |
| bf16 | 4096 | train step | OOM → 61.9 | 62.9 → 61.4 | OOM → OOM | 88.0 → 89.7 |
| fp32 | 1024 | forward | 8.27 → 4.24 | 4.28 → 4.25 | 8.28 → 8.27 | — → — |
| fp32 | 1024 | train step | 25.7 → 17.2 | 17.1 → 17.2 | 25.8 → 25.7 | — → — |
| fp32 | 4096 | forward | OOM → 28.5 | 28.9 → 28.2 | OOM → OOM | — → — |
| fp32 | 4096 | train step | OOM → 138 | 136 → 138 | OOM → OOM | — → — |

??? note "256 tokens"

    **Transformer**

    | dtype | tokens | op | default `dot_product_attention` | `flex_attention` | `jax.nn` XLA | `jax.nn` cuDNN |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | bf16 | 256 | forward | 0.737 → 0.525 | 0.535 → 0.531 | 0.765 → 0.739 | 0.574 → 0.581 |
    | bf16 | 256 | train step | 3.44 → 2.65 | 2.66 → 2.43 | 3.42 → 3.17 | 2.65 → 2.98 |
    | fp32 | 256 | forward | 1.15 → 0.908 | 0.892 → 0.908 | 1.18 → 1.18 | — → — |
    | fp32 | 256 | train step | 4.26 → 3.83 | 3.79 → 3.54 | 4.16 → 4.20 | — → — |

    **DiffusionTransformer**

    | dtype | tokens | op | default `dot_product_attention` | `flex_attention` | `jax.nn` XLA | `jax.nn` cuDNN |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | bf16 | 256 | forward | 0.836 → 0.626 | 0.628 → 0.625 | 0.836 → 0.835 | 0.680 → 0.675 |
    | bf16 | 256 | train step | 3.96 → 3.03 | 3.29 → 3.29 | 3.98 → 3.94 | 3.68 → 3.63 |
    | fp32 | 256 | forward | 1.27 → 0.995 | 1.01 → 0.996 | 1.25 → 1.26 | — → — |
    | fp32 | 256 | train step | 5.32 → 4.44 | 4.14 → 4.18 | 4.87 → 4.69 | — → — |

With the new default, the bf16 train step at 1024 tokens drops from 17.8 to 7.55 ms and the 4096-token train step, which ran out of memory before, takes 59.8 ms. At this head size (32) the default is also faster than cuDNN at 4096 tokens: forward 14.8 vs 24.0 ms, train step 59.8 vs 87.4 ms. Compile times are in the JSON (`compile_s`).

The same comparison, repeated three times on each side on the attention lane ([`kernel_attn_gpu_transformer.json`](kernel_attn_gpu_transformer.json), default `attention_fn` only, median of the three runs):

| dtype | tokens | op | before (ms) | after (ms) |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 256 | forward | 0.740 | 0.529 |
| bf16 | 256 | train step | 3.27 | 2.40 |
| bf16 | 1024 | forward | 5.57 | 2.24 |
| bf16 | 1024 | train step | 17.7 | 7.70 |
| bf16 | 4096 | forward | 66.5 | 14.8 |
| bf16 | 4096 | train step | OOM | 60.2 |
| fp32 | 256 | forward | 1.15 | 0.910 |
| fp32 | 256 | train step | 4.09 | 3.74 |
| fp32 | 1024 | forward | 8.21 | 4.15 |
| fp32 | 1024 | train step | 24.2 | 16.2 |
| fp32 | 4096 | forward | OOM | 28.1 |
| fp32 | 4096 | train step | OOM | 135 |

## Scan over layers and rematerialization

`Transformer` and `DiffusionTransformer` take `remat_policy=None | 'full' | 'nothing_saveable' | <jax.checkpoint_policies name> | callable` and `scan_layers=False`. `scan_layers=True` stacks the layer parameters and runs the blocks under `jax.lax.scan`; it changes the parameter tree, so existing checkpoints do not load into a scanned model. Both are opt-in.

**GPU**, 24 layers, width 512, 4096 tokens, batch 2, bf16, `flex_attention` (device peak measured in a fresh process):

| `scan_layers` | `remat_policy` | train step (ms) | compile (s) | XLA temp (MB) | device peak (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| no | none | 212 | 9.46 | 3,460 | 5,212 |
| no | dots_saveable | 248 | 7.13 | 1,808 | 3,560 |
| no | full | 260 | 7.32 | 320 | 2,072 |
| yes | none | 248 | 1.83 | 7,373 | OOM |
| yes | full | 257 | 1.07 | 235 | 2,126 |

Same with the attention that was the default when this was measured (the dense einsum; it now dispatches to the kernel):

| `scan_layers` | `remat_policy` | train step (ms) | compile (s) | XLA temp (MB) | device peak (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| no | none | OOM | — | — | OOM |
| no | dots_saveable | OOM | — | — | OOM |
| no | full | 492 | 8.69 | 2,808 | 4,560 |
| yes | none | OOM | — | — | OOM |
| yes | full | 489 | 1.22 | 2,769 | 4,661 |

`DiffusionTransformer`, 24 layers, `flex_attention`:

| `scan_layers` | `remat_policy` | train step (ms) | compile (s) | XLA temp (MB) | device peak (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| no | none | 229 | 16.9 | 3,848 | 6,894 |
| no | dots_saveable | 263 | 13.0 | 1,824 | 4,870 |
| no | full | 274 | 13.8 | 328 | 3,374 |
| yes | none | 257 | 2.90 | 8,081 | OOM |
| yes | full | 265 | 1.55 | 204 | 3,385 |

??? note "12 layers, dense attention"

    | `scan_layers` | `remat_policy` | train step (ms) | compile (s) | XLA temp (MB) | device peak (MB) |
    | --- | --- | ---: | ---: | ---: | ---: |
    | no | none | OOM | — | — | OOM |
    | no | dots_saveable | OOM | — | — | OOM |
    | no | full | 245 | 4.10 | 2,712 | 3,599 |
    | yes | none | OOM | — | — | OOM |
    | yes | full | 245 | 1.22 | 2,673 | 3,619 |

**CPU**, width 512, 512 tokens, batch 2, fp32, default attention:

24 layers:

| `scan_layers` | `remat_policy` | train step (ms) | compile (s) | XLA temp (MB) | device peak (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| no | none | 1,755 | 3.19 | 2,042 | — |
| no | dots_saveable | 1,541 | 3.06 | 670 | — |
| no | full | 1,505 | 3.07 | 124 | — |
| yes | none | 2,110 | 0.52 | 2,130 | — |
| yes | full | 1,611 | 0.45 | 26.0 | — |

12 layers:

| `scan_layers` | `remat_policy` | train step (ms) | compile (s) | XLA temp (MB) | device peak (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| no | none | 881 | 1.56 | 1,034 | — |
| no | dots_saveable | 785 | 1.42 | 382 | — |
| no | full | 761 | 1.40 | 124 | — |
| yes | none | 1,035 | 0.49 | 1,101 | — |
| yes | full | 811 | 0.44 | 40.0 | — |

**Trade-off.** `remat_policy='full'` cuts the device peak of the 24-layer bf16 model from 5,212 to 2,072 MB for 260 instead of 212 ms per step (the forward runs twice), and it is what lets the dense-attention model train at all at 4096 tokens. `scan_layers=True` compiles 1.07 s instead of 7.32 s, but without remat it keeps every layer's activations as scan residuals and ran out of memory in the peak measurement. On CPU, `remat_policy='full'` is also faster, not only smaller, because the smaller working set stays in cache. Recommendation: for deep stacks use `scan_layers=True, remat_policy='full'`; `dots_saveable` sits between the two.

## SSMModel cells

`SSMModel(16, 128, 16, 4)` with each recurrent cell, batch 8, fp32 ([`gpu_nn_before.json`](gpu_nn_before.json) → [`gpu_nn.json`](gpu_nn.json)). The `finite` column reports whether every output of the timed call was finite on the merged branch.

| cell | tokens | op | time before → after (ms) | compile before → after (s) | finite |
| --- | ---: | --- | ---: | ---: | --- |
| lru | 1024 | forward | 1.76 → 1.61 | 1.33 → 1.48 | true |
| lru | 1024 | train step | 6.24 → 5.43 | 7.60 → 5.07 | true |
| lru | 4096 | forward | 6.38 → 6.20 | 1.46 → 1.43 | true |
| lru | 4096 | train step | 22.5 → 18.3 | 12.9 → 5.53 | true |
| mamba | 1024 | forward | 14.6 → 5.09 | 2.50 → 2.54 | true |
| mamba | 1024 | train step | 139 → 37.2 | 7.57 → 6.38 | true |
| mamba | 4096 | forward | 65.1 → 18.7 | 2.30 → 2.58 | true |
| mamba | 4096 | train step | OOM → 147 | OOM → 6.29 | true |
| ssd | 1024 | forward | unsup. → 1.90 | unsup. → 4.93 | true |
| ssd | 1024 | train step | unsup. → 7.17 | unsup. → 10.4 | true |
| ssd | 4096 | forward | unsup. → 7.18 | unsup. → 4.99 | true |
| ssd | 4096 | train step | unsup. → 25.2 | unsup. → 10.8 | true |
| mamba2 | 1024 | forward | — → 3.12 | — → 3.16 | true |
| mamba2 | 1024 | train step | — → 12.1 | — → 11.3 | true |
| mamba2 | 4096 | forward | — → 11.7 | — → 2.52 | true |
| mamba2 | 4096 | train step | — → 44.6 | — → 10.1 | true |
| gdn | 1024 | forward | — → 2.62 | — → 4.35 | true |
| gdn | 1024 | train step | — → 11.9 | — → 8.33 | true |
| gdn | 4096 | forward | — → 10.2 | — → 4.28 | true |
| gdn | 4096 | train step | — → 41.1 | — → 8.49 | true |

- **LRUCell.** `linear_recurrence_scan` got a custom JVP, which shortens gradient compilation and speeds up the train step.
- **SSDCell** raised before (the old GPU SSD kernel rejected one head, and on CPU `ssd` had no implementation). Its first fixed version ran but returned non-finite outputs: `log_alpha` came straight from a Linear layer (so the state could grow), heads were copies of one projection, and with `num_heads > 1` the output width was `model_dim // num_heads`. It is now a proper multi-head SSD layer (Mamba-2 decay `-exp(a_log) * softplus(dt)`, real heads with grouped `b`/`c`, skip term, output projection), and the `ssd` rows above are that version.
- **Mamba2Cell** (new) is the full Mamba-2 block on the same SSD kernel: `expand=2`, short causal convolution, gated RMSNorm, `head_dim=64`, `state_dim=64`. **GatedDeltaNetCell** (`gdn`, new) uses its defaults. Neither existed before, so they have no *before* numbers.
- **MambaCell** uses `state_dim = model_dim` by default (128 here). The new Triton scan was first tuned only for state 16 and spilled at this size; after the state-size-dependent tiles of commit 9386bfb ([kernel page](gpu_kernels.md#mamba-selective-scan)) the `ssm` rows of `gpu_nn.json` were re-run (again after the SSDCell rework, adding `mamba2` and `gdn`), and those are the *after* numbers above (`config.ssm_rerun` in the JSON). The 4096-token train step, which ran out of memory before, now runs.

??? note "Lane measurements of the SSM cells (kernel_nn_gpu / kernel_nn_cpu, pre-merge branch)"

    | device | cell | tokens | batch | op | time (ms) | compile (s) | XLA temp (MB) |
    | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
    | GPU | mamba | 1024 | 8 | forward | 14.5 | 2.20 | 468 |
    | GPU | mamba | 1024 | 8 | train step | 139 | 7.33 | 5,646 |
    | GPU | ssd | 1024 | 8 | forward | unsup. | — | — |
    | GPU | ssd | 1024 | 8 | train step | unsup. | — | — |
    | GPU | lru | 1024 | 8 | forward | 1.61 | 1.45 | 32.0 |
    | GPU | lru | 1024 | 8 | train step | 5.37 | 7.31 | 444 |
    | GPU | mamba | 4096 | 8 | forward | 65.1 | 2.38 | 2,640 |
    | GPU | mamba | 4096 | 8 | train step | OOM | — | — |
    | GPU | ssd | 4096 | 8 | forward | unsup. | — | — |
    | GPU | ssd | 4096 | 8 | train step | unsup. | — | — |
    | GPU | lru | 4096 | 8 | forward | 6.19 | 1.54 | 128 |
    | GPU | lru | 4096 | 8 | train step | 18.2 | 6.75 | 1,862 |
    | CPU | mamba | 256 | 4 | forward | 355 | 0.42 | 310 |
    | CPU | mamba | 256 | 4 | train step | 1,440 | 1.80 | 2,956 |
    | CPU | ssd | 256 | 4 | forward | unsup. | — | — |
    | CPU | ssd | 256 | 4 | train step | unsup. | — | — |
    | CPU | lru | 256 | 4 | forward | 12.4 | 0.33 | 4.88 |
    | CPU | lru | 256 | 4 | train step | 53.2 | 1.29 | 73.0 |
    | CPU | mamba | 1024 | 4 | forward | 1,679 | 0.37 | 1,321 |
    | CPU | mamba | 1024 | 4 | train step | 6,658 | 2.07 | 12,017 |
    | CPU | ssd | 1024 | 4 | forward | unsup. | — | — |
    | CPU | ssd | 1024 | 4 | train step | unsup. | — | — |
    | CPU | lru | 1024 | 4 | forward | 47.6 | 0.38 | 60.4 |
    | CPU | lru | 1024 | 4 | train step | 186 | 1.41 | 326 |

For `GatedDeltaNetCell` at the kernel level see the [Gated DeltaNet section](gpu_kernels.md#gated-deltanet).

## UNet

`UNet(3, (64, 128, 256))`, batch 8. In bf16 the `dtype` is now forwarded to the skip convolutions and GroupNorms, which previously computed in fp32:

| dtype | resolution | op | before → after (ms) | XLA temp after (MB) |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 32 | forward | 1.33 → 1.15 | 11.8 |
| bf16 | 32 | train step | 4.49 → 3.97 | 72.7 |
| bf16 | 64 | forward | 3.41 → 2.91 | 44.8 |
| bf16 | 64 | train step | 10.8 → 10.1 | 225 |
| fp32 | 32 | forward | 1.65 → 1.64 | 23.5 |
| fp32 | 32 | train step | 5.46 → 5.41 | 114 |
| fp32 | 64 | forward | 4.72 → 4.70 | 83.0 |
| fp32 | 64 | train step | 15.2 → 15.3 | 436 |

Device peak at resolution 64, fresh process ([`kernel_nn_gpu.json`](kernel_nn_gpu.json)):

| dtype | resolution | op | peak (MB) |
| --- | ---: | ---: | ---: |
| fp32 | 64 | forward | 261 |
| fp32 | 64 | train step | 12,662 |
| bf16 | 64 | forward | 254 |
| bf16 | 64 | train step | 398 |

The fp32 train step peaks far higher than bf16 at the same resolution. Remat of the UNet train step (`variant='remat'` rows in the JSON) costs time without reducing temporaries at this size.

## MoE layer

Dropless `MoELayer`, 8 experts, top-2, d_model 512, d_hidden 1024, one column per `dropless_impl`. Times did not change at the layer level; the gains are in memory and in the largest token count that fits ([kernel page](gpu_kernels.md#moe-grouped-matmul)).

| dtype | tokens | op | Pallas (default) | blocked | ragged_dot |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 4,096 | forward | 0.849 → 0.857 | 2.33 → 2.35 | 6.66 → 6.69 |
| bf16 | 4,096 | train step | 2.74 → 2.73 | 5.66 → 5.69 | 18.9 → 19.2 |
| bf16 | 32,768 | forward | 4.89 → 5.06 | 15.6 → 15.6 | 48.8 → 48.8 |
| bf16 | 32,768 | train step | 16.9 → 15.9 | 36.4 → 36.4 | 144 → 144 |
| fp32 | 4,096 | forward | 1.29 → 1.29 | 2.36 → 2.37 | 6.67 → 6.66 |
| fp32 | 4,096 | train step | 4.65 → 4.64 | 5.80 → 5.80 | 19.1 → 19.4 |
| fp32 | 32,768 | forward | 8.64 → 9.29 | 15.7 → 15.6 | 49.0 → 48.9 |
| fp32 | 32,768 | train step | 32.1 → 31.9 | 36.7 → 36.4 | 146 → 146 |

## Training-loop overhead

Per-step time of an MLP trained with a bare jitted step vs `fit()` on an in-memory array and on a stream. `fit()` still recompiles on every call: a second call with the same model and data costs nearly as much wall time as the first (last two columns). That is not fixed.

| device | width | bare jit (ms/step) | `fit` array (ms/step) | `fit` stream (ms/step) | `fit` first call (s) | `fit` second call (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU | 64 | 0.159 | 0.205 | 0.378 | 0.60 | 0.36 |
| GPU | 512 | 0.459 | 0.532 | 0.636 | 0.85 | 0.65 |
| CPU | 64 | 0.142 | 0.130 | 0.227 | 0.40 | 0.29 |
| CPU | 512 | 23.4 | 13.4 | 17.7 | 3.09 | 2.97 |

## Reproduce

```bash
# Before/after model grid (run once on the base commit, once on the branch).
python -m docs.scripts.benchmark_gpu_nn --output docs/benchmarks/gpu_nn.json

# scan_layers / remat_policy, SSM, UNet, training loop; --peak adds device peaks.
python -m docs.scripts.benchmark_kernel_nn --device gpu --output docs/benchmarks/kernel_nn_gpu.json
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_nn --device cpu \
  --output docs/benchmarks/kernel_nn_cpu.json
```
