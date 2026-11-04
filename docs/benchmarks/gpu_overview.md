# GPU kernel and model benchmarks

Results of a round of kernel and `probjax.nn` optimization aimed at consumer NVIDIA GPUs: fused attention as the default, faster and memory-lean sequence mixers (SSD, Mamba, residual linear attention, Gated DeltaNet), streaming kernel sums and optimal transport, grouped-matmul MoE, and scan-over-layers with rematerialization. The work also found and fixed several kernels that returned **wrong results on GPU without raising**. Details by area:

- [Attention](gpu_attention.md): probjax vs `jax.nn` (XLA, cuDNN) vs PyTorch SDPA, tile tuning, dispatch, memory, GQA and split-KV decoding, the CPU path.
- [Kernels](gpu_kernels.md): SSD, Mamba, residual linear attention, Gated DeltaNet, `rbf_kernel_mv` / KDE, `kernel_logsumexp` / Sinkhorn / OT-CFM, MoE grouped matmul.
- [Models](gpu_nn.md): `Transformer` / `DiffusionTransformer` per `attention_fn`, `scan_layers` and `remat_policy`, `SSMModel` cells, `UNet`, `MoELayer`.

## Setup

|  |  |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, 24 GB (sm_86, about 99 KB opt-in shared memory per block) |
| Driver | 615.78.08 |
| JAX | 0.11.2 (Pallas on Triton for GPU kernels) |
| PyTorch (comparisons only) | 2.14.1+cu130 (CUDA 13.0) |
| Python | 3.14.8 |
| Host | 24 CPU threads, shared with other jobs |

All JSON files record the device, the library versions and the runner configuration under `device` and `config`.

## Methodology

- **Timing.** Each case is compiled once, then timed as `inner` back-to-back calls bracketed by a device synchronization, repeated several times. Tables show the median per call; every JSON row also has the inter-quartile range (`q1_ms`, `q3_ms`). Compilation is timed separately (`compile_s`) and never included in the step time.
- **Before/after claims** come from at least three interleaved runs on each side (two for the MoE A/B) with non-overlapping IQRs, held under a lock so that no two benchmarks share the GPU. The cross-cutting files `gpu_kernels_before.json` / `gpu_kernels.json` and `gpu_nn_before.json` / `gpu_nn.json` run the same runner on the pre-optimization base and on the merged branch, on identical shapes.
- **Memory**, two kinds. `xla_temp_mb` / `xla_arg_mb` / `xla_out_mb` are XLA's compile-time `memory_analysis()` of the jitted function: temporaries, arguments, outputs. `peak_mb` is the true device peak (`peak_bytes_in_use`) read in a fresh subprocess that runs only that case. Temporaries say what a kernel allocates; the peak says whether the whole call fits. Where the backward pass matters, `residual_mb` is the size of the `jax.vjp` residuals.
- **Accuracy** is always reported as a *relative* error (`max|a − b| / max|b|` or a norm ratio; the JSON says which) against an fp32 reference at `HIGHEST` matmul precision or an fp64 host reference. The `max_abs_err` columns of `gpu_kernels*.json` are the exception and are labelled as absolute.
- **TF32.** On Ampere GPUs JAX runs fp32 matmuls in TF32 by default, which costs about three decimal digits. fp32 kernels that are not pinned to `HIGHEST` therefore show relative errors around 1e-3 against an fp32 reference, and tests with tight absolute tolerances fail identically before and after this work.
- **Shared host.** CPU timings ran on a shared 24-thread machine and carry 10–30% run-to-run noise; read CPU ratios as approximate. GPU timings were taken with the GPU otherwise idle.
- **Missing cells.** OOM: out of device memory. fail†: XLA could not profile any kernel configuration at this size during autotuning (reported as `NOT_FOUND`), so the case did not run. unsup.: the implementation rejects the shape or backend. —: not measured. Failed cases stay in the JSON with their `error` text.

## Biggest changes

Representative before → after pairs; each links to the full tables. Ratios at other shapes differ, sometimes a lot.

| Family | Workload | Before | After | Details |
| --- | --- | ---: | ---: | --- |
| Attention (default `attention_fn`) | bf16 fwd+bwd, B4 H8 d64 T4096: device peak | 5,249 MB | 161 MB | [attention](gpu_attention.md#memory) |
| Attention (default `attention_fn`) | longest trainable T, same shape | 4,096 | ≥ 65,536 | [attention](gpu_attention.md#memory) |
| `mha` tile heuristics | fp32 d64 T4096 fwd+bwd | 64.8 ms | 23.5 ms | [attention](gpu_attention.md#tile-heuristics-for-consumer-gpus) |
| `Transformer` (d256, 4 layers, B8) | bf16 train step, 1024 tokens | 17.8 ms | 7.55 ms | [nn](gpu_nn.md#transformer-and-diffusiontransformer) |
| `Transformer` (d256, 4 layers, B8) | bf16 train step, 4096 tokens | OOM | 59.8 ms | [nn](gpu_nn.md#transformer-and-diffusiontransformer) |
| GQA decoding (split-KV) | bf16 B8, 1 query, 64k keys, 32/8 heads | 24.4 ms, 10,242 MB | 2.43 ms, 2,050 MB | [attention](gpu_attention.md#grouped-query-attention-and-decoding) |
| Attention on CPU | causal fwd+bwd T4096 vs `jax.nn` XLA | 1,011 ms, 1,544 MB temp¹ | 392 ms, 40.1 MB temp¹ | [attention](gpu_attention.md#cpu-path) |
| SSD `ssd()` | GPU fwd+bwd L8192 (before: wrong results) | 29.8 ms | 16.0 ms | [kernels](gpu_kernels.md#ssd-mamba-2-state-space-duality) |
| SSD on CPU | fwd+bwd L2048 (before: only `ssd_linear_scan` ran) | 4,263 ms, 8,357 MB | 204 ms, 323 MB | [kernels](gpu_kernels.md#ssd-mamba-2-state-space-duality) |
| Mamba selective scan | GPU fwd+bwd B4 L8192 D512 | 54.3 ms, 4,360 MB | 4.75 ms, 776 MB | [kernels](gpu_kernels.md#mamba-selective-scan) |
| Residual linear attention | GPU fwd+bwd B2 H4 L8192 D64 | 145 ms | 3.79 ms | [kernels](gpu_kernels.md#residual-linear-attention-rattention) |
| Residual linear attention | GPU fwd+bwd device peak B4 H8 L2048 | 6,225 MB | 389 MB | [kernels](gpu_kernels.md#residual-linear-attention-rattention) |
| Gated DeltaNet (new) | GPU bf16 fwd L8192 d128, recurrence → Pallas | 109 ms | 5.86 ms | [kernels](gpu_kernels.md#gated-deltanet) |
| `rbf_kernel_mv` | GPU fwd+bwd N=M=65536, D8 | 122 ms | 58.9 ms | [kernels](gpu_kernels.md#rbf-kernel-matrix-vector-products-and-kde) |
| `rbf_kernel_mv` on CPU | fwd+bwd N=M=1024, D8 | 13.5 ms | 2.83 ms | [kernels](gpu_kernels.md#rbf-kernel-matrix-vector-products-and-kde) |
| `kernel_logsumexp` (new) | GPU fwd N=M=16384, D2, vs dense | 4.48 ms, 1,028 MB | 1.22 ms, 2.00 MB | [kernels](gpu_kernels.md#log-domain-kernel-reductions-sinkhorn-and-ot-cfm) |
| Sinkhorn (new) | GPU 50 iterations N=M=16384, vs dense | 250 ms, 2,048 MB | 127 ms, 5.31 MB | [kernels](gpu_kernels.md#log-domain-kernel-reductions-sinkhorn-and-ot-cfm) |
| MoE dropless (GPU) | bf16 train step, 524k tokens: device peak | 9,004 MB | 4,910 MB | [kernels](gpu_kernels.md#moe-grouped-matmul) |
| MoE dropless (CPU) | 4096 tokens fwd, `blocked` → `chunked` | 151 ms | 78.2 ms | [kernels](gpu_kernels.md#moe-grouped-matmul) |
| 24-layer `Transformer` | bf16 L4096 train step: peak, compile | 5,212 MB, 9.46 s | 2,126 MB, 1.07 s | [nn](gpu_nn.md#scan-over-layers-and-rematerialization) |
| `SSMModel` + `MambaCell` | train step, 1024 tokens (state 128) | 139 ms | 37.2 ms | [nn](gpu_nn.md#ssmmodel-cells) |
| Mamba scan, large state | GPU bwd B4 L2048 D512 state 128 | 174 ms | 26.6 ms | [kernels](gpu_kernels.md#mamba-selective-scan) |
| `SSMModel` + `LRUCell` | train step L4096: time, compile | 22.5 ms, 12.9 s | 18.3 ms, 5.53 s | [nn](gpu_nn.md#ssmmodel-cells) |

¹ The CPU attention temporaries are for the non-causal fwd+bwd at the same length.

## Correctness bugs found and fixed

Several of these produced wrong numbers on GPU without any error or warning. If you used the affected functions on GPU before this change, re-run the affected computations.

**Silent wrong results on GPU**

1. **`ssd` (Mamba-2 SSD), every `seq_len > 256`.** A race between sequence chunks in the Triton kernel corrupted the carried state, giving errors from several percent to several hundred percent and occasional NaN. `gpu_kernels_before.json` shows a max absolute error of 2.2e-01 at seq 8192 (forward) against 1.2e-05 after the fix. Fixed by carrying the state in registers; the default is now the chunked pure-JAX algorithm anyway.
2. **`rbf_kernel_mv` / `kernel_mv` for feature dimension D ∈ {1, 2, 4}.** The Pallas kernel returned wrong values. Fixed by zero-padding D to a power of two ≥ 8.
3. **Every `kde_density` / `rbf_kde_density` gradient on GPU.** The backward was wrong whenever the output tile `block_o` was 1, which these functions always used. Fixed by clamping `block_o` to ≥ 8.
4. **`dot_product_attention(..., sm_scale=...)`** ignored `sm_scale` on the einsum path (any non-default scale was silently replaced by `1/sqrt(d)`).
5. **`mha` with `seq_len` not a multiple of the block size** returned NaN. It now pads with a length mask and slices.
6. **`SameSegmentMask(key_segment_ids=None)`** used the query tile's segment ids for the keys in the kernel path.
7. **Residual linear attention, fp32 key gradient under TF32**, was off by several percent relative; it is now computed at `HIGHEST` precision.

**Crashes and silent fallbacks**

- `compute_mamba_scan` on GPU never ran its Pallas kernel for multi-tile shapes; it fell back to an associative scan that materialized `[B, L, D, S]` (correct, but slow, and `MambaCell` ran out of memory at 4096 tokens). The old benchmark numbers labelled "Pallas" measured that scan.
- `RAttention`'s default Pallas path requested 288 KiB of shared memory and failed on every GPU; the RLA kernel also crashed for non-power-of-two head dims and under-estimated shared memory at `precision='highest'`. The default is now the chunked form, with a dtype-aware shared-memory model, chunk auto-shrink and fallback for the Pallas path.
- The new Mamba GPU scan was tuned only at `state_dim = 16` and spilled registers at larger states (`MambaCell` defaults to `state_dim = model_dim`); commit 9386bfb added state-size-dependent tiles ([A/B](gpu_kernels.md#mamba-selective-scan)). The per-family Mamba JSON files use state 16 throughout.
- `ssd` raised on CPU, and on GPU for one head (`SSDCell` defaults) and for `seq_len % 256 != 0`; all of these now run.
- `SSDCell` returned non-finite outputs (unconstrained `log_alpha`), its heads were copies of one projection, and with `num_heads > 1` its output width was `model_dim // num_heads`. It is now a multi-head SSD layer with the Mamba-2 decay parameterization; the full Mamba-2 block is the new `Mamba2Cell` ([nn](gpu_nn.md#ssmmodel-cells)).

**Known open issues** (measured, not fixed)

- `KeyPaddingMask` gives wrong results in the kernel path for general use; use `KVLenMask`.
- The `RAttention` layer runs out of memory at 8192 tokens because of its dense sliding-window branch.
- `fit()` recompiles on every call.

## What runs by default

With no `implementation` argument, the public functions dispatch as follows. "Kernel" means a Pallas kernel compiled through Triton.

| Operation | CPU | NVIDIA GPU | Override |
| --- | --- | --- | --- |
| `dot_product_attention` (default `attention_fn`) | Flax einsum | `flex_attention` kernel when exact (no dense mask or bias, no dropout, default precision, ≥ 64 keys); einsum otherwise | `implementation='xla' / 'pallas'` |
| `flex_attention`, `mha`, `mla` | XLA: dense ≤ 256, single tile ≤ 512, else 256×256 tiles | kernel; split-KV forward for unmasked Lq ≤ 64, Lkv ≥ 1024 | `split_kv=False` |
| `ssd` | chunked pure JAX | chunked pure JAX (Pallas only on TPU) | `implementation='pallas' / 'xla'` |
| `compute_mamba_scan` | chunked XLA scan | Triton sequence-loop kernel (chunked XLA for unsupported shapes) | — |
| `residual_linear_attention` | chunked pure JAX | chunked pure JAX | `implementation='pallas' / 'scan'` |
| `gated_delta_rule` | chunked pure JAX | kernel for supported chunk / head sizes, else chunked | `implementation='recurrent' / 'chunked' / 'pallas'` |
| `rbf_kernel_mv`, `kde_density` | row-blocked XLA | kernel | `implementation='xla' / 'pallas'` |
| `kernel_logsumexp` | blocked XLA | kernel (3×TF32 cross term); dense for N·M ≤ 2²¹ | `impl='pallas' / 'xla' / 'dense'` |
| `MoELayer(routing='dropless')` | `chunked` | grouped-matmul kernel, `sorted` dispatch | `dropless_impl`, `dropless_dispatch` |
| `Transformer`, `DiffusionTransformer` | no remat, no scan | no remat, no scan | `remat_policy`, `scan_layers` (changes the parameter tree) |

## Reproduce

All runners live in `docs/scripts/` and write JSON to `docs/benchmarks/`. GPU runs need a CUDA GPU; torch is optional and only adds comparison rows. Run one benchmark at a time on an otherwise idle GPU.

```bash
# Cross-cutting before/after files: run once on the base commit, once on the branch.
python -m docs.scripts.benchmark_gpu_kernels --output docs/benchmarks/gpu_kernels.json
python -m docs.scripts.benchmark_gpu_nn --output docs/benchmarks/gpu_nn.json

# Attention grid against JAX and PyTorch, and the tile sweep.
python -m docs.scripts.benchmark_gpu_attention --output docs/benchmarks/gpu_attention.json
python -m docs.scripts.benchmark_gpu_attention_tiles --output docs/benchmarks/gpu_attention_tiles.json

# One family, CPU then GPU (other families: see each page).
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_ssd --device cpu \
  --output docs/benchmarks/kernel_ssd_cpu.json
python -m docs.scripts.benchmark_kernel_ssd --device gpu --output docs/benchmarks/kernel_ssd_gpu.json
```

Shared timing helpers are in `docs/scripts/_gpu_bench.py`. Each page ends with the commands for its own files.
