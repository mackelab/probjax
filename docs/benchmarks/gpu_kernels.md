# GPU kernel benchmarks: sequence mixers, kernel sums, OT and MoE

Time and memory of the non-softmax kernels before and after the optimization work, on CPU and GPU, together with the largest problem that fits and the default each public function dispatches to. Methodology, the device and the full list of correctness bugs are on the [overview](gpu_overview.md); attention is on its [own page](gpu_attention.md). Times are medians in milliseconds. OOM: out of device memory. fail†: XLA could not profile any kernel configuration at this size during autotuning (reported as `NOT_FOUND`), so the case did not run. unsup.: the implementation rejects the shape or backend. —: not measured. CPU timings come from a shared 24-thread host and are noisier than the GPU numbers; read CPU ratios as approximate.

## SSD (Mamba-2 state-space duality)

`ssd(q, k, v, log_alpha)` computes the Mamba-2 / SSD recurrence. Before this work the GPU path was a Pallas/Triton kernel that **returned wrong results for every `seq_len > 256`**: a race between sequence chunks corrupted the carried state (several-percent to several-hundred-percent error, sometimes NaN). On CPU `ssd` raised; only the per-step reference `ssd_linear_scan` ran. The kernel is fixed (state carried in registers), its `seq_len % 256` restriction relaxed to `% 64` and its `num_heads >= 2` restriction removed. A new pure-JAX chunked implementation (`ssd_xla.py`, the Mamba-2 chunked algorithm with `remat='full'`) is now the default on both CPU and GPU (`implementation=None`); `implementation='pallas'` still selects the Triton kernel.

**Cross-check on identical shapes** (batch 4, 8 heads, head_dim 128, [`gpu_kernels_before.json`](gpu_kernels_before.json) → [`gpu_kernels.json`](gpu_kernels.json)). `ssd()` is the public function: the buggy kernel before, the chunked default after. The error column is the max absolute error against `ssd_linear_scan` (empty when the reference itself ran out of memory):

| seq | mode | `ssd_linear_scan` (ms) | `ssd()` before → after (ms) | max abs err before → after |
| --- | ---: | ---: | ---: | ---: |
| 512 | fwd | 4.92 | 0.359 → 0.329 | 1.3e-01 → 6.7e-06 |
| 512 | fwd+bwd | 28.7 | 2.02 → 1.23 | 1.1e+00 → 1.6e-04 |
| 2048 | fwd | 21.8 | 1.47 → 1.16 | 1.2e-01 → 8.6e-06 |
| 2048 | fwd+bwd | fail† | 7.50 → 4.36 | — → — |
| 8192 | fwd | 86.8 | 5.85 → 4.16 | 2.2e-01 → 1.2e-05 |
| 8192 | fwd+bwd | OOM | 29.8 → 16.0 | — → — |

**GPU, all implementations** (fixed kernel vs chunked default), batch 4, 2 groups, 8 heads, d_k = d_v = 128, fp32 ([`kernel_ssd_gpu.json`](kernel_ssd_gpu.json)):

| seq | mode | scan (ms) | Pallas, fixed (ms) | chunked, default (ms) | Pallas temp (MB) | chunked temp (MB) | Pallas residuals (MB) | chunked residuals (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | fwd | 4.90 | 0.363 | 0.332 | 16.1 | 28.1 |  |  |
| 512 | fwd+bwd | 27.7 | 1.46 | 0.901 | 48.0 | 48.2 | 8.00 | 0 |
| 2048 | fwd | 21.8 | 1.46 | 1.17 | 64.5 | 113 |  |  |
| 2048 | fwd+bwd | fail† | 5.57 | 3.21 | 192 | 225 | 32.0 | 0 |
| 8192 | fwd | 90.6 | 6.00 | 4.18 | 258 | 292 |  |  |
| 8192 | fwd+bwd | OOM | 22.2 | 11.9 | 768 | 804 | 128 | 0 |

Device peak at seq 8192 (fresh process, MB), and the longest fwd+bwd that fit on the 24 GB card:

| config | impl | fwd peak | fwd+bwd peak |
| --- | --- | ---: | ---: |
| g2h8_d128 | scan | 964 | OOM |
| g2h8_d128 | pallas | 964 | 1,287 |
| g2h8_d128 | chunked | 1,128 | 1,323 |
| g1h1_d128 | scan | 144 | 6,257 |
| g1h1_d128 | pallas | 176 | 241 |
| g1h1_d128 | chunked | 320 | 369 |

| impl (g2h8_d128, fwd+bwd) | longest seq that ran | peak there (MB) |
| --- | ---: | ---: |
| chunked | 65,536 | 12,076 |
| scan | 1,024 | 6,228 |
| pallas | 65,536 | 10,260 |

**CPU**, same shape ([`kernel_ssd_cpu.json`](kernel_ssd_cpu.json)). `scan` was not run at 8192. Before this work `ssd` itself raised on CPU, so `scan` is the only baseline:

| seq | mode | scan (ms) | chunked (ms) | scan temp (MB) | chunked temp (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | fwd | 34.3 | 13.8 | 34.1 | 30.1 |
| 512 | fwd+bwd | 1,127 | 47.2 | 2,092 | 80.1 |
| 2048 | fwd | 141 | 72.8 | 130 | 114 |
| 2048 | fwd+bwd | 4,263 | 204 | 8,357 | 323 |
| 8192 | fwd | — | 296 | — | 451 |
| 8192 | fwd+bwd | — | 798 | — | 1,284 |

??? note "Other head configurations (24 heads with d_v 64, one head, d_k 48 / d_v 40)"

    **GPU** ([`kernel_ssd_gpu.json`](kernel_ssd_gpu.json))

    | config | seq | mode | scan (ms) | pallas (ms) | chunked (ms) |
    | --- | ---: | --- | ---: | ---: | ---: |
    | g1h24_dk128_dv64 | 512 | fwd | 4.71 | 0.321 | 0.258 |
    | g1h24_dk128_dv64 | 512 | fwd+bwd | 22.2 | 1.20 | 0.725 |
    | g1h24_dk128_dv64 | 2048 | fwd | 18.8 | 1.22 | 0.943 |
    | g1h24_dk128_dv64 | 2048 | fwd+bwd | OOM | 4.48 | 2.63 |
    | g1h24_dk128_dv64 | 8192 | fwd | 82.8 | 4.83 | 3.46 |
    | g1h24_dk128_dv64 | 8192 | fwd+bwd | OOM | 18.4 | 10.7 |
    | g1h1_d128 | 512 | fwd | 4.34 | 0.103 | 0.089 |
    | g1h1_d128 | 512 | fwd+bwd | 12.5 | 0.371 | 0.252 |
    | g1h1_d128 | 2048 | fwd | 17.5 | 0.389 | 0.284 |
    | g1h1_d128 | 2048 | fwd+bwd | 49.6 | 1.38 | 0.656 |
    | g1h1_d128 | 8192 | fwd | 67.9 | 1.53 | 0.909 |
    | g1h1_d128 | 8192 | fwd+bwd | 197 | 5.37 | 2.44 |
    | g2h4_dk48_dv40 | 512 | fwd | unsup. | unsup. | 0.112 |
    | g2h4_dk48_dv40 | 512 | fwd+bwd | unsup. | unsup. | 0.296 |
    | g2h4_dk48_dv40 | 2048 | fwd | unsup. | unsup. | 0.371 |
    | g2h4_dk48_dv40 | 2048 | fwd+bwd | unsup. | unsup. | 0.968 |
    | g2h4_dk48_dv40 | 8192 | fwd | unsup. | unsup. | 1.45 |
    | g2h4_dk48_dv40 | 8192 | fwd+bwd | unsup. | unsup. | 3.74 |

    **CPU** ([`kernel_ssd_cpu.json`](kernel_ssd_cpu.json))

    | config | seq | mode | scan (ms) | chunked (ms) |
    | --- | ---: | --- | ---: | ---: |
    | g1h24_dk128_dv64 | 512 | fwd | 25.0 | 9.81 |
    | g1h24_dk128_dv64 | 512 | fwd+bwd | 814 | 41.1 |
    | g1h24_dk128_dv64 | 2048 | fwd | 133 | 62.0 |
    | g1h24_dk128_dv64 | 2048 | fwd+bwd | 3,047 | 175 |
    | g1h24_dk128_dv64 | 8192 | fwd | — | 242 |
    | g1h24_dk128_dv64 | 8192 | fwd+bwd | — | 706 |
    | g1h1_d128 | 512 | fwd | 3.82 | 1.71 |
    | g1h1_d128 | 512 | fwd+bwd | 119 | 5.01 |
    | g1h1_d128 | 2048 | fwd | 15.2 | 6.00 |
    | g1h1_d128 | 2048 | fwd+bwd | 496 | 28.5 |
    | g1h1_d128 | 8192 | fwd | — | 40.9 |
    | g1h1_d128 | 8192 | fwd+bwd | — | 119 |
    | g2h4_dk48_dv40 | 512 | fwd | unsup. | 2.18 |
    | g2h4_dk48_dv40 | 512 | fwd+bwd | unsup. | 6.71 |
    | g2h4_dk48_dv40 | 2048 | fwd | unsup. | 8.47 |
    | g2h4_dk48_dv40 | 2048 | fwd+bwd | unsup. | 44.2 |
    | g2h4_dk48_dv40 | 8192 | fwd | — | 59.2 |
    | g2h4_dk48_dv40 | 8192 | fwd+bwd | — | 176 |

**Accuracy.** Errors are relative (`max|a − b| / max|b|`) against an fp32 `HIGHEST`-precision run of the chunked algorithm, so the chunked default shows 0 on outputs by construction; its gradients differ from that reference by at most 1.3e-07 at seq 8192. The fixed Pallas kernel runs its matmuls in TF32: output error 5.2e-04, gradient error up to 1.1e-02 (largest on `log_alpha`).

**When to use which.** Keep the default. The chunked form is faster than the fixed Triton kernel at every measured GPU shape, keeps no residuals (`remat='full'`), runs any shape, and is the only CPU path. `ssd_linear_scan` is a reference only: its backward stores every state and runs out of memory from seq 2048 on GPU. Known open issues: `SSDCell` with `num_heads > 1` produces outputs of width `model_dim // num_heads`, and its `log_alpha` is unconstrained (see the [nn page](gpu_nn.md#ssmmodel-cells)).

## Mamba selective scan

`compute_mamba_scan` is the selective scan used by `MambaCell`. On GPU it **never ran its Pallas kernel for any multi-tile shape** (that is, any realistic shape): it silently fell back to a whole-sequence associative scan that materializes a `[B, L, D, S]` tensor. The results were correct but slow and memory-hungry, and the "Pallas" numbers in the old benchmarks were that XLA scan. It is replaced by Triton sequence-loop kernels: the state stays in registers, the sequence is processed in segments that are combined afterwards, and the backward pass recomputes instead of storing states. `state_dim < 16` is padded. On CPU a chunked XLA scan with a custom VJP (`mamba_chunked.py`) replaces the associative scan.

**Cross-check on identical shapes**, batch 4, D 512, state 16, fp32 ([`gpu_kernels_before.json`](gpu_kernels_before.json) → [`gpu_kernels.json`](gpu_kernels.json); error is max absolute vs `lax.scan`, see the accuracy note):

| seq | mode | `lax.scan` (ms) | `compute_mamba_scan` before → after (ms) | max abs err before → after |
| --- | ---: | ---: | ---: | ---: |
| 512 | fwd | 3.40 | 0.631 → 0.199 | 9.2e-05 → 3.1e-05 |
| 512 | fwd+bwd | 14.3 | 2.88 → 0.705 | 7.0e-02 → 2.3e-02 |
| 2048 | fwd | 13.5 | 2.76 → 0.394 | 7.9e-04 → 3.1e-03 |
| 2048 | fwd+bwd | 57.4 | 13.8 → 1.41 | 3.1e+00 → 7.7e+00 |
| 8192 | fwd | 54.0 | 11.0 → 1.24 | 7.7e-03 → 1.9e-02 |
| 8192 | fwd+bwd | 230 | 54.3 → 4.75 | 1.0e+02 → 2.7e+02 |

**GPU, all implementations**, batch 4, state 16, fp32, time in ms ([`kernel_mamba_gpu.json`](kernel_mamba_gpu.json)):

| D | seq | mode | `lax.scan` | assoc (old GPU path) | chunked XLA | Triton (new GPU path) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2048 | fwd | 13.8 | 2.73 | 4.96 | 0.394 |
| 512 | 2048 | fwd+bwd | 63.4 | 11.8 | 16.0 | 1.49 |
| 512 | 8192 | fwd | 55.1 | 10.7 | 19.3 | 1.32 |
| 512 | 8192 | fwd+bwd | 255 | 75.5 | 65.6 | 5.07 |
| 1024 | 2048 | fwd | 14.0 | 5.62 | 9.75 | 0.639 |
| 1024 | 2048 | fwd+bwd | 68.0 | 23.7 | 31.0 | 2.53 |
| 1024 | 8192 | fwd | 55.6 | 23.0 | 41.2 | 2.21 |
| 1024 | 8192 | fwd+bwd | OOM | 148 | 125 | 9.16 |

XLA temporaries of fwd+bwd (MB):

| D | seq | `lax.scan` | assoc (old GPU path) | chunked XLA | Triton (new GPU path) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 2048 | 1,345 | 1,024 | 78.8 | 98.0 |
| 512 | 8192 | 5,378 | 4,032 | 238 | 392 |
| 1024 | 2048 | 2,689 | 2,047 | 155 | 196 |
| 1024 | 8192 | OOM | 8,064 | 470 | 784 |

Device peak (fresh process), D 512 (MB):

| seq | mode | `lax.scan` | assoc (old GPU path) | chunked XLA | Triton (new GPU path) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | fwd | 113 | 321 | 113 | 97.1 |
| 2048 | fwd+bwd | 1,411 | 1,091 | 195 | 195 |
| 8192 | fwd | 452 | 1,284 | 452 | 229 |
| 8192 | fwd+bwd | 5,642 | 4,360 | 502 | 776 |

Longest power-of-two sequence whose fwd+bwd fit (batch 4, D 1024; the search limit was 262144):

| impl | longest seq | peak there (MB) |
| --- | ---: | ---: |
| `lax.scan` | 4,096 | 5,638 |
| assoc (old GPU path) | 8,192 | 8,584 |
| chunked XLA | 65,536 | 7,570 |
| Triton (new GPU path) | 65,536 | 10,432 |

**CPU**, batch 4, state 16, fp32 ([`kernel_mamba_cpu.json`](kernel_mamba_cpu.json); assoc was skipped at 8192, where it takes over a second per call and gigabytes of temporaries):

| D | seq | mode | `lax.scan` (ms) | assoc (old CPU path) (ms) | chunked (new CPU path) (ms) | `lax.scan` temp (MB) | assoc (old CPU path) temp (MB) | chunked (new CPU path) temp (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2048 | fwd | 56.9 | 661 | 92.2 | 49.6 | 1,257 | 33.1 |
| 512 | 2048 | fwd+bwd | 1,934 | 1,923 | 383 | 1,122 | 3,320 | 73.1 |
| 512 | 8192 | fwd | 232 | — | 205 | 198 | — | 132 |
| 512 | 8192 | fwd+bwd | 5,166 | — | 1,037 | 4,486 | — | 292 |
| 1024 | 2048 | fwd | 176 | 1,275 | 95.4 | 97.8 | 2,514 | 65.3 |
| 1024 | 2048 | fwd+bwd | 2,677 | 3,648 | 568 | 2,242 | 6,641 | 145 |
| 1024 | 8192 | fwd | 319 | — | 299 | 390 | — | 260 |
| 1024 | 8192 | fwd+bwd | 7,782 | — | 1,868 | 8,967 | — | 580 |

**Accuracy.** All fp32 implementations agree with `lax.scan` to 7.8e-05 relative error at seq 8192 (fwd+bwd, D 512). The large *absolute* gradient errors in the cross-check table are not a bug: the gradients themselves become very large at these lengths, so a small relative error is a large absolute one, and `lax.scan` in fp32 is no closer to an fp64 reference than the kernels are.

**State size.** All tables above use `state_dim = 16`. The GPU tile configuration was originally tuned only there and spilled registers at larger states, which matters because `MambaCell` defaults to `state_dim = model_dim`. Commit 9386bfb made the tiles depend on the state size. Locked A/B, three interleaved rounds per side, medians ([`kernel_mamba_gpu_state.json`](kernel_mamba_gpu_state.json)); the last column says whether the before/after IQRs overlap:

| B | L | D | state | pass | before → after (ms) | IQRs overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1024 | 128 | 16 | fwd | 0.230 → 0.210 | yes |
| 8 | 1024 | 128 | 16 | bwd | 0.570 → 0.570 | yes |
| 8 | 1024 | 128 | 64 | fwd | 6.90 → 0.440 | no |
| 8 | 1024 | 128 | 64 | bwd | 8.68 → 1.27 | no |
| 8 | 1024 | 128 | 128 | fwd | 8.21 → 0.920 | no |
| 8 | 1024 | 128 | 128 | bwd | 41.6 → 6.52 | no |
| 4 | 2048 | 512 | 64 | fwd | 26.7 → 0.980 | no |
| 4 | 2048 | 512 | 64 | bwd | 33.6 → 3.96 | no |
| 4 | 2048 | 512 | 128 | fwd | 28.3 → 3.22 | no |
| 4 | 2048 | 512 | 128 | bwd | 174 → 26.6 | no |
| 2 | 4096 | 256 | 32 | fwd | 0.480 → 0.460 | yes |
| 2 | 4096 | 256 | 32 | bwd | 1.60 → 1.54 | no |

**When to use which.** The default dispatch is right on both backends (Triton on GPU, chunked on CPU). The chunked XLA scan is also the fallback on GPU for shapes the kernel does not support.

## Residual linear attention (RAttention)

`residual_linear_attention` (the linear-attention branch of `RAttention`) now defaults to a chunk-parallel pure-JAX form (`residual_linear_attention_chunked`, `implementation='auto'`) on every backend. Before, the GPU default was a Pallas forward with a `lax.scan` backward (`pallas_scanbwd` below), which crashed for non-power-of-two head dims, under-predicted its shared memory at `precision='highest'`, and, as the `RAttention` default, failed on every GPU because it requested 288 KiB of shared memory. Under TF32 the fp32 key gradient was also inaccurate; the chunked form computes it at `HIGHEST` precision.

**Cross-check on identical shapes**, batch 4, 8 heads, D 64, fp32 ([`gpu_kernels_before.json`](gpu_kernels_before.json) → [`gpu_kernels.json`](gpu_kernels.json); `pallas` there is `residual_linear_attention_pallas`, which errored on this GPU before):

| seq | mode | `lax.scan` (ms) | Pallas before → after (ms) | max abs err before → after |
| --- | ---: | ---: | ---: | ---: |
| 512 | fwd | 4.28 | unsup. → 0.223 | unsup. → 8.3e-05 |
| 512 | fwd+bwd | 14.5 | unsup. → 0.999 | unsup. → 5.0e-04 |
| 2048 | fwd | 17.1 | unsup. → 0.870 | unsup. → 1.5e-04 |
| 2048 | fwd+bwd | 58.0 | unsup. → 3.93 | unsup. → 4.6e-03 |
| 8192 | fwd | 68.1 | unsup. → 3.44 | unsup. → 2.7e-04 |
| 8192 | fwd+bwd | OOM | unsup. → 14.9 | unsup. → — |

**GPU kernel**, batch 2, 4 heads, chunk 128, fp32 ([`kernel_rla_gpu.json`](kernel_rla_gpu.json)):

| D | seq | mode | `lax.scan` (ms) | previous default (Pallas fwd, scan bwd) (ms) | Pallas fwd, chunked bwd (ms) | chunked (default) (ms) | chunked temp (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2048 | fwd | 30.0 | 0.786 | 0.699 | 0.314 | 40.0 |
| 64 | 2048 | fwd+bwd | 34.4 | 36.4 | 1.62 | 0.970 | 68.5 |
| 64 | 8192 | fwd | 64.5 | 2.79 | 2.79 | 1.23 | 160 |
| 64 | 8192 | fwd+bwd | 136 | 145 | 6.36 | 3.79 | 274 |
| 128 | 2048 | fwd | 21.5 | — | — | 0.667 | 68.0 |
| 128 | 2048 | fwd+bwd | 62.1 | — | — | 2.12 | 129 |
| 128 | 8192 | fwd | 83.5 | — | — | 2.69 | 260 |
| 128 | 8192 | fwd+bwd | OOM | — | — | 8.39 | 483 |

??? note "bf16 kernel timings"

    | D | seq | mode | `lax.scan` (ms) | previous default (Pallas fwd, scan bwd) (ms) | Pallas fwd, chunked bwd (ms) | chunked (default) (ms) |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | 32 | 512 | fwd | 4.20 | — | 0.063 | 0.049 |
    | 32 | 512 | fwd+bwd | 7.97 | — | 0.132 | 0.111 |
    | 32 | 2048 | fwd | 16.3 | — | 0.215 | 0.088 |
    | 32 | 2048 | fwd+bwd | 28.2 | — | 0.416 | 0.288 |
    | 32 | 8192 | fwd | 64.9 | — | 0.763 | 0.386 |
    | 32 | 8192 | fwd+bwd | 113 | — | 1.68 | 1.14 |
    | 64 | 512 | fwd | 4.12 | — | 0.174 | 0.053 |
    | 64 | 512 | fwd+bwd | 8.57 | — | 0.308 | 0.193 |
    | 64 | 2048 | fwd | 16.3 | — | 0.594 | 0.190 |
    | 64 | 2048 | fwd+bwd | 34.2 | — | 1.15 | 0.689 |
    | 64 | 8192 | fwd | 66.5 | — | 2.37 | 0.765 |
    | 64 | 8192 | fwd+bwd | 136 | — | 4.74 | 2.76 |
    | 128 | 512 | fwd | 4.38 | — | — | 0.108 |
    | 128 | 512 | fwd+bwd | 13.8 | — | — | 0.397 |
    | 128 | 2048 | fwd | 17.8 | — | — | 0.421 |
    | 128 | 2048 | fwd+bwd | 55.4 | — | — | 1.48 |
    | 128 | 8192 | fwd | 71.0 | — | — | 1.73 |
    | 128 | 8192 | fwd+bwd | OOM | — | — | 5.90 |

Device peak of fwd+bwd, batch 4, 8 heads, D 64, fp32 (fresh process, MB):

| impl | seq=2048 | seq=8192 | seq=32768 | seq=65536 | seq=131072 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lax.scan` | 6,225 | fail† | OOM | OOM | OOM |
| previous default | 6,225 | fail† | OOM | OOM | OOM |
| Pallas fwd, chunked bwd | 465 | 1,164 | 4,644 | 11,843 | OOM |
| chunked (default) | 389 | 1,411 | 5,155 | 12,867 | OOM |

**`RAttention` layer** end to end (forward, and gradient of a scalar loss), batch 2, 8 heads, head_dim 64, window 128. The layer still runs out of memory at seq 8192 on GPU because its sliding-window attention branch is dense; that is not fixed:

| seq | mode | GPU scan | GPU Pallas | GPU chunked | CPU scan | CPU chunked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | fwd | 4.16 | 0.437 | 0.372 | 40.0 | 20.3 |
| 512 | fwd+bwd | 12.2 | 1.33 | 1.21 | 195 | 81.1 |
| 2048 | fwd | 17.9 | 2.57 | 2.45 | 333 | 295 |
| 2048 | fwd+bwd | 46.4 | 8.02 | 7.60 | 1,286 | 865 |
| 8192 | fwd | OOM | OOM | OOM | — | — |
| 8192 | fwd+bwd | fail† | OOM | OOM | — | — |

**CPU kernel**, batch 2, 4 heads, fp32 ([`kernel_rla_cpu.json`](kernel_rla_cpu.json)):

| D | seq | mode | scan (ms) | chunked (ms) | scan temp (MB) | chunked temp (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | fwd | 6.14 | 2.69 | 9.00 | 8.00 |
| 64 | 512 | fwd+bwd | 66.3 | 10.4 | 154 | 28.3 |
| 64 | 2048 | fwd | 19.9 | 13.0 | 36.0 | 32.0 |
| 64 | 2048 | fwd+bwd | 266 | 66.9 | 616 | 113 |
| 64 | 8192 | fwd | 105 | 81.5 | 144 | 144 |
| 64 | 8192 | fwd+bwd | 1,094 | 296 | 2,465 | 450 |
| 128 | 512 | fwd | 37.0 | 5.66 | 18.0 | 20.0 |
| 128 | 512 | fwd+bwd | 271 | 28.8 | 564 | 61.1 |
| 128 | 2048 | fwd | 113 | 40.0 | 72.0 | 80.0 |
| 128 | 2048 | fwd+bwd | 947 | 145 | 2,256 | 241 |
| 128 | 8192 | fwd | 397 | 203 | 288 | 352 |
| 128 | 8192 | fwd+bwd | 3,685 | 593 | 9,025 | 930 |

**Accuracy**, relative to the fp32 `lax.scan`: chunked 1.6e-06 (forward) and 4.9e-04 (fwd+bwd, max over outputs and gradients) at seq 8192, D 64; the Pallas forward 3.1e-04 (TF32).

**When to use which.** Keep `implementation='auto'`. It is the fastest form measured on both backends, and its backward (which `implementation='pallas'` shares) does not store per-token state the way the scan backward does. The Pallas forward is slower than the chunked one on this GPU.

## Gated DeltaNet

New in this work: `GatedDeltaNetCell` (`probjax.nn`) and the kernels `gated_delta_rule_recurrent`, `gated_delta_rule_chunked` and `gated_delta_rule_pallas`, dispatched by `gated_delta_rule(implementation='auto')`: Pallas on GPU for supported chunk sizes and head dims, the chunked pure-JAX form otherwise and on CPU. The chunked forms use the WY/UT representation with a hand-written VJP that keeps only chunk-boundary states as residuals. There is no "before" implementation; the baseline is the token-by-token recurrence, and `ssd` on the same shape is listed as a cost reference (SSD has no delta-rule correction term).

**GPU**, bf16, batch 4, 8 heads, chunk 64, d_k = d_v, time in ms ([`kernel_delta_gpu.json`](kernel_delta_gpu.json)):

| d | seq | mode | recurrent | chunked | chunked, more residuals | Pallas (GPU default) | `ssd` (reference cost) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2048 | fwd | 17.3 | 1.72 | 1.67 | 0.536 | 0.520 |
| 64 | 2048 | fwd+bwd | 89.1 | 5.54 | 4.67 | 3.32 | 1.84 |
| 64 | 8192 | fwd | 72.1 | 7.23 | 7.21 | 2.10 | 2.03 |
| 64 | 8192 | fwd+bwd | fail† | 19.3 | 17.3 | 10.8 | 7.21 |
| 128 | 2048 | fwd | 27.6 | 2.65 | 2.66 | 1.51 | 1.86 |
| 128 | 2048 | fwd+bwd | fail† | 8.68 | 7.94 | 5.40 | 6.17 |
| 128 | 8192 | fwd | 109 | 10.7 | 10.7 | 5.86 | 7.09 |
| 128 | 8192 | fwd+bwd | OOM | 33.1 | 30.4 | 20.5 | 24.2 |

Backward residuals (MB) and the Pallas device peak, bf16 fwd+bwd:

| d | seq | recurrent | chunked | chunked, more residuals | Pallas (GPU default) | `ssd` (reference cost) | Pallas peak (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | 784 | 4.00 | 28.0 | 4.00 | 17.0 | — |
| 64 | 2048 | 3,137 | 16.0 | 112 | 16.0 | 65.0 | — |
| 64 | 8192 | fail† | 64.0 | 448 | 64.0 | 257 | 1,028 |
| 128 | 512 | 3,104 | 16.0 | 56.0 | 16.0 | 36.0 | — |
| 128 | 2048 | fail† | 64.0 | 224 | 64.0 | 132 | — |
| 128 | 8192 | OOM | 256 | 896 | 256 | 516 | 2,319 |

??? note "fp32 GPU timings"

    | d | seq | mode | recurrent | chunked | chunked_fat | pallas | pallas_unfused | ssd |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | 64 | 512 | fwd | — | — | — | — | — | — |
    | 64 | 512 | fwd+bwd | — | — | — | — | — | — |
    | 64 | 2048 | fwd | 17.0 | 1.73 | 1.73 | 0.771 | 0.788 | 0.386 |
    | 64 | 2048 | fwd+bwd | 94.2 | 5.58 | 4.69 | 3.33 | 3.43 | 1.63 |
    | 64 | 8192 | fwd | 73.8 | 7.00 | 7.22 | 2.77 | 2.76 | 1.55 |
    | 64 | 8192 | fwd+bwd | fail† | 25.9 | 18.0 | 11.6 | 11.7 | 5.85 |
    | 128 | 512 | fwd | — | — | — | — | — | — |
    | 128 | 512 | fwd+bwd | — | — | — | — | — | — |
    | 128 | 2048 | fwd | 31.0 | 2.79 | 2.79 | 1.61 | 1.61 | 1.58 |
    | 128 | 2048 | fwd+bwd | fail† | 8.87 | 7.99 | 5.80 | 5.81 | 6.11 |
    | 128 | 8192 | fwd | 109 | 11.3 | 11.3 | 6.10 | 6.11 | 6.30 |
    | 128 | 8192 | fwd+bwd | OOM | 34.3 | 31.1 | 22.1 | 22.1 | 24.1 |

**CPU**, fp32, batch 2, 8 heads ([`kernel_delta_cpu.json`](kernel_delta_cpu.json); the recurrence was not run at 8192, and `ssd` is unsupported on CPU in this run):

| d | seq | mode | recurrent (ms) | chunked (ms) | chunked_fat (ms) | recurrent residuals (MB) | chunked residuals (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | fwd | 8.85 | 7.65 | 7.55 |  |  |
| 64 | 512 | fwd+bwd | 175 | 22.9 | 21.3 | 392 | 2.00 |
| 64 | 2048 | fwd | 39.8 | 40.7 | 42.3 |  |  |
| 64 | 2048 | fwd+bwd | 756 | 125 | 112 | 1,568 | 8.00 |
| 64 | 8192 | fwd | — | 179 | 179 |  |  |
| 64 | 8192 | fwd+bwd | — | 535 | 483 | — | 32.0 |
| 128 | 512 | fwd | 48.2 | 15.8 | 17.4 |  |  |
| 128 | 512 | fwd+bwd | 801 | 59.3 | 56.4 | 1,552 | 8.00 |
| 128 | 2048 | fwd | 163 | 86.2 | 84.4 |  |  |
| 128 | 2048 | fwd+bwd | 3,274 | 259 | 248 | 6,208 | 32.0 |

**Accuracy.** Relative errors against an fp32 `HIGHEST`-precision chunked reference: bf16 Pallas 4.5e-03 (fwd+bwd, d 128, seq 8192), fp32 Pallas 5.4e-04; on CPU the recurrence is within 7.7e-07 of it.

**When to use which.** Use the default. The recurrence is only a reference: its backward stores every state and fails at long sequences. On GPU, Gated DeltaNet at d 128 is faster than `ssd` on the same shape, at d 64 slower (compare the last two columns); pick the model for its inductive bias, not its speed.

## RBF kernel-matrix-vector products and KDE

`rbf_kernel_mv` computes `K(q, k) @ v` without materializing K, and `kde_density` / `rbf_kde_density` build on it. Two silent GPU bugs were fixed: the Pallas kernel **returned wrong values for feature dimension D in {1, 2, 4}**, and its backward **was wrong whenever the output tile `block_o` was 1, which made every `kde_density` / `rbf_kde_density` gradient on GPU wrong**. D is now zero-padded to a power of two ≥ 8 and `block_o` is clamped to ≥ 8. The clamp also made the kernel faster. On CPU the default changed from Pallas interpret mode to a row-blocked XLA implementation (`implementation='xla'`).

**Cross-check on identical shapes**, D 8, 16 outputs, fp32 ([`gpu_kernels_before.json`](gpu_kernels_before.json) → [`gpu_kernels.json`](gpu_kernels.json); error is max absolute vs the dense product):

| N = M | mode | dense (ms) | `rbf_kernel_mv` before → after (ms) | max abs err before → after |
| --- | ---: | ---: | ---: | ---: |
| 1024 | fwd | 0.040 | 0.052 → 0.037 | 2.1e-04 → 2.1e-04 |
| 1024 | fwd+bwd | 0.065 | 0.176 → 0.118 | 9.2e-03 → 9.2e-03 |
| 4096 | fwd | 0.331 | 0.162 → 0.075 | 1.2e-03 → 1.2e-03 |
| 4096 | fwd+bwd | 0.977 | 0.666 → 0.390 | 1.0e-01 → 1.0e-01 |
| 16384 | fwd | 5.00 | 1.97 → 0.806 | 3.3e-02 → 3.3e-02 |
| 16384 | fwd+bwd | 15.1 | 9.02 → 4.47 | 8.8e+00 → 8.8e+00 |
| 65536 | fwd | OOM | 25.6 → 10.8 | — → — |
| 65536 | fwd+bwd | OOM | 122 → 58.9 | — → — |

**GPU** `rbf_kernel_mv`, fp32, time in ms, median over repeats ([`kernel_kmv_gpu.json`](kernel_kmv_gpu.json)):

| D | outputs | N = M | mode | before (default) | after: Pallas (default) | row-blocked XLA | dense | torch dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 16 | 1024 | fwd | 0.050 | 0.035 | 0.045 | 0.045 | 0.171 |
| 8 | 16 | 1024 | fwd+bwd | 0.182 | 0.115 | 0.082 | 0.079 | 0.678 |
| 8 | 16 | 16384 | fwd | 2.11 | 0.805 | 6.14 | 5.02 | 18.1 |
| 8 | 16 | 16384 | fwd+bwd | 9.17 | 4.40 | 23.6 | 16.1 | 51.9 |
| 8 | 16 | 65536 | fwd | 26.6 | 10.6 | 105 | OOM | OOM |
| 8 | 16 | 65536 | fwd+bwd | 164 | 58.7 | 399 | OOM | OOM |
| 32 | 1 | 1024 | fwd | 0.085 | 0.054 | 0.052 | 0.049 | 0.169 |
| 32 | 1 | 1024 | fwd+bwd | 0.332 | 0.201 | 0.066 | 0.068 | 0.754 |
| 32 | 1 | 16384 | fwd | 3.94 | 1.86 | 4.13 | 5.15 | 18.5 |
| 32 | 1 | 16384 | fwd+bwd | 15.8 | 7.71 | 18.6 | 14.0 | 52.3 |
| 32 | 1 | 65536 | fwd | 75.4 | 23.7 | 72.6 | OOM | OOM |
| 32 | 1 | 65536 | fwd+bwd | 215 | 106 | 354 | OOM | OOM |

**GPU** `kde_density`. The *before* timings at D 2 measure the kernel that returned wrong values:

| D | N = M | mode | before (default) | after: Pallas | XLA | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4096 | fwd | 0.365 | 0.080 | 0.212 | 0.327 |
| 2 | 4096 | fwd+bwd | 0.337 | 0.172 | 0.686 | 0.574 |
| 2 | 65536 | fwd | 8.41 | 8.55 | 52.7 | fail† |
| 2 | 65536 | fwd+bwd | 19.8 | 20.1 | 177 | fail† |
| 8 | 4096 | fwd | 0.141 | 0.079 | 0.210 | 0.326 |
| 8 | 4096 | fwd+bwd | 0.164 | 0.168 | 0.681 | 0.576 |
| 8 | 65536 | fwd | 16.9 | 8.71 | 55.2 | fail† |
| 8 | 65536 | fwd+bwd | 20.3 | 20.5 | 184 | OOM |

Device peak (MB, fresh process), D 8, 16 outputs:

| impl | N=4096 fwd | N=4096 fwd+bwd | N=16384 fwd | N=16384 fwd+bwd | N=65536 fwd | N=65536 fwd+bwd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 322 | 353 | 2,150 | 4,101 | OOM | OOM |
| xla | 149 | 149 | 150 | 150 | 156 | 156 |
| pallas | 0.751 | 64.5 | 3.00 | 66.0 | 12.0 | 72.0 |

Largest problems: wall time of one fwd+bwd call in seconds (GPU), D 8, 16 outputs:

| impl | N=16k | N=32k | N=64k | N=128k | N=256k | N=512k | N=1024k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 1.85 | OOM | — | — | — | — | — |
| xla | 1.32 | 1.29 | 1.39 | 2.59 | 9.26 | 48.1 | timeout |
| pallas | 0.38 | 0.39 | 0.44 | 0.61 | 1.28 | 4.01 | 15.1 |

**CPU** `rbf_kernel_mv`, fp32 ([`kernel_kmv_cpu.json`](kernel_kmv_cpu.json)). *before* is Pallas interpret mode:

| D | outputs | N = M | mode | before (ms) | XLA, default (ms) | dense (ms) | torch dense (ms) | XLA temp (MB) | dense temp (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 16 | 1024 | fwd | 2.83 | 0.369 | 0.395 | 1.00 | 0.031 | 0.031 |
| 8 | 16 | 1024 | fwd+bwd | 13.5 | 2.83 | 3.27 | 5.72 | 12.1 | 16.1 |
| 8 | 16 | 4096 | fwd | — | 4.89 | 19.7 | 83.5 | 0.360 | 0.125 |
| 8 | 16 | 4096 | fwd+bwd | — | 89.5 | 101 | 289 | 80.5 | 256 |
| 8 | 16 | 16384 | fwd | — | 156 | 155 | 1,464 | 32.6 | 0.500 |
| 8 | 16 | 16384 | fwd+bwd | — | 996 | 1,570 | 4,630 | 81.6 | 4,098 |
| 32 | 1 | 1024 | fwd | 3.61 | 0.468 | 0.481 | 1.02 | 4.13 | 4.13 |
| 32 | 1 | 1024 | fwd+bwd | 20.4 | 2.06 | 3.13 | 4.14 | 12.0 | 16.0 |
| 32 | 1 | 4096 | fwd | — | 6.08 | 20.3 | 79.7 | 17.2 | 64.5 |
| 32 | 1 | 4096 | fwd+bwd | — | 82.8 | 99.0 | 256 | 64.8 | 256 |

**Accuracy.** Each JSON row carries `rel_err`, the norm-relative error against the row-blocked XLA implementation run at `HIGHEST` matmul precision (for fwd+bwd one value per gradient). After the fix the Pallas kernel is within 3.6e-04 (fwd+bwd, D 8, N = 16384, largest gradient error) of it.

**When to use which.** Keep the defaults (Pallas on GPU, XLA on CPU). The dense product is competitive for N ≤ 4096 and runs out of memory around N = 32k on the 24 GB GPU; the Pallas kernel ran fwd+bwd at N = 1M with a 320 MB device peak.

## Log-domain kernel reductions, Sinkhorn and OT-CFM

New in this work: `kernel_logsumexp`, a streaming `logsumexp_j(-c(x_i, y_j) / ε + g_j)` that never materializes the N × M cost (Pallas on GPU with a 3×TF32 cross term, blocked XLA on CPU, dense for small problems), and `probjax.nn.ot` on top of it: Sinkhorn with implicit or unrolled gradients, Sinkhorn divergence, entropic map, plan sampler, MMD, KDE log-density and `ot_pairing`. `FlowMatcher` / `LinearFlow` accept `ot_coupling=None | 'exact' | 'sinkhorn'` for OT-CFM.

**GPU `kernel_logsumexp`** vs dense `jax.nn.logsumexp` on the materialized cost, N = M ([`kernel_ot_gpu.json`](kernel_ot_gpu.json)):

| D | N | fwd dense (ms) | fwd kernel (ms) | fwd+bwd dense (ms) | fwd+bwd kernel (ms) | fwd+bwd temp dense (MB) | fwd+bwd temp kernel (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4096 | 0.243 | 0.152 | 0.647 | 0.570 | 128 | 0.782 |
| 2 | 16384 | 4.48 | 1.22 | 10.7 | 6.51 | 2,064 | 3.13 |
| 2 | 65536 | OOM | 15.9 | OOM | 87.2 | OOM | 12.5 |
| 8 | 4096 | 0.244 | 0.152 | 0.650 | 0.607 | 128 | 0.782 |
| 8 | 16384 | 4.38 | 1.24 | 10.6 | 6.92 | 2,064 | 3.13 |
| 8 | 65536 | OOM | 18.3 | fail† | 90.1 | fail† | 12.5 |
| 32 | 4096 | 0.276 | 0.273 | 0.698 | 1.12 | 128 | 0.047 |
| 32 | 16384 | 4.89 | 2.53 | 12.1 | 15.0 | 2,064 | 0.188 |
| 32 | 65536 | OOM | 35.9 | OOM | 239 | OOM | 0.750 |

Largest problems, D 2:

| N = M | dense fwd (ms) | dense fwd+bwd (ms) | kernel fwd (ms) | kernel fwd+bwd (ms) |
| --- | ---: | ---: | ---: | ---: |
| 32,768 | 15.0 | OOM | — | — |
| 49,152 | OOM | OOM | — | — |
| 65,536 | OOM | OOM | — | — |
| 131,072 | — | — | 61.1 | 337 |
| 262,144 | — | — | 246 | 1,341 |
| 1,048,576 | — | — | 3,977 | 21,638 |

Device peak (fresh process):

| impl | N = M | D | mode | peak (MB) |
| --- | ---: | ---: | --- | ---: |
| kernel | 65,536 | 8 | fwd | 12.5 |
| kernel | 65,536 | 8 | fwd+bwd | 24.5 |
| dense | 32,768 | 8 | fwd | 4,259 |
| dense | 32,768 | 8 | fwd+bwd | OOM |
| kernel | 1,048,576 | 2 | fwd+bwd | 296 |

**CPU `kernel_logsumexp`** (blocked XLA) vs dense ([`kernel_ot_cpu.json`](kernel_ot_cpu.json)):

| D | N | fwd dense (ms) | fwd kernel (ms) | fwd+bwd dense (ms) | fwd+bwd kernel (ms) | fwd+bwd temp dense (MB) | fwd+bwd temp kernel (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4096 | 23.7 | 6.89 | 94.8 | 29.9 | 192 | 16.1 |
| 2 | 16384 | 328 | 101 | 1,387 | 485 | 3,072 | 16.2 |
| 8 | 4096 | 17.6 | 6.85 | 84.8 | 32.9 | 192 | 16.4 |
| 8 | 16384 | 302 | 93.0 | 1,246 | 465 | 3,073 | 16.5 |
| 32 | 4096 | 19.2 | 7.16 | 78.3 | 31.9 | 193 | 17.6 |
| 32 | 16384 | 303 | 105 | 1,200 | 483 | 3,076 | 17.6 |

**Sinkhorn**, 50 log-domain iterations, D 8, against a dense log-domain solver on the materialized cost (ott-jax is not installed). The last column is the absolute difference of the two transport costs:

| device | N = M | dense (ms) | kernel (ms) | kernel + implicit grad (ms) | dense temp (MB) | kernel temp (MB) | abs. cost difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU | 4,096 | 16.3 | 13.8 | 13.5 | 128 | 1.33 | 0 |
| GPU | 16,384 | 250 | 127 | 130 | 2,048 | 5.31 | 9.5e-07 |
| GPU | 65,536 | OOM | 1,656 | 1,688 | OOM | 21.3 | — |
| CPU | 4,096 | 1,110 | 536 | 538 | 129 | 17.0 | 0 |
| CPU | 16,384 | 19,861 | 8,475 | 8,719 | 2,051 | 19.5 | 9.5e-07 |

**Accuracy** against an fp64 reference, N = M = 2048. `dense_default_precision` is the dense fp32 computation at JAX's default (TF32) matmul precision, which is why the kernel computes the cross term in 3×TF32:

| device | impl | D | ε | max abs err (logsumexp) | rel err (gradient) |
| --- | --- | ---: | ---: | ---: | ---: |
| GPU | dense | 8 | 0.5 | 8.9e-06 | 1.3e-06 |
| GPU | pallas | 8 | 0.5 | 1.3e-05 | 1.7e-06 |
| GPU | dense | 8 | 0.01 | 4.4e-04 | 3.9e-05 |
| GPU | pallas | 8 | 0.01 | 6.7e-04 | 5.1e-05 |
| GPU | dense | 32 | 0.05 | 3.3e-04 | 3.9e-05 |
| GPU | pallas | 32 | 0.05 | 4.4e-04 | 3.6e-05 |
| GPU | dense_default_precision | 8 | 0.01 | 1.2e+00 | — |
| CPU | dense | 8 | 0.5 | 7.0e-06 | 1.9e-06 |
| CPU | xla | 8 | 0.5 | 7.0e-06 | 1.8e-06 |
| CPU | dense | 8 | 0.01 | 4.8e-04 | 4.9e-05 |
| CPU | xla | 8 | 0.01 | 4.8e-04 | 6.0e-05 |
| CPU | dense | 32 | 0.05 | 3.4e-04 | 3.9e-05 |
| CPU | xla | 32 | 0.05 | 3.4e-04 | 3.8e-05 |

**OT-CFM demo** (`suite='cfm'`): `LinearFlow` from N(0, I) to 8 Gaussians, 3000 steps, batch 256, GPU. Straightness is lower for straighter paths; the Sinkhorn divergence to held-out target samples after 1, 2, 4 and 32 Euler steps measures sample quality (the floor is the divergence between two target samples):

| coupling | seed | ms / step | final loss | straightness | S-div 1 step | 2 steps | 4 steps | 32 steps | floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent | 0 | 3.69 | 6.68 | 3.11 | 13.6 | 11.2 | 0.417 | 0.0532 | 0.0115 |
| exact | 0 | 12.1 | 0.203 | 0.108 | 0.322 | 2.45 | 0.278 | 0.0514 | 0.0115 |
| sinkhorn | 0 | 3.96 | 0.9 | 0.174 | 0.543 | 2.86 | 0.218 | 0.05 | 0.0115 |
| independent | 1 | 3.72 | 6.98 | 3.79 | 14.7 | 11.9 | 0.481 | 0.0592 | 0.0115 |
| exact | 1 | 12.2 | 0.144 | 0.1 | 0.302 | 2.26 | 0.341 | 0.0718 | 0.0115 |
| sinkhorn | 1 | 3.99 | 0.571 | 0.178 | 0.58 | 2.79 | 0.246 | 0.0439 | 0.0115 |

**When to use which.** On GPU the kernel is at least as fast as dense in the forward pass from N = 4096 on, but its backward costs five to six times its forward, so for fwd+bwd at D 32 dense is faster while it fits (N=16384: dense 12.1 ms vs kernel 15.0 ms, at 2,064 vs 0.188 MB of temporaries). The dense cost matrix stops fitting around N = 32k–65k. For OT-CFM, Sinkhorn pairing costs little more per step than independent pairing and gives nearly the straightness of exact OT; exact pairing is about three times as expensive per step at batch 256.

## MoE grouped matmul

`MoELayer(routing='dropless')` sorts token-expert assignments by expert and runs one grouped matmul. On GPU the default is the Pallas grouped-GEMM (`dropless_impl='auto'` → Pallas, `dropless_dispatch='sorted'`). The change here processes the grouped SwiGLU in row slabs of 2**26 elements, gathered per slab, so the train step no longer holds full-size intermediates. On CPU there was no good dropless path (the gather/fused dispatches fell back to Pallas interpret mode); the new `dropless_impl='chunked'` is a plain-XLA loop over expert-sorted row chunks and is now the CPU default.

**GPU A/B**, sorted Pallas path, 8 experts, top-2, d_model 512, d_hidden 1024, two interleaved repetitions per side, median ([`kernel_gmm_gpu.json`](kernel_gmm_gpu.json), runs `ab6_*`):

| dtype | tokens | op | time before → after (ms) | XLA temp before → after (MB) |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 4,096 | forward | 1.28 → 1.29 | 85.2 → 85.2 |
| fp32 | 4,096 | train step | 4.65 → 4.65 | 136 → 136 |
| fp32 | 32,768 | forward | 8.63 → 9.26 | 646 → 581 |
| fp32 | 32,768 | train step | 31.9 → 31.8 | 1,034 → 774 |
| fp32 | 131,072 | forward | 33.9 → 36.6 | 2,570 → 1,339 |
| fp32 | 131,072 | train step | 131 → 131 | 4,111 → 1,852 |
| bf16 | 4,096 | forward | 0.872 → 0.861 | 50.7 → 50.7 |
| bf16 | 4,096 | train step | 2.74 → 2.74 | 59.0 → 59.0 |
| bf16 | 32,768 | forward | 4.88 → 5.01 | 324 → 307 |
| bf16 | 32,768 | train step | 17.0 → 16.0 | 524 → 386 |
| bf16 | 131,072 | forward | 18.7 → 19.3 | 1,289 → 697 |
| bf16 | 131,072 | train step | 68.8 → 65.3 | 2,060 → 1,032 |

Largest bf16 train step that fit on the 24 GB GPU (fresh process):

| code | routing | dropless_impl | dispatch | max tokens |
| --- | --- | --- | --- | ---: |
| before | dropless | pallas | sorted | 524,288 |
| before | dropless | pallas | fused | 524,288 |
| before | capacity | auto | sorted | 262,144 |
| after | dropless | pallas | sorted | 1,048,576 |
| after | dropless | pallas | fused_gsum | 524,288 |
| after | dropless | blocked | sorted | 65,536 |
| after | dropless | ragged_dot | sorted | 32,768 |
| after | dense | auto | sorted | 65,536 |

Device peak of the bf16 sorted-Pallas train step (MB):

| tokens | before | after |
| --- | ---: | ---: |
| 131,072 | 2,476 | 1,448 |
| 262,144 | 4,652 | 2,605 |
| 524,288 | 9,004 | 4,910 |
| 1,048,576 | OOM | 9,558 |

**All GPU paths**, 32768 tokens, final code. Columns 4–7: 8 experts top-2, d_hidden 1024; columns 8–9: 64 experts top-4, d_hidden 256 (ragged_dot was not run there). `capacity` drops tokens beyond the expert capacity and is therefore not exact:

| routing | impl | dispatch | E8 fp32 fwd | E8 fp32 train | E8 bf16 fwd | E8 bf16 train | E64 bf16 fwd | E64 bf16 train |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dropless | pallas | sorted | 11.6 | 32.3 | 5.04 | 16.0 | 3.79 | 13.4 |
| dropless | pallas | gather | 9.15 | 37.7 | 5.12 | 19.6 | 4.24 | 13.3 |
| dropless | pallas | fused | 11.4 | 37.0 | 5.47 | 19.3 | 4.61 | 12.6 |
| dropless | pallas | fused_gsum | 8.64 | 33.4 | 4.59 | 20.7 | 2.99 | 11.0 |
| dropless | blocked | sorted | 15.9 | 45.6 | 15.6 | 36.1 | 8.11 | 21.3 |
| dropless | ragged_dot | sorted | 49.9 | 159 | 55.1 | 151 | OOM | OOM |
| capacity | auto | sorted | 10.2 | 26.8 | 6.45 | 14.0 | 4.54 | 11.2 |
| dense | auto | sorted | 29.0 | 76.5 | 15.3 | 38.0 | — | — |

**CPU**, fp32, 8 experts, top-2 ([`kernel_gmm_cpu.json`](kernel_gmm_cpu.json)). `chunked` is the new default; `blocked` was the best CPU dropless path before:

| routing | impl | 1k fwd | 1k train | 4k fwd | 4k train | 16k fwd | 16k train | 16k train temp (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dropless | chunked | 29.6 | 125 | 78.2 | 278 | 192 | 619 | 65.8 |
| dropless | blocked | 80.4 | 125 | 151 | 339 | 506 | 1,015 | 1,913 |
| dropless | ragged_dot | 108 | 245 | 434 | 1,129 | 1,733 | 4,207 | 3,584 |
| capacity | auto | 11.2 | 77.2 | 63.5 | 197 | 262 | 767 | 1,217 |
| dense | auto | 50.6 | 168 | 187 | 531 | 842 | 2,143 | 3,585 |

**Accuracy.** Relative output error against an fp32 dense-routing reference: 1.1e-02 for fp32 Pallas on GPU (TF32 matmuls), 3.2e-02 in bf16, 3.6e-08 for the CPU chunked path.

**When to use which.** Keep the defaults. On GPU at 8 experts top-2 the sorted Pallas dispatch has the fastest exact train step, while `dropless_dispatch='fused_gsum'` has the fastest forward; with many small experts (64, top-4) `fused_gsum` also wins the train step. `ragged_dot` is far slower on this GPU. On CPU `chunked` is the fastest exact path from 4k tokens up, ties `blocked` at 1k, and needs a fraction of the temporaries.

## Reproduce

```bash
# Cross-cutting before/after on identical shapes (CUDA GPU).
python -m docs.scripts.benchmark_gpu_kernels --output docs/benchmarks/gpu_kernels.json

# Per-family runners; add JAX_PLATFORMS=cpu and --device cpu for the CPU files.
python -m docs.scripts.benchmark_kernel_ssd --device gpu --output docs/benchmarks/kernel_ssd_gpu.json
python -m docs.scripts.benchmark_kernel_mamba --device gpu --output docs/benchmarks/kernel_mamba_gpu.json
python -m docs.scripts.benchmark_kernel_rla --device gpu --suites kernel layer \
  --output docs/benchmarks/kernel_rla_gpu.json
python -m docs.scripts.benchmark_kernel_delta --device gpu --output docs/benchmarks/kernel_delta_gpu.json
python -m docs.scripts.benchmark_kernel_kmv --device gpu --output docs/benchmarks/kernel_kmv_gpu.json
python -m docs.scripts.benchmark_kernel_ot --device gpu --suite lse
python -m docs.scripts.benchmark_kernel_gmm --device gpu --output docs/benchmarks/kernel_gmm_gpu.json
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_ssd --device cpu \
  --output docs/benchmarks/kernel_ssd_cpu.json
```

The Mamba runner's `--peak` and `--max-len` flags, and the `peak` / `scale` / `cfm` suites of the OT runner, produce the memory and largest-size rows.
