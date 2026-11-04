# GPU attention benchmarks

probjax's attention kernels (`mha`, `flex_attention` and the default `dot_product_attention` that every probjax transformer uses as `attention_fn`) against `jax.nn.dot_product_attention` (XLA and cuDNN backends) and PyTorch `scaled_dot_product_attention` pinned to its flash, memory-efficient and math backends. Measured on an NVIDIA GeForce RTX 3090 (24 GB), JAX 0.11.2, torch 2.14.1+cu130. See the [overview](gpu_overview.md) for methodology, the bug list and the default dispatch table.

Inputs are `[batch, seq, heads, head_dim]` with batch 4 and 8 heads unless stated otherwise; torch receives the transposed layout it expects. Times are medians in milliseconds. OOM: out of device memory. fail†: XLA could not profile any kernel configuration at this size during autotuning (reported as `NOT_FOUND`), so the case did not run. unsup.: the implementation rejects the shape or backend. —: not measured. cuDNN and torch flash have no fp32 path and torch math was not run beyond T=4096, so those cells are empty.

## Comparison with JAX and PyTorch

Source: [`gpu_attention.json`](gpu_attention.json).

### bf16, head_dim 128, non-causal, forward + backward

| Implementation | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| probjax `dot_product_attention` (default) | 0.443 | 1.49 | 5.35 | 20.2 | 80.0 |
| probjax `flex_attention` | 0.443 | 1.50 | 5.38 | 20.2 | 80.0 |
| probjax `mha` kernel | 0.443 | 1.49 | 5.39 | 20.2 | 80.3 |
| `jax.nn` cuDNN | 0.441 | 1.32 | 4.36 | 15.9 | 60.7 |
| `jax.nn` XLA | 0.545 | 1.87 | 6.77 | 25.5 | fail† |
| torch SDPA flash | 0.462 | 1.34 | 4.28 | 15.1 | 57.4 |
| torch SDPA mem-efficient | 1.11 | 3.91 | 15.3 | 61.0 | 245 |
| torch SDPA math | 1.64 | 5.61 | 20.0 | OOM | — |

### bf16, head_dim 64, non-causal, forward + backward

| Implementation | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| probjax `dot_product_attention` (default) | 0.207 | 0.681 | 2.45 | 9.37 | 39.0 |
| probjax `flex_attention` | 0.207 | 0.681 | 2.46 | 9.34 | 39.1 |
| probjax `mha` kernel | 0.206 | 0.681 | 2.46 | 9.36 | 39.2 |
| `jax.nn` cuDNN | 0.237 | 0.732 | 2.37 | 8.23 | 30.7 |
| `jax.nn` XLA | 0.459 | 1.61 | 6.12 | 23.3 | OOM |
| torch SDPA flash | 0.234 | 0.728 | 2.25 | 7.78 | 28.8 |
| torch SDPA mem-efficient | 0.324 | 1.06 | 3.55 | 13.0 | 50.8 |
| torch SDPA math | 1.37 | 4.83 | 17.6 | OOM | — |

### bf16, head_dim 128, causal, forward + backward

| Implementation | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| probjax `dot_product_attention` (default) | 0.311 | 0.937 | 3.17 | 11.7 | 45.0 |
| probjax `flex_attention` | 0.311 | 0.937 | 3.18 | 11.7 | 45.1 |
| probjax `mha` kernel | 0.311 | 0.937 | 3.18 | 11.7 | 45.6 |
| `jax.nn` cuDNN | 0.341 | 0.874 | 2.60 | 8.78 | 32.0 |
| `jax.nn` XLA | 0.611 | 2.27 | 7.66 | 30.1 | fail† |
| torch SDPA flash | 0.357 | 0.897 | 2.59 | 8.50 | 30.6 |
| torch SDPA mem-efficient | 0.681 | 2.21 | 7.97 | 30.4 | 125 |
| torch SDPA math | 1.72 | 6.04 | 21.7 | OOM | — |

### bf16, head_dim 128, non-causal, forward

| Implementation | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| probjax `dot_product_attention` (default) | 0.097 | 0.340 | 1.18 | 4.36 | 16.9 |
| probjax `flex_attention` | 0.096 | 0.340 | 1.17 | 4.35 | 16.9 |
| probjax `mha` kernel | 0.108 | 0.367 | 1.20 | 4.72 | 16.8 |
| `jax.nn` cuDNN | 0.094 | 0.330 | 1.12 | 4.10 | 15.8 |
| `jax.nn` XLA | 0.198 | 0.650 | 2.34 | 8.86 | fail† |
| torch SDPA flash | 0.087 | 0.315 | 1.09 | 4.02 | 15.5 |
| torch SDPA mem-efficient | 0.126 | 0.422 | 1.55 | 5.93 | 23.3 |
| torch SDPA math | 0.671 | 2.30 | 8.40 | 32.7 | — |

### fp32, head_dim 64, non-causal, forward + backward

| Implementation | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| probjax `dot_product_attention` (default) | 0.445 | 1.55 | 5.87 | 23.6 | 101 |
| probjax `flex_attention` | 0.444 | 1.55 | 5.87 | 23.7 | 101 |
| probjax `mha` kernel | 0.444 | 1.55 | 5.85 | 23.6 | 101 |
| `jax.nn` cuDNN | — | — | — | — | — |
| `jax.nn` XLA | 0.606 | 1.94 | 7.64 | 29.3 | OOM |
| torch SDPA flash | — | — | — | — | — |
| torch SDPA mem-efficient | 1.27 | 4.39 | 16.1 | 63.4 | 254 |
| torch SDPA math | 1.23 | 4.48 | 16.5 | OOM | — |

### fp32, head_dim 128, non-causal, forward + backward

| Implementation | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| probjax `dot_product_attention` (default) | 0.986 | 3.30 | 12.1 | 46.2 | 185 |
| probjax `flex_attention` | 0.987 | 3.29 | 12.1 | 46.3 | 185 |
| probjax `mha` kernel | 0.986 | 3.29 | 12.1 | 46.2 | 185 |
| `jax.nn` cuDNN | — | — | — | — | — |
| `jax.nn` XLA | 0.841 | 2.61 | 10.0 | 38.2 | OOM |
| torch SDPA flash | — | — | — | — | — |
| torch SDPA mem-efficient | 2.32 | 8.02 | 30.0 | 116 | 456 |
| torch SDPA math | 1.43 | 5.10 | 18.6 | OOM | — |

### Where cuDNN and PyTorch are faster

- **bf16, head_dim 128, T ≥ 1024, training.** cuDNN and torch flash are faster than the probjax kernel: at T=4096 fwd+bwd, cuDNN takes 15.9 ms and torch flash 15.1 ms against 20.2 ms for the probjax default; at T=8192 60.7 / 57.4 vs 80.0 ms. The causal variant shows the same gap (T=8192: cuDNN 32.0 ms, probjax 45.0 ms). If you train bf16 models with 128-dim heads on long sequences and need neither masks beyond causal nor probjax-specific features, `jax.nn.dot_product_attention(implementation='cudnn')` is the faster choice on this GPU.
- **bf16, head_dim 64.** Within a few percent at T ≤ 1024 and up to 39.0 vs 30.7 ms (probjax vs cuDNN, fwd+bwd T=8192) at long sequences. In the forward pass alone probjax is on par with cuDNN at every length (T=4096: 2.12 vs 2.19 ms).
- **fp32, head_dim 128, non-causal training.** The dense XLA einsum (`jax.nn` XLA) is faster while it fits: T=4096 38.2 ms vs 46.2 ms for probjax. It runs out of memory at T=8192, where probjax takes 185 ms. Neither cuDNN nor torch flash support fp32; torch's fp32 options (mem-efficient, math) are slower than probjax from T=512 on.
- **Short sequences.** At T=128 probjax and `jax.nn` XLA are the fastest in fwd+bwd; cuDNN and torch pay a fixed launch overhead (bf16 d64: probjax 0.049 ms, cuDNN 0.113 ms, torch flash 0.241 ms).

`probjax_default`, `probjax_flex` and `probjax_mha` coincide within noise because the default now dispatches to `flex_attention`, which calls the `mha` kernel. `jax.nn` XLA materializes the score matrix and fails at T=8192 (OOM or fail†) in every configuration.

??? note "Full grid: all 16 configurations, including ragged T = 600 / 1500 / 3000"

    **bf16, head_dim 64, non-causal, forward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.035 | 0.054 | 0.087 | 0.157 | 0.386 | 0.558 | 1.30 | 2.12 | 8.42 |
    | probjax `flex_attention` | 0.039 | 0.054 | 0.087 | 0.156 | 0.387 | 0.558 | 1.30 | 2.11 | 8.40 |
    | probjax `mha` kernel | 0.035 | 0.054 | 0.092 | 0.162 | 0.396 | 0.568 | 1.31 | 2.12 | 8.45 |
    | `jax.nn` cuDNN | 0.058 | 0.059 | 0.108 | 0.170 | 0.365 | 0.631 | 1.18 | 2.19 | 8.19 |
    | `jax.nn` XLA | 0.041 | 0.160 | 0.218 | 0.562 | 1.16 | 2.05 | 4.42 | 7.84 | fail† |
    | torch SDPA flash | 0.025 | 0.046 | 0.062 | 0.162 | 0.314 | 0.553 | 1.20 | 2.05 | 7.83 |
    | torch SDPA mem-efficient | 0.032 | 0.061 | 0.077 | 0.200 | 0.415 | 0.711 | 1.50 | 2.67 | 10.5 |
    | torch SDPA math | 0.161 | 0.567 | 0.734 | 1.99 | 4.03 | 7.33 | 16.1 | 28.7 | — |

    **bf16, head_dim 64, non-causal, forward + backward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.049 | 0.207 | 0.312 | 0.681 | 1.57 | 2.45 | 5.57 | 9.37 | 39.0 |
    | probjax `flex_attention` | 0.049 | 0.207 | 0.313 | 0.681 | 1.57 | 2.46 | 5.59 | 9.34 | 39.1 |
    | probjax `mha` kernel | 0.049 | 0.206 | 0.311 | 0.681 | 1.54 | 2.46 | 5.71 | 9.36 | 39.2 |
    | `jax.nn` cuDNN | 0.113 | 0.237 | 0.339 | 0.732 | 1.41 | 2.37 | 4.89 | 8.23 | 30.7 |
    | `jax.nn` XLA | 0.054 | 0.459 | 0.741 | 1.61 | 3.37 | 6.12 | 12.7 | 23.3 | OOM |
    | torch SDPA flash | 0.241 | 0.234 | 0.295 | 0.728 | 1.32 | 2.25 | 4.61 | 7.78 | 28.8 |
    | torch SDPA mem-efficient | 0.250 | 0.324 | 0.443 | 1.06 | 2.21 | 3.55 | 7.81 | 13.0 | 50.8 |
    | torch SDPA math | 0.643 | 1.37 | 1.74 | 4.83 | 9.63 | 17.6 | 37.8 | OOM | — |

    **bf16, head_dim 64, causal, forward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.036 | 0.042 | 0.074 | 0.102 | 0.260 | 0.332 | 0.788 | 1.19 | 4.49 |
    | probjax `flex_attention` | 0.037 | 0.042 | 0.074 | 0.103 | 0.261 | 0.333 | 0.788 | 1.19 | 4.49 |
    | probjax `mha` kernel | 0.035 | 0.042 | 0.077 | 0.109 | 0.261 | 0.342 | 0.829 | 1.21 | 4.54 |
    | `jax.nn` cuDNN | 0.058 | 0.069 | 0.083 | 0.138 | 0.256 | 0.401 | 0.782 | 1.30 | 4.63 |
    | `jax.nn` XLA | 0.040 | 0.206 | 0.264 | 0.593 | 1.20 | 2.38 | 4.40 | 7.84 | fail† |
    | torch SDPA flash | 0.025 | 0.039 | 0.054 | 0.115 | 0.225 | 0.341 | 0.726 | 1.17 | 4.31 |
    | torch SDPA mem-efficient | 0.032 | 0.047 | 0.064 | 0.128 | 0.255 | 0.423 | 0.837 | 1.48 | 5.55 |
    | torch SDPA math | 0.199 | 0.663 | 0.861 | 2.46 | 5.05 | 9.23 | 20.4 | 36.6 | — |

    **bf16, head_dim 64, causal, forward + backward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.051 | 0.139 | 0.214 | 0.412 | 0.911 | 1.40 | 3.06 | 5.12 | 19.8 |
    | probjax `flex_attention` | 0.052 | 0.139 | 0.214 | 0.409 | 0.911 | 1.40 | 3.06 | 5.13 | 19.8 |
    | probjax `mha` kernel | 0.050 | 0.139 | 0.213 | 0.412 | 0.923 | 1.40 | 3.14 | 5.15 | 19.8 |
    | `jax.nn` cuDNN | 0.116 | 0.217 | 0.299 | 0.541 | 0.986 | 1.54 | 2.95 | 4.95 | 17.5 |
    | `jax.nn` XLA | 0.055 | 0.529 | 0.751 | 2.02 | 3.57 | 7.05 | 12.8 | 28.0 | OOM |
    | torch SDPA flash | 0.241 | 0.237 | 0.267 | 0.527 | 0.948 | 1.45 | 2.77 | 4.57 | 16.0 |
    | torch SDPA mem-efficient | 0.224 | 0.257 | 0.355 | 0.693 | 1.39 | 2.14 | 4.42 | 7.43 | 27.8 |
    | torch SDPA math | 0.629 | 1.45 | 1.86 | 5.27 | 10.6 | 19.3 | 41.7 | OOM | — |

    **bf16, head_dim 128, non-causal, forward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.036 | 0.097 | 0.174 | 0.340 | 0.774 | 1.18 | 2.77 | 4.36 | 16.9 |
    | probjax `flex_attention` | 0.036 | 0.096 | 0.173 | 0.340 | 0.774 | 1.17 | 2.77 | 4.35 | 16.9 |
    | probjax `mha` kernel | 0.036 | 0.108 | 0.186 | 0.367 | 0.845 | 1.20 | 3.00 | 4.72 | 16.8 |
    | `jax.nn` cuDNN | 0.056 | 0.094 | 0.119 | 0.330 | 0.622 | 1.12 | 2.36 | 4.10 | 15.8 |
    | `jax.nn` XLA | 0.047 | 0.198 | 0.271 | 0.650 | 1.38 | 2.34 | 5.07 | 8.86 | fail† |
    | torch SDPA flash | 0.026 | 0.087 | 0.109 | 0.315 | 0.590 | 1.09 | 2.29 | 4.02 | 15.5 |
    | torch SDPA mem-efficient | 0.033 | 0.126 | 0.158 | 0.422 | 0.893 | 1.55 | 3.34 | 5.93 | 23.3 |
    | torch SDPA math | 0.159 | 0.671 | 0.876 | 2.30 | 4.67 | 8.40 | 18.4 | 32.7 | — |

    **bf16, head_dim 128, non-causal, forward + backward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.058 | 0.443 | 0.663 | 1.49 | 3.48 | 5.35 | 12.8 | 20.2 | 80.0 |
    | probjax `flex_attention` | 0.052 | 0.443 | 0.663 | 1.50 | 3.48 | 5.38 | 12.8 | 20.2 | 80.0 |
    | probjax `mha` kernel | 0.053 | 0.443 | 0.676 | 1.49 | 3.51 | 5.39 | 12.8 | 20.2 | 80.3 |
    | `jax.nn` cuDNN | 0.109 | 0.441 | 0.587 | 1.32 | 2.82 | 4.36 | 9.97 | 15.9 | 60.7 |
    | `jax.nn` XLA | 0.074 | 0.545 | 0.864 | 1.87 | 3.87 | 6.77 | 14.3 | 25.5 | fail† |
    | torch SDPA flash | 0.253 | 0.462 | 0.571 | 1.34 | 2.53 | 4.28 | 8.66 | 15.1 | 57.4 |
    | torch SDPA mem-efficient | 0.247 | 1.11 | 1.56 | 3.91 | 8.81 | 15.3 | 31.7 | 61.0 | 245 |
    | torch SDPA math | 0.613 | 1.64 | 2.11 | 5.61 | 11.1 | 20.0 | 42.8 | OOM | — |

    **bf16, head_dim 128, causal, forward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.038 | 0.078 | 0.162 | 0.218 | 0.558 | 0.699 | 1.73 | 2.47 | 9.32 |
    | probjax `flex_attention` | 0.037 | 0.087 | 0.162 | 0.218 | 0.560 | 0.699 | 1.73 | 2.47 | 9.35 |
    | probjax `mha` kernel | 0.037 | 0.078 | 0.159 | 0.243 | 0.552 | 0.769 | 1.70 | 2.70 | 10.1 |
    | `jax.nn` cuDNN | 0.057 | 0.079 | 0.106 | 0.210 | 0.399 | 0.651 | 1.33 | 2.26 | 8.46 |
    | `jax.nn` XLA | 0.038 | 0.239 | 0.319 | 0.688 | 1.43 | 2.63 | 5.07 | 8.86 | fail† |
    | torch SDPA flash | 0.026 | 0.063 | 0.084 | 0.181 | 0.359 | 0.604 | 1.23 | 2.23 | 8.54 |
    | torch SDPA mem-efficient | 0.032 | 0.083 | 0.116 | 0.254 | 0.513 | 0.862 | 1.77 | 3.17 | 12.1 |
    | torch SDPA math | 0.198 | 0.765 | 1.00 | 2.78 | 5.68 | 10.3 | 22.6 | 40.4 | — |

    **bf16, head_dim 128, causal, forward + backward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.055 | 0.311 | 0.467 | 0.937 | 2.05 | 3.17 | 7.00 | 11.7 | 45.0 |
    | probjax `flex_attention` | 0.051 | 0.311 | 0.467 | 0.937 | 2.04 | 3.18 | 7.01 | 11.7 | 45.1 |
    | probjax `mha` kernel | 0.050 | 0.311 | 0.475 | 0.937 | 2.07 | 3.18 | 7.09 | 11.7 | 45.6 |
    | `jax.nn` cuDNN | 0.119 | 0.341 | 0.468 | 0.874 | 1.72 | 2.60 | 5.53 | 8.78 | 32.0 |
    | `jax.nn` XLA | 0.075 | 0.611 | 0.874 | 2.27 | 4.05 | 7.66 | 14.3 | 30.1 | fail† |
    | torch SDPA flash | 0.240 | 0.357 | 0.462 | 0.897 | 1.63 | 2.59 | 4.96 | 8.50 | 30.6 |
    | torch SDPA mem-efficient | 0.226 | 0.681 | 0.959 | 2.21 | 4.63 | 7.97 | 16.8 | 30.4 | 125 |
    | torch SDPA math | 0.639 | 1.72 | 2.22 | 6.04 | 12.0 | 21.7 | 46.7 | OOM | — |

    **fp32, head_dim 64, non-causal, forward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.042 | 0.102 | 0.175 | 0.328 | 0.803 | 1.19 | 2.81 | 4.56 | 18.2 |
    | probjax `flex_attention` | 0.036 | 0.101 | 0.175 | 0.328 | 0.809 | 1.19 | 2.80 | 4.53 | 18.0 |
    | probjax `mha` kernel | 0.036 | 0.107 | 0.174 | 0.331 | 0.801 | 1.18 | 2.80 | 4.55 | 18.1 |
    | `jax.nn` cuDNN | — | — | — | — | — | — | — | — | — |
    | `jax.nn` XLA | 0.037 | 0.231 | 0.317 | 0.794 | 1.63 | 2.85 | 5.98 | 10.8 | fail† |
    | torch SDPA flash | — | — | — | — | — | — | — | — | — |
    | torch SDPA mem-efficient | 0.033 | 0.320 | 0.417 | 1.12 | 2.34 | 4.09 | 8.74 | 16.1 | 68.3 |
    | torch SDPA math | 0.129 | 0.469 | 0.612 | 1.69 | 3.46 | 6.32 | 14.2 | 24.8 | — |

    **fp32, head_dim 64, non-causal, forward + backward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.047 | 0.445 | 0.681 | 1.55 | 3.50 | 5.87 | 13.4 | 23.6 | 101 |
    | probjax `flex_attention` | 0.043 | 0.444 | 0.681 | 1.55 | 3.49 | 5.87 | 13.3 | 23.7 | 101 |
    | probjax `mha` kernel | 0.043 | 0.444 | 0.695 | 1.55 | 3.53 | 5.85 | 13.5 | 23.6 | 101 |
    | `jax.nn` cuDNN | — | — | — | — | — | — | — | — | — |
    | `jax.nn` XLA | 0.073 | 0.606 | 0.820 | 1.94 | 4.38 | 7.64 | 16.2 | 29.3 | OOM |
    | torch SDPA flash | — | — | — | — | — | — | — | — | — |
    | torch SDPA mem-efficient | 0.222 | 1.27 | 1.64 | 4.39 | 9.20 | 16.1 | 34.3 | 63.4 | 254 |
    | torch SDPA math | 0.504 | 1.23 | 1.58 | 4.48 | 8.97 | 16.5 | 35.7 | OOM | — |

    **fp32, head_dim 64, causal, forward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.039 | 0.075 | 0.149 | 0.213 | 0.541 | 0.702 | 1.66 | 2.55 | 9.75 |
    | probjax `flex_attention` | 0.040 | 0.075 | 0.149 | 0.213 | 0.539 | 0.702 | 1.66 | 2.55 | 9.76 |
    | probjax `mha` kernel | 0.037 | 0.082 | 0.147 | 0.213 | 0.532 | 0.710 | 1.69 | 2.55 | 9.76 |
    | `jax.nn` cuDNN | — | — | — | — | — | — | — | — | — |
    | `jax.nn` XLA | 0.039 | 0.265 | 0.352 | 0.815 | 1.63 | 2.89 | 5.99 | 10.9 | fail† |
    | torch SDPA flash | — | — | — | — | — | — | — | — | — |
    | torch SDPA mem-efficient | 0.033 | 0.213 | 0.297 | 0.661 | 1.35 | 2.33 | 4.85 | 8.90 | 35.7 |
    | torch SDPA math | 0.168 | 0.562 | 0.740 | 2.17 | 4.46 | 8.19 | 18.3 | 32.7 | — |

    **fp32, head_dim 64, causal, forward + backward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.052 | 0.268 | 0.439 | 0.873 | 1.97 | 3.08 | 6.88 | 11.8 | 48.2 |
    | probjax `flex_attention` | 0.051 | 0.268 | 0.436 | 0.873 | 1.96 | 3.09 | 6.92 | 11.9 | 48.9 |
    | probjax `mha` kernel | 0.054 | 0.269 | 0.447 | 0.873 | 2.00 | 3.09 | 7.18 | 11.8 | 48.8 |
    | `jax.nn` cuDNN | — | — | — | — | — | — | — | — | — |
    | `jax.nn` XLA | 0.087 | 0.625 | 0.860 | 2.13 | 4.53 | 7.64 | 16.2 | 28.5 | OOM |
    | torch SDPA flash | — | — | — | — | — | — | — | — | — |
    | torch SDPA mem-efficient | 0.223 | 0.805 | 1.12 | 2.52 | 5.12 | 8.80 | 18.1 | 32.9 | 131 |
    | torch SDPA math | 0.537 | 1.31 | 1.69 | 4.90 | 9.87 | 18.2 | 39.5 | OOM | — |

    **fp32, head_dim 128, non-causal, forward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.037 | 0.209 | 0.360 | 0.704 | 1.72 | 2.58 | 6.01 | 9.95 | 39.4 |
    | probjax `flex_attention` | 0.033 | 0.209 | 0.362 | 0.704 | 1.72 | 2.58 | 6.01 | 9.94 | 39.7 |
    | probjax `mha` kernel | 0.040 | 0.226 | 0.359 | 0.704 | 1.72 | 2.71 | 6.11 | 9.96 | 41.1 |
    | `jax.nn` cuDNN | — | — | — | — | — | — | — | — | — |
    | `jax.nn` XLA | 0.049 | 0.322 | 0.451 | 1.05 | 2.28 | 3.78 | 8.11 | 14.4 | fail† |
    | torch SDPA flash | — | — | — | — | — | — | — | — | — |
    | torch SDPA mem-efficient | 0.044 | 0.490 | 0.608 | 1.70 | 3.62 | 6.37 | 13.7 | 24.4 | 98.2 |
    | torch SDPA math | 0.132 | 0.547 | 0.721 | 1.96 | 4.01 | 7.28 | 16.2 | 28.9 | — |

    **fp32, head_dim 128, non-causal, forward + backward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.095 | 0.986 | 1.40 | 3.30 | 7.39 | 12.1 | 27.0 | 46.2 | 185 |
    | probjax `flex_attention` | 0.095 | 0.987 | 1.39 | 3.29 | 7.40 | 12.1 | 26.9 | 46.3 | 185 |
    | probjax `mha` kernel | 0.095 | 0.986 | 1.42 | 3.29 | 7.46 | 12.1 | 27.2 | 46.2 | 185 |
    | `jax.nn` cuDNN | — | — | — | — | — | — | — | — | — |
    | `jax.nn` XLA | 0.098 | 0.841 | 1.18 | 2.61 | 5.99 | 10.0 | 21.9 | 38.2 | OOM |
    | torch SDPA flash | — | — | — | — | — | — | — | — | — |
    | torch SDPA mem-efficient | 0.238 | 2.32 | 2.97 | 8.02 | 17.0 | 30.0 | 64.1 | 116 | 456 |
    | torch SDPA math | 0.494 | 1.43 | 1.84 | 5.10 | 10.3 | 18.6 | 40.3 | OOM | — |

    **fp32, head_dim 128, causal, forward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.032 | 0.146 | 0.297 | 0.429 | 1.10 | 1.44 | 3.45 | 5.35 | 20.7 |
    | probjax `flex_attention` | 0.032 | 0.146 | 0.297 | 0.427 | 1.10 | 1.44 | 3.46 | 5.33 | 20.6 |
    | probjax `mha` kernel | 0.035 | 0.146 | 0.296 | 0.425 | 1.10 | 1.53 | 3.56 | 5.53 | 20.9 |
    | `jax.nn` cuDNN | — | — | — | — | — | — | — | — | — |
    | `jax.nn` XLA | 0.049 | 0.356 | 0.485 | 1.06 | 2.26 | 3.79 | 8.04 | 14.3 | fail† |
    | torch SDPA flash | — | — | — | — | — | — | — | — | — |
    | torch SDPA mem-efficient | 0.043 | 0.308 | 0.444 | 0.989 | 2.03 | 3.49 | 7.25 | 13.1 | 51.5 |
    | torch SDPA math | 0.168 | 0.639 | 0.846 | 2.41 | 5.01 | 9.12 | 20.3 | 36.6 | — |

    **fp32, head_dim 128, causal, forward + backward**

    | Implementation | T=128 | T=512 | T=600 | T=1024 | T=1500 | T=2048 | T=3000 | T=4096 | T=8192 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | probjax `dot_product_attention` (default) | 0.084 | 0.617 | 0.937 | 1.88 | 4.24 | 6.57 | 14.6 | 24.5 | 96.7 |
    | probjax `flex_attention` | 0.092 | 0.618 | 0.937 | 1.88 | 4.25 | 6.54 | 14.6 | 24.5 | 97.0 |
    | probjax `mha` kernel | 0.084 | 0.618 | 0.958 | 1.88 | 4.30 | 6.56 | 14.9 | 24.5 | 96.5 |
    | `jax.nn` cuDNN | — | — | — | — | — | — | — | — | — |
    | `jax.nn` XLA | 0.117 | 0.856 | 1.22 | 2.80 | 6.05 | 9.93 | 21.7 | 37.1 | OOM |
    | torch SDPA flash | — | — | — | — | — | — | — | — | — |
    | torch SDPA mem-efficient | 0.240 | 1.45 | 2.01 | 4.61 | 9.39 | 16.1 | 33.4 | 60.4 | 233 |
    | torch SDPA math | 0.542 | 1.51 | 1.96 | 5.52 | 11.1 | 20.2 | 44.0 | OOM | — |

## Changes to the probjax kernels

### Tile heuristics for consumer GPUs

On GPUs with less than 150 KB of shared memory per block (the RTX 3090 opts into about 99 KB), the previous tile heuristic picked configurations that either spilled or were rejected. It was replaced by shared-memory estimators fitted to Triton's sm_86 reports plus a tuned table per (dtype, head_dim). Auto-selected tiles, `mha`, batch 4, 8 heads, median of three interleaved runs per side ([`kernel_attn_gpu_tiles_auto.json`](kernel_attn_gpu_tiles_auto.json)):

| dtype | head_dim | T | mode | before (ms) | after (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 32 | 1024 | fwd | 0.101 | 0.090 |
| bf16 | 32 | 1024 | fwd+bwd | 0.377 | 0.373 |
| bf16 | 32 | 4096 | fwd | 1.18 | 1.16 |
| bf16 | 32 | 4096 | fwd+bwd | 5.09 | 5.10 |
| bf16 | 64 | 1024 | fwd | 0.204 | 0.162 |
| bf16 | 64 | 1024 | fwd+bwd | 0.755 | 0.680 |
| bf16 | 64 | 4096 | fwd | 2.36 | 2.13 |
| bf16 | 64 | 4096 | fwd+bwd | 10.0 | 9.34 |
| bf16 | 128 | 1024 | fwd | 0.414 | 0.359 |
| bf16 | 128 | 1024 | fwd+bwd | 1.61 | 1.49 |
| bf16 | 128 | 4096 | fwd | 5.26 | 4.72 |
| bf16 | 128 | 4096 | fwd+bwd | 21.9 | 20.1 |
| fp32 | 32 | 1024 | fwd | 0.185 | 0.179 |
| fp32 | 32 | 1024 | fwd+bwd | 0.878 | 0.875 |
| fp32 | 32 | 4096 | fwd | 2.33 | 2.27 |
| fp32 | 32 | 4096 | fwd+bwd | 12.5 | 12.6 |
| fp32 | 64 | 1024 | fwd | 0.348 | 0.379 |
| fp32 | 64 | 1024 | fwd+bwd | 4.06 | 1.55 |
| fp32 | 64 | 4096 | fwd | 4.93 | 4.53 |
| fp32 | 64 | 4096 | fwd+bwd | 64.4 | 23.3 |
| fp32 | 128 | 1024 | fwd | 0.970 | 0.699 |
| fp32 | 128 | 1024 | fwd+bwd | 4.07 | 3.29 |
| fp32 | 128 | 4096 | fwd | 13.8 | 9.87 |
| fp32 | 128 | 4096 | fwd+bwd | 60.1 | 45.9 |

The fp32 head_dim 64 rows were re-measured after a follow-up change ([`kernel_attn_gpu_tiles_auto_fp32d64.json`](kernel_attn_gpu_tiles_auto_fp32d64.json)); these are the final numbers for that configuration:

| T | mode | before (ms) | after (ms) |
| --- | ---: | ---: | ---: |
| 1024 | fwd | 0.360 | 0.330 |
| 1024 | fwd+bwd | 4.09 | 1.54 |
| 4096 | fwd | 5.04 | 4.48 |
| 4096 | fwd+bwd | 64.8 | 23.5 |

The remaining tile cost is measured by forcing every (block, stages, warps) combination and comparing with the auto choice ([`gpu_attention_tiles.json`](gpu_attention_tiles.json)). Configurations that exceed shared memory are rejected by Triton at compile time; the last column counts them.

| dtype | head_dim | T | mode | auto (ms) | best forced (ms) | best forced config | forced ok / SMEM-rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 64 | 2048 | fwd | 0.568 | 0.558 | q64/k64, 2 st, 4 w | 54 / 0 |
| bf16 | 64 | 2048 | fwd+bwd | 2.44 | 2.46 | fwd q64/k64, bwd 64/64, 2 st, 4 w | 48 / 0 |
| bf16 | 64 | 8192 | fwd | 8.46 | 7.98 | q128/k64, 2 st, 4 w | 54 / 0 |
| bf16 | 64 | 8192 | fwd+bwd | 38.9 | 39.6 | fwd q64/k64, bwd 64/64, 2 st, 4 w | 48 / 0 |
| bf16 | 128 | 2048 | fwd | 1.28 | 1.14 | q128/k32, 1 st, 4 w | 38 / 16 |
| bf16 | 128 | 2048 | fwd+bwd | 5.34 | 5.41 | fwd q128/k64, bwd 64/32, 2 st, 4 w | 36 / 12 |
| bf16 | 128 | 8192 | fwd | 16.7 | 15.9 | q128/k32, 1 st, 4 w | 38 / 16 |
| bf16 | 128 | 8192 | fwd+bwd | 79.6 | 80.3 | fwd q128/k64, bwd 64/32, 2 st, 4 w | 36 / 12 |
| fp32 | 64 | 2048 | fwd | 1.18 | 1.18 | q128/k64, 1 st, 4 w | 40 / 14 |
| fp32 | 64 | 2048 | fwd+bwd | 5.86 | 5.75 | fwd q64/k32, bwd 32/32, 2 st, 4 w | 44 / 4 |
| fp32 | 64 | 8192 | fwd | 18.0 | 16.7 | q128/k64, 1 st, 4 w | 40 / 14 |
| fp32 | 64 | 8192 | fwd+bwd | 101 | 99.9 | fwd q64/k32, bwd 32/32, 2 st, 4 w | 44 / 4 |
| fp32 | 128 | 2048 | fwd | 2.71 | 2.58 | q64/k32, 1 st, 4 w | 22 / 32 |
| fp32 | 128 | 2048 | fwd+bwd | 12.1 | 12.1 | fwd q64/k32, bwd 64/32, 1 st, 4 w | 20 / 28 |
| fp32 | 128 | 8192 | fwd | 39.7 | 38.1 | q128/k32, 1 st, 8 w | 22 / 32 |
| fp32 | 128 | 8192 | fwd+bwd | 185 | 186 | fwd q64/k32, bwd 64/32, 1 st, 4 w | 20 / 28 |

Where *best forced* is clearly below *auto*, the tuned table can still be improved; elsewhere the heuristic picks the best configuration or one within noise of it.

### Padding for ragged lengths

`flex_attention` used to pad sequences to the next power of two (600 → 1024, 3000 → 4096); it now pads to the next tile multiple (600 → 640, 3000 → 3072). `mha` itself returned NaN for lengths that were not a multiple of the block size and now pads and slices (see the bug list). bf16 and fp32, head_dim 64, fwd+bwd, before ([`kernel_attn_gpu_before.json`](kernel_attn_gpu_before.json); two values where the baseline was measured twice) → after ([`kernel_attn_gpu.json`](kernel_attn_gpu.json)):

| dtype | T | `flex_attention` before → after (ms) | `mha` before → after (ms) |
| --- | ---: | ---: | ---: |
| bf16 | 600 | 0.386 → 0.314 | 0.338 → 0.309 |
| bf16 | 1500 | 1.74 → 1.56 | 1.70 → 1.54 |
| bf16 | 3000 | 6.40 → 5.57 | 6.23 → 5.65 |
| fp32 | 600 | 1.56 → 0.679 | 1.48 → 0.691 |
| fp32 | 1500 | 8.49 → 3.49 | 8.54 → 3.54 |
| fp32 | 3000 | 30.9 → 13.3 | 31.6 → 13.5 |

## Default `attention_fn` dispatch

`probjax.nn.layers.attention.dot_product_attention` is the default `attention_fn` of `Transformer`, `DiffusionTransformer` and the attention conditioners. With `implementation=None` it now runs the fused kernel (`flex_attention`) on NVIDIA GPUs whenever that is exact:

- no mask, or a position-only mask (causal, length/window masks); no dense boolean mask,
- no additive bias, no dropout, default `precision`,
- at least 64 keys.

Otherwise, and on CPU, it uses the Flax einsum. `implementation='xla'` forces the einsum and `implementation='pallas'` forces the kernel. GQA inputs (fewer KV heads than query heads) also go to the kernel.

Before → after for the default itself (non-causal):

| dtype | head_dim | T | mode | before (ms) | after (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 64 | 512 | fwd | 0.200 | 0.054 |
| bf16 | 64 | 512 | fwd+bwd | 0.460 | 0.207 |
| bf16 | 64 | 1024 | fwd | 0.655 | 0.156 |
| bf16 | 64 | 1024 | fwd+bwd | 1.65 | 0.681 |
| bf16 | 64 | 2048 | fwd | 2.47 | 0.563 |
| bf16 | 64 | 2048 | fwd+bwd | 6.33 | 2.46 |
| bf16 | 64 | 3000 | fwd | 5.34 | 1.30 |
| bf16 | 64 | 3000 | fwd+bwd | 14.8 | 5.60 |
| bf16 | 64 | 4096 | fwd | 8.82 | 2.10 |
| bf16 | 64 | 4096 | fwd+bwd | 25.0 | 9.34 |
| bf16 | 128 | 512 | fwd | 0.196 | 0.096 |
| bf16 | 128 | 512 | fwd+bwd | 0.544 | 0.444 |
| bf16 | 128 | 1024 | fwd | 0.650 | 0.341 |
| bf16 | 128 | 1024 | fwd+bwd | 1.87 | 1.50 |
| bf16 | 128 | 2048 | fwd | 2.34 | 1.18 |
| bf16 | 128 | 2048 | fwd+bwd | 6.79 | 5.35 |

The 64-key threshold comes from the small-T sweep below ([`kernel_attn_gpu_dispatch.json`](kernel_attn_gpu_dispatch.json), batch 64, 8 heads, bf16, fwd+bwd): at 16 and 32 keys the einsum and the kernel take about the same time; from 64 keys on the kernel is faster.

**head_dim 32**

| Implementation | T=16 | T=32 | T=64 | T=128 | T=256 | T=512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.054 | 0.065 | 0.060 | 0.144 | 0.435 | 1.51 |
| `mha` | 0.050 | 0.052 | 0.058 | 0.155 | 0.466 | 1.59 |
| flax dense | 0.054 | 0.063 | 0.155 | 0.512 | 1.80 | 6.46 |
| `jax.nn` XLA | 0.058 | 0.059 | 0.155 | 0.523 | 1.79 | 6.47 |
| `jax.nn` cuDNN | 0.109 | 0.109 | 0.141 | 0.301 | 0.850 | 2.72 |

**head_dim 64**

| Implementation | T=16 | T=32 | T=64 | T=128 | T=256 | T=512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.049 | 0.089 | 0.110 | 0.266 | 0.784 | 2.72 |
| `mha` | 0.050 | 0.064 | 0.109 | 0.290 | 0.894 | 2.99 |
| flax dense | 0.059 | 0.090 | 0.224 | 0.681 | 2.04 | 6.92 |
| `jax.nn` XLA | 0.071 | 0.087 | 0.224 | 0.681 | 2.04 | 6.91 |
| `jax.nn` cuDNN | 0.140 | 0.164 | 0.234 | 0.419 | 1.00 | 2.91 |

## Memory

True device peak (`peak_bytes_in_use` read in a fresh process) of one fwd+bwd call, bf16, head_dim 64, batch 4, 8 heads, non-causal, in MB ([`kernel_attn_gpu_memory.json`](kernel_attn_gpu_memory.json)). The last column is the largest power-of-two length that ran; the search stopped at 65536, so 65536 means *at least* 65536.

| Implementation | T=1024 | T=4096 | T=16384 | T=32768 | T=65536 | largest T that ran |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| probjax default (now `flex_attention`) | 40.3 | 161 | 644 | 1,288 | 2,576 | 65,536 |
| flax `dot_product_attention` (dense, the previous default path) | 500 | 5,249 | OOM | — | — | 4,096 |
| `jax.nn` XLA | 500 | 5,249 | OOM | — | — | 4,096 |
| `jax.nn` cuDNN | 60.0 | 256 | 1,024 | 1,800 | 3,600 | 65,536 |
| torch flash | 56.3 | 225 | 900 | 1,800 | 3,600 | 65,536 |
| torch mem-efficient | 52.3 | 209 | 836 | 1,672 | 3,345 | 65,536 |
| torch math | 576 | 8,400 | OOM | — | — | 4,096 |

Bytes saved for the backward pass (`jax.vjp` residuals), batch 2, 8 heads, head_dim 64, in MB ([`kernel_attn_gpu_residuals.json`](kernel_attn_gpu_residuals.json)). The dense paths save the T×T probabilities; the kernels save the output and the log-sum-exp only, so residuals grow linearly in T.

| Implementation | bf16 T=1000 | bf16 T=2048 | fp32 T=1000 | fp32 T=2048 |
| --- | ---: | ---: | ---: | ---: |
| probjax default | 8.06 | 16.1 | 16.1 | 32.1 |
| probjax `mha` | 8.06 | 16.1 | 16.1 | 32.1 |
| flax dense | 97.5 | 396 | 134 | 536 |
| `jax.nn` XLA | 97.5 | 396 | 134 | 536 |
| `jax.nn` cuDNN | 7.87 | 16.1 | — | — |

## Grouped-query attention and decoding

The `mha` forward, JVP and backward kernels now read the KV head `h // G` directly instead of materializing repeated K/V; dK/dV of the query heads in a group are summed in fp32. `flex_attention` no longer calls `jnp.repeat`, and `dot_product_attention` routes GQA inputs to the kernel. Outputs are bit-identical to the repeat path in bf16.

**Training-shaped GQA**, 32 query heads, head_dim 128, batch 1, Lq = Lkv, median of three rounds ([`kernel_attn_ext_gpu.json`](kernel_attn_ext_gpu.json), suite `gqa`). Time is at parity; the saving is in temporaries and residuals.

bf16, T=4096:

| KV heads | Implementation | fwd (ms) | fwd+bwd (ms) | fwd XLA temp (MB) | bwd residuals (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | before: repeat K/V | 4.54 | 21.1 | 64.0 | 96.5 |
| 1 | after: native GQA | 4.45 | 20.5 | 0 | 32.5 |
| 1 | `jax.nn` cuDNN | 4.22 | 16.3 | ≈0 | — |
| 1 | torch SDPA | 4.14 | 15.5 | — | — |
| 8 | before: repeat K/V | 4.56 | 21.2 | 64.0 | 96.5 |
| 8 | after: native GQA | 4.94 | 21.5 | 0 | 32.5 |
| 8 | `jax.nn` cuDNN | 4.23 | 17.8 | ≈0 | — |
| 8 | torch SDPA | 4.59 | 15.7 | — | — |

??? note "bf16 T=1024, fp32 T=1024 and fp32 T=4096"

    bf16, T=1024:

    | KV heads | Implementation | fwd (ms) | fwd+bwd (ms) | fwd XLA temp (MB) | bwd residuals (MB) |
    | --- | --- | ---: | ---: | ---: | ---: |
    | 1 | before: repeat K/V | 0.367 | 1.56 | 16.0 | 24.1 |
    | 1 | after: native GQA | 0.343 | 1.53 | 0 | 8.13 |
    | 1 | `jax.nn` XLA | 0.590 | 2.07 | 192 | — |
    | 1 | `jax.nn` cuDNN | 0.332 | 1.36 | ≈0 | — |
    | 1 | torch SDPA | 0.316 | 1.36 | — | — |
    | 8 | before: repeat K/V | 0.377 | 1.57 | 16.0 | 24.1 |
    | 8 | after: native GQA | 0.351 | 1.54 | 0 | 8.13 |
    | 8 | `jax.nn` XLA | 0.772 | 2.08 | 192 | — |
    | 8 | `jax.nn` cuDNN | 0.334 | 1.37 | ≈0 | — |
    | 8 | torch SDPA | 0.315 | 1.43 | — | — |

    fp32, T=1024:

    | KV heads | Implementation | fwd (ms) | fwd+bwd (ms) | fwd XLA temp (MB) | bwd residuals (MB) |
    | --- | --- | ---: | ---: | ---: | ---: |
    | 1 | before: repeat K/V | 0.778 | 3.55 | 32.0 | 48.1 |
    | 1 | after: native GQA | 0.726 | 3.40 | 0 | 16.1 |
    | 1 | `jax.nn` XLA | 1.03 | 3.08 | 256 | — |
    | 1 | torch SDPA | 1.77 | 8.35 | — | — |
    | 8 | before: repeat K/V | 0.775 | 3.49 | 32.0 | 48.1 |
    | 8 | after: native GQA | 0.719 | 3.42 | 0 | 16.1 |
    | 8 | `jax.nn` XLA | 1.20 | 3.09 | 256 | — |
    | 8 | torch SDPA | 1.79 | 8.25 | — | — |

    fp32, T=4096:

    | KV heads | Implementation | fwd (ms) | fwd+bwd (ms) | fwd XLA temp (MB) | bwd residuals (MB) |
    | --- | --- | ---: | ---: | ---: | ---: |
    | 1 | before: repeat K/V | 10.1 | 46.4 | 128 | 193 |
    | 1 | after: native GQA | 9.88 | 45.7 | 0 | 64.5 |
    | 1 | torch SDPA | 24.5 | 116 | — | — |
    | 8 | before: repeat K/V | 10.1 | 46.5 | 128 | 193 |
    | 8 | after: native GQA | 9.96 | 46.0 | 0 | 64.5 |
    | 8 | torch SDPA | 24.4 | 116 | — | — |

Device peak of one call, bf16, T=4096, 32 query heads (fresh process):

| KV heads | Implementation | fwd peak (MB) | fwd+bwd peak (MB) | inputs (MB) |
| --- | --- | ---: | ---: | ---: |
| 8 | before: repeat K/V | 240 | 416 | 112 |
| 8 | after: native GQA | 144 | 416 | 112 |
| 8 | torch SDPA | 145 | 417 | 112 |
| 1 | before: repeat K/V | 227 | 389 | 99.0 |
| 1 | after: native GQA | 131 | 389 | 99.0 |
| 1 | torch SDPA | 131 | 389 | 98.0 |

**Split-KV decoding** (flash-decoding, `attention_splitkv.py`). For few queries against a long KV cache the regular kernel parallelizes only over batch × heads × query tiles and leaves most SMs idle. The split-KV forward splits the keys across programs, packs the query heads of a GQA group into one query tile and combines the partial results with their log-sum-exp in XLA. It is selected automatically for unmasked inputs with Lq ≤ 64, Lkv ≥ 1024 and at least two splits; `split_kv=False` disables it. It has no backward of its own: under `jax.grad` the regular kernel runs.

bf16, 32 query heads, 8 KV heads (GQA, group 4), head_dim 128, median of three rounds:

| B | Lq | Lkv | split-KV (default) | `split_kv=False` | previous GQA path | `jax.nn` XLA | `jax.nn` cuDNN | torch SDPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 4k | 0.039 | 0.136 | 0.237 | 0.039 | 0.064 | 0.036 |
| 1 | 1 | 16k | 0.089 | 0.515 | 0.924 | 0.097 | 0.091 | 0.097 |
| 1 | 1 | 64k | 0.318 | 2.03 | 3.68 | 0.372 | 0.318 | 0.348 |
| 1 | 16 | 4k | 0.047 | 0.127 | 0.233 | 0.067 | 0.160 | 0.083 |
| 1 | 16 | 16k | 0.126 | 0.490 | 0.905 | 0.265 | 0.619 | 0.279 |
| 1 | 16 | 64k | 0.579 | 1.96 | 3.61 | 1.12 | 2.47 | 1.08 |
| 8 | 1 | 4k | 0.165 | 0.522 | 1.53 | 0.497 | 0.161 | 0.180 |
| 8 | 1 | 16k | 0.622 | 2.05 | 6.07 | 1.99 | 0.617 | 0.629 |
| 8 | 1 | 64k | 2.43 | 8.23 | 24.4 | 8.19 | 2.44 | 2.45 |
| 8 | 16 | 4k | 0.282 | 0.521 | 1.53 | 0.817 | 0.612 | 1.24 |
| 8 | 16 | 16k | 1.03 | 2.08 | 6.10 | 3.24 | 2.46 | 4.95 |
| 8 | 16 | 64k | 4.02 | 8.32 | 24.4 | 14.2 | 9.99 | 19.8 |

Same with 32 KV heads (plain multi-head attention; batch 8 at Lkv=64k was skipped because K+V exceed 4.5 GB):

| B | Lq | Lkv | split-KV (default) | `split_kv=False` | previous GQA path | `jax.nn` XLA | `jax.nn` cuDNN | torch SDPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 4k | 0.090 | 0.139 | 0.139 | 0.113 | 0.090 | 0.096 |
| 1 | 1 | 16k | 0.319 | 0.523 | 0.523 | 0.342 | 0.318 | 0.323 |
| 1 | 1 | 64k | 1.23 | 2.07 | 2.06 | 1.39 | 1.24 | 1.23 |
| 1 | 16 | 4k | 0.090 | 0.130 | 0.130 | 0.118 | 0.161 | 0.099 |
| 1 | 16 | 16k | 0.317 | 0.502 | 0.502 | 0.450 | 0.619 | 0.326 |
| 1 | 16 | 64k | 1.23 | 1.99 | 1.99 | 2.04 | 2.47 | 1.24 |
| 8 | 1 | 4k | 0.627 | 0.722 | 0.722 | 0.668 | 0.637 | 1.27 |
| 8 | 1 | 16k | 2.45 | 2.85 | 2.85 | 2.64 | 2.52 | 5.05 |
| 8 | 1 | 64k | — | — | — | — | — | — |
| 8 | 16 | 4k | 0.633 | 0.716 | 0.715 | 2.16 | 0.719 | 1.28 |
| 8 | 16 | 16k | 2.45 | 2.85 | 2.85 | 8.78 | 2.72 | 5.08 |
| 8 | 16 | 64k | — | — | — | — | — | — |

??? note "Lq = 4 and the fp32 decode runs"

    bf16, 8 KV heads, Lq = 4:

    | B | Lq | Lkv | split-KV (default) | `split_kv=False` | previous GQA path | `jax.nn` XLA | `jax.nn` cuDNN | torch SDPA |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | 1 | 4 | 4k | 0.041 | 0.136 | 0.237 | 0.044 | 0.160 | 0.082 |
    | 1 | 4 | 16k | 0.089 | 0.514 | 0.924 | 0.168 | 0.619 | 0.278 |
    | 1 | 4 | 64k | 0.319 | 2.03 | 3.67 | 0.520 | 2.46 | 1.07 |
    | 8 | 4 | 4k | 0.168 | 0.524 | 1.53 | 0.576 | 0.606 | 1.22 |
    | 8 | 4 | 16k | 0.622 | 2.07 | 6.09 | 2.42 | 2.44 | 4.92 |
    | 8 | 4 | 64k | 2.44 | 8.33 | 24.4 | 9.35 | 9.89 | 19.8 |

    fp32, 8 KV heads:

    | B | Lq | Lkv | split-KV (default) | `split_kv=False` | previous GQA path | `jax.nn` XLA | `jax.nn` cuDNN | torch SDPA |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | 1 | 1 | 4k | — | — | — | — | — | — |
    | 1 | 1 | 16k | — | — | — | — | — | — |
    | 1 | 1 | 64k | — | — | — | — | — | — |
    | 1 | 4 | 4k | — | — | — | — | — | — |
    | 1 | 4 | 16k | — | — | — | — | — | — |
    | 1 | 4 | 64k | — | — | — | — | — | — |
    | 1 | 16 | 4k | — | — | — | — | — | — |
    | 1 | 16 | 16k | — | — | — | — | — | — |
    | 1 | 16 | 64k | — | — | — | — | — | — |
    | 8 | 1 | 4k | — | — | — | — | — | — |
    | 8 | 1 | 16k | — | — | — | — | — | — |
    | 8 | 1 | 64k | — | — | — | — | — | — |
    | 8 | 4 | 4k | — | — | — | — | — | — |
    | 8 | 4 | 16k | — | — | — | — | — | — |
    | 8 | 4 | 64k | — | — | — | — | — | — |
    | 8 | 16 | 4k | — | — | — | — | — | — |
    | 8 | 16 | 16k | — | — | — | — | — | — |
    | 8 | 16 | 64k | — | — | — | — | — | — |

With GQA, split-KV matches cuDNN and torch for single-query decoding and is roughly 2–5× faster than both once Lq > 1 (speculative or chunked decoding). For plain multi-head attention (32 KV heads) it is on par with cuDNN and torch. The previous GQA path repeated K/V in memory, which is where most of its time and its 10 GB peak went. Device peak at batch 8, Lkv = 64k (the bf16 KV cache alone is 2 GB):

| Lq | Implementation | peak (MB) | inputs incl. KV cache (MB) |
| --- | --- | ---: | ---: |
| 1 | split-KV (default) | 2,050 | 2,048 |
| 1 | `split_kv=False` | 2,050 | 2,048 |
| 1 | previous GQA path | 10,242 | 2,048 |
| 1 | torch SDPA | 2,049 | 2,048 |
| 16 | split-KV (default) | 2,089 | 2,054 |
| 16 | `split_kv=False` | 2,055 | 2,054 |
| 16 | previous GQA path | 10,247 | 2,054 |
| 16 | torch SDPA | 2,052 | 2,051 |

## CPU path

On CPU, `mha`, `mla` and `flex_attention` previously ran the Pallas kernels in interpret mode. They now run an XLA formulation (`attention_xla.py`): dense for T ≤ 256, a single-tile flash `custom_vjp` up to 512, and 256 × 256 tiles with block-mask tile skipping above that; the backward recomputes from the log-sum-exp. Masks, biases (with gradients), dropout (bit-identical counter dropout), GQA without repeat and forward-mode AD are supported. fp32, batch 1, 8 heads, head_dim 64 ([`kernel_attn-cpu_cpu.json`](kernel_attn-cpu_cpu.json); interpret mode was only run up to T=1024). The host is a shared 24-thread machine; expect 10–30% run-to-run noise.

**Non-causal forward**

| Implementation | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before: `mha` in Pallas interpret mode | 0.668 | 2.99 | 11.6 | 41.3 | — | — |
| after: `mha` (auto XLA strategy) | 0.220 | 1.09 | 2.57 | 15.4 | 62.1 | 244 |
| XLA dense strategy | 0.279 | 1.10 | 3.14 | 26.6 | 111 | — |
| XLA tiled strategy | 0.309 | 0.912 | 4.16 | 15.6 | 59.8 | 241 |
| `jax.nn` XLA | 0.293 | 0.775 | 2.17 | 46.8 | 124 | 406 |
| torch SDPA (CPU) | 0.063 | 0.180 | 0.639 | 2.16 | 9.31 | 35.3 |

**Non-causal forward + backward**

| Implementation | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before: `mha` in Pallas interpret mode | 4.02 | 13.6 | 51.9 | — | — | — |
| after: `mha` (auto XLA strategy) | 0.441 | 2.09 | 13.5 | 48.5 | 207 | 721 |
| XLA dense strategy | 0.747 | 2.07 | 16.8 | 84.6 | 337 | — |
| XLA tiled strategy | 0.835 | 2.21 | 13.5 | 48.9 | 230 | 707 |
| `jax.nn` XLA | 0.689 | 1.88 | 5.71 | 43.5 | 193 | 779 |
| torch SDPA (CPU) | 0.407 | 0.961 | 3.31 | 11.1 | 73.9 | 169 |

**Causal forward**

| Implementation | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before: `mha` in Pallas interpret mode | 0.742 | 2.76 | 9.76 | 35.1 | — | — |
| after: `mha` (auto XLA strategy) | 0.300 | 1.07 | 2.64 | 10.1 | 35.4 | 131 |
| XLA dense strategy | 0.231 | 1.07 | 3.26 | 26.8 | 113 | — |
| XLA tiled strategy | 0.235 | 0.907 | 3.26 | 10.0 | 34.9 | 133 |
| `jax.nn` XLA | 0.300 | 1.16 | 2.71 | 42.6 | 144 | 415 |
| torch SDPA (CPU) | 0.070 | 0.179 | 0.694 | 1.76 | 6.49 | 28.6 |

**Causal forward + backward**

| Implementation | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before: `mha` in Pallas interpret mode | 3.82 | 19.1 | 46.6 | — | — | — |
| after: `mha` (auto XLA strategy) | 0.747 | 2.18 | 15.3 | 32.5 | 107 | 392 |
| XLA dense strategy | 0.503 | 2.10 | 16.2 | 84.7 | 344 | — |
| XLA tiled strategy | 0.710 | 2.28 | 10.5 | 31.4 | 110 | 386 |
| `jax.nn` XLA | 0.707 | 1.94 | 7.26 | 55.1 | 238 | 1,011 |
| torch SDPA (CPU) | 0.425 | 0.999 | 3.33 | 9.40 | 32.5 | 101 |

torch's native CPU SDPA kernel is still roughly 2.5–7× faster than the fastest XLA formulation at T ≥ 1024; closing that gap needs a native CPU kernel, not a different XLA graph. XLA temporaries of the non-causal fwd+bwd (MB):

| Implementation | T=512 | T=1024 | T=2048 | T=4096 |
| --- | ---: | ---: | ---: | ---: |
| before: `mha` in Pallas interpret mode | 12.4 | — | — | — |
| after: `mha` (auto XLA strategy) | 33.0 | 19.1 | 25.1 | 40.1 |
| XLA dense strategy | 41.0 | 162 | 644 | — |
| XLA tiled strategy | 16.0 | 19.1 | 25.1 | 40.1 |
| `jax.nn` XLA | 25.0 | 98.1 | 388 | 1,544 |

Process-level peak (resident set size increase in a fresh process) ([`kernel_attn-cpu_cpu_memory.json`](kernel_attn-cpu_cpu_memory.json)):

| Implementation | T | XLA temp (MB) | process RSS increase (MB) |
| --- | ---: | ---: | ---: |
| probjax_mha | 4096 | 40.1 | 43.5 |
| xla_dense | 4096 | 2,568 | 2,590 |
| jax_xla | 4096 | 1,544 | 1,668 |
| torch_sdpa | 4096 | — | 197 |
| probjax_mha | 16384 | 161 | 189 |
| torch_sdpa | 16384 | — | 464 |

Memory needed for one fwd+bwd (XLA temp + arguments + outputs, MB) by length; ✗ marks sizes that exceed the host's RAM. The tiled path scales linearly and fits T = 131072 in under 3 GB; the dense formulations stop fitting at T = 32768.

| Implementation | T=4096 | T=8192 | T=16384 | T=32768 | T=65536 | T=131072 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xla_tiled | 88.1 | 176 | 353 | 705 | 1,410 | 2,820 |
| xla_dense | 2,624 | 10,368 | 41,217 | 164,353 ✗ | 656,386 ✗ | 2,623,492 ✗ |
| jax_xla | 1,600 | 6,273 | 24,833 | 98,818 ✗ | 394,244 ✗ | 1,574,920 ✗ |

CPU decoding, 32 query heads, 8 KV heads, head_dim 128 ([`kernel_attn_ext_cpu.json`](kernel_attn_ext_cpu.json)). `flex_attention` no longer pads a short query block to 16 rows on CPU, which helps Lq = 1 and leaves Lq = 16 unchanged. torch remains much faster here as well.

| Lq | Lkv | `flex_attention` before → after | previous GQA path (repeat K/V) | `jax.nn` XLA | torch SDPA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4k | 15.7 → 9.73 | 55.5 | 9.28 | 0.700 |
| 1 | 16k | 61.5 → 42.5 | 222 | 40.4 | 13.4 |
| 1 | 64k | 377 → 168 | 1,515 | 165 | 69.0 |
| 16 | 4k | 16.1 → 15.9 | 58.6 | 15.3 | 2.20 |
| 16 | 16k | 69.1 → 64.7 | 241 | 62.8 | 16.4 |
| 16 | 64k | 381 → 379 | 1,121 | 250 | 69.6 |

## Reproduce

```bash
# Cross-library grid and the tile sweep (CUDA GPU; torch optional).
python -m docs.scripts.benchmark_gpu_attention --output docs/benchmarks/gpu_attention.json
python -m docs.scripts.benchmark_gpu_attention_tiles --output docs/benchmarks/gpu_attention_tiles.json

# Dispatch sweep, device peak and residuals.
python -m docs.scripts.benchmark_kernel_attn_gpu --device gpu --suite dispatch \
  --output docs/benchmarks/kernel_attn_gpu_dispatch.json
python -m docs.scripts.benchmark_kernel_attn_gpu --device gpu --suite peak \
  --output docs/benchmarks/kernel_attn_gpu_memory.json

# GQA and split-KV decoding.
python -m docs.scripts.benchmark_kernel_attn_ext --device gpu --suite gqa \
  --output docs/benchmarks/kernel_attn_ext_gpu.json
python -m docs.scripts.benchmark_kernel_attn_ext --device gpu --suite decode \
  --output docs/benchmarks/kernel_attn_ext_gpu.json --append

# CPU path.
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_attn-cpu \
  --device cpu --output docs/benchmarks/kernel_attn-cpu_cpu.json
```
