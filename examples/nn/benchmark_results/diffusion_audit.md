# Diffusion and transformer audit

Measured on AMD Ryzen 9 5900X 12-Core Processor (CPU) with JAX 0.10.0. Benchmarks synchronize device work, separate export/first-call cost from five warm calls, and use a fixed seed. Runtime numbers are indicative of this machine; these are not GPU measurements. Gaussian and mixture samplers draw 4,096 scalar events. The transformer uses batch 8, width 64, two layers and four heads.

## Confirmed bugs fixed

- Diffusion sampling now initializes at the distribution view’s overridden `t_max`.
- Explicit-state sampling uses that state for the base distribution as well as the solver; Python scalar state leaves can be snapshotted.
- DDIM/v solver coefficients use the physical SDE instead of differentiating clipped endpoints.
- Exponential AB2 uses phi2-weighted extrapolation, the current/previous step-size ratio, and midpoint scalar linear coefficients. Analytic forward/reverse, uniform/nonuniform grids verify second-order convergence.
- TimeMLP and DiffusionTransformer accept broadcast batch time vectors; bfloat16 positional encoding casts explicitly.

## Gaussian accuracy and runtime (64 time points)

Exact zero-network Gaussian denoiser. Percent errors below include Monte Carlo error: the shared ODE initial sample itself is 0.282% above the true standard deviation. VE ODE previously had +11.703% error at 64 points; after the AB2 correction it has +0.282%. The coefficient correction alone did not remove that numerical error.

| Preset | Mode | Std error | Warm ms | Export + first call s |
|---|---|---:|---:|---:|
| EDM | ode | +0.633% | 0.773 | 0.361 |
| EDM | sde | +4.084% | 11.377 | 0.507 |
| VE | ode | +0.282% | 0.411 | 0.412 |
| VE | sde | -0.656% | 10.934 | 0.532 |
| VP | ode | +0.282% | 0.628 | 0.353 |
| VP | sde | +1.690% | 11.371 | 0.551 |
| CosineDM | ode | +0.282% | 0.598 | 0.296 |
| CosineDM | sde | +2.121% | 11.580 | 0.566 |

## Nonlinear mixture (64 time points)

Exact posterior denoiser for an equal mixture of N(-3, 0.5²) and N(3, 0.5²). The terminal Gaussian is a moment-matched approximation; its error is not isolated from solver error. Target mean absolute value is approximately 3, and target probability of |x|<1 is approximately 0.000032. This is a nonlinear score test, not a learned-model or high-dimensional quality benchmark.

| Preset | Mode | Std error | Mean absolute value | P(abs(x)<1) | Warm ms |
|---|---|---:|---:|---:|---:|
| EDM | ode | +0.174% | 3.0049 | 0.000244 | 0.980 |
| EDM | sde | +0.185% | 3.0023 | 0.000488 | 11.917 |
| VE | ode | -0.298% | 2.9898 | 0.000244 | 0.635 |
| VE | sde | -0.934% | 2.9700 | 0.000000 | 10.982 |
| VP | ode | -0.558% | 2.9771 | 0.000732 | 0.737 |
| VP | sde | -0.126% | 2.9944 | 0.000000 | 11.033 |
| CosineDM | ode | +0.677% | 3.0145 | 0.000732 | 0.690 |
| CosineDM | sde | +0.472% | 3.0100 | 0.000000 | 10.664 |

## Transformer scaling

| Dtype | Tokens | Forward ms | Loss + gradient ms |
|---|---:|---:|---:|
| float32 | 32 | 2.915 | 6.257 |
| float32 | 128 | 6.368 | 14.935 |
| float32 | 512 | 51.301 | 153.396 |
| bfloat16 | 32 | 3.043 | 6.696 |
| bfloat16 | 128 | 6.533 | 16.830 |
| bfloat16 | 512 | 47.655 | 155.107 |

## Remaining weaknesses

- Coarse SDE grids can be unstable: VE at 16 points produces Gaussian standard deviation about 5.97 instead of 1. More points are essential; finite outputs alone are not a quality check.
- Oracle SDE runs are much slower than ODE because of per-step random draws; expensive learned denoisers may change that ratio.
- Full self-attention grows quadratically in token count. bfloat16 does not guarantee a CPU speedup. These checks establish finite forward/gradient outputs, not low-precision training equivalence.
- The categorical linear/sigmoid defaults retain about 95% signal at the endpoint; the previous audit documented this without changing defaults. Categorical mixing quality remains unaudited here.
- The exponential AB3 method was regression-tested for shared-state compatibility, but its convergence/order on nonuniform grids was not established by this audit.
- Precompile a distribution with `dist.compile()` before placing its calls inside an outer `jax.jit`; lazy export during tracing currently fails.
- No GPU benchmark or end-to-end trained diffusion comparison was run in this audit.

## Reproduce

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/nn/benchmark_diffusion_audit.py --output /tmp/gaussian.json
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/nn/benchmark_diffusion_audit.py --profile mixture --output /tmp/mixture.json
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/nn/benchmark_transformer_audit.py --output /tmp/transformer.json
```

Raw measurements are stored alongside this report as JSON, including the Gaussian baseline taken before this audit’s fixes.

Validation: 97 focused audit/diffusion/transformer/sampling tests and 152 selected split-drift ODE/SDE regression tests passed on CPU. The positional-encoding benchmark was rerun after the bfloat16 cast fix.
