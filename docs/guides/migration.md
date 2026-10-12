# Migrating to 0.2

This release collects the numerical, inference, neural-network and kernel
refactors since v0.1.1. It includes breaking API changes. The website follows
`main`; use documentation from the matching Git tag when reproducing an older
experiment.

## Environment

Use Python 3.11–3.13 and JAX `>=0.9.0,<0.12`. The package requires Flax
`>=0.12.6` and pins BlackJAX to `1.6.2`. Install accelerator dependencies through
the `cuda13` or `metal` extras and check `jax.devices()`. Accelerator kernels
have additional hardware constraints; CPU tests do not certify GPU performance.

## Required changes

| Previous usage | Current usage |
| --- | --- |
| `AutoregressiveModel` | `Autoregressive`; the old name has no compatibility alias |
| Sampling from `EDM(net)`, `VP(net)`, `VE(net)`, or `LinearFlow(net)` | Supply `event_spec` at construction or to `as_dist(event_spec=...)` |
| `LinearMeanFlow(net)` | Supply `event_spec` at construction |
| Categorical diffusion without event metadata | Supply token-position shape, excluding the class axis |
| Imports from `probjax.nn.pallas_kernels.old_pallas` | Use `probjax.nn.pallas_kernels`; the legacy implementation was removed |
| Inspecting named sites with raw `jax.make_jaxpr(model)` | Trace inside `probjax.enable_rv_tracing()` |
| Drift/diffusion keyword arguments passed to solvers | Pass positional arguments, or bind configuration in the callable |

An event shape excludes all sample and batch axes. `event_spec=3` means a
three-feature vector; `(16, 3)` means sixteen tokens with three features.
`as_dist(event_spec=...)` overrides one distribution view;
`model.set_event_spec(...)` changes future defaults. Existing views keep their
bound shapes, and changing metadata does not resize a network's weights.
See [event specifications](../reference/nn.md#diffusion-and-flow-matching).

## Training and checkpoints

Existing module `fit` calls still return losses; functional `fit` returns
`(params, losses)`. Request `return_result=True` for `FitState`, `FitInfo`,
metrics and resume state. EMA is opt-in with `ema_decay`; `use_ema=True` installs
averaged parameters after fitting while the result retains the raw parameters.
Resume with `initial_state=result.state` and the same optimizer/kernel setup.
The snapshot does not include a data-loader position or optimizer configuration.

General callbacks run between compiled chunks. Under JIT they require explicit
`callback_mode="io"`; ordered IO effects cannot be differentiated or vmapped.
Use `result.valid` or `valid_steps` to ignore padded histories after stopping
inside JIT. See [training and EMA](../reference/nn.md#training-and-ema).

## Changes that affect numerical results

- Mean-flow training defaults to Improved MeanFlow (`imf=True`) and uses corrected
  time-pair sampling. Select `loss_kwargs={"imf": False}` for the older loss;
  this does not restore the old time-pair distribution.
- VP/cosine diffusion preconditioning and SDE coefficients were corrected.
  Existing checkpoints may need retraining. `VSolverConfig(mode="sde")` now
  uses stochastic reverse dynamics; explicit DDIM remains deterministic.
- Transformer autoregressive full-forward evaluation is now causal, matching
  cached decoding. Earlier likelihoods from the noncausal path were invalid.
- Upper-tail beta/gamma quantiles avoid forming `1 - q`. Tail values can change
  substantially where that subtraction previously rounded to one.
- Rectangular `LinearOperator` shape metadata and multiplication order now match
  dense linear algebra, including under JIT.

ODE/SDE gradient behavior is described in the [solver guide](numerical-solvers.md).
In particular, adaptive ODEs still use a custom reverse-mode adjoint, and there
is currently no selectable checkpointed adaptive ODE gradient mode. The
backward path is unreliable for adaptive gradients; use a converged fixed-grid
solve when you need differentiable integration.

## Sampling traces and kernels

Prefer `condition`, `do`, and `log_joint_fn` for probabilistic-program
transformations. Their legacy aliases `observe`, `intervene`,
`log_potential_fn`, and `log_prob_fn` still work but emit `DeprecationWarning`.
Use `trace(..., kind_labels="probabilistic")` instead of the deprecated
`kind_labels="legacy"` option.

Ordinary distribution sampling no longer emits random-variable primitives.
ProbJax PPL transformations enable tracing automatically. Opt in explicitly only
when inspecting sampling jaxprs or writing a custom interpreter; the flag is
read at tracing time. `PROBJAX_RV_TRACING=1` restores global tracing behavior.

The Pallas migration preserves supported features, with JAX fallbacks where
needed. Multi-tile GPU Mamba uses an associative scan; its GPU runtime has not
been benchmarked. See [kernel coverage](../reference/nn.md#pallas-kernels) before
choosing a backend for an existing experiment.

## Special function accuracy and domain notes

`betainccinv(a, b, q)` and `gammainccinv(a, q)` invert the complementary
regularized incomplete beta and gamma functions. They accept the upper-tail
probability directly, avoiding the loss of tiny probabilities when forming
`1 - q`. Beta and gamma distribution `isf` methods use them directly.

```python
from probjax.utils import gammainccinv, log1mexp, logdiffexp

x = gammainccinv(1.0, 1e-30)  # Approximately 69.07755, including in float32.
log_survival = log1mexp(-1e-8)
log_difference = logdiffexp(1000.0, 999.0)
```

Inverse functions require finite positive shapes and clip probabilities to
[0, 1]. Invalid shapes return NaN. At upper-tail probabilities zero and one,
beta returns one and zero; gamma returns infinity and zero.

Beta inversion uses cached safeguarded Halley solvers, reflecting according
to the root's location. Its total iteration budget is
`2 + max_halley_steps + max_bisection_steps`. Reverse differentiation uses the
inverse density for probability gradients and central differences for shape
gradients; forward AD and higher shape derivatives are unsupported.
Gamma uses log-space initialization and an implicit JVP, avoiding differentiation
through solver iterations. Higher shape derivatives depend on JAX's
`igamma_grad_a` support.

FP32 accuracy remains limited by the underlying CDF for some extremely unequal
beta shapes: the seeded stress test includes a case near `a=0.051, b=784` with
about 2% relative quantile error. Increasing the iteration budget does not fix
that case. Use FP64 for demanding tail calculations. The originally identified
FP32 regression near `a=0.070, b=626` is covered separately at 0.5% tolerance.

`log1mexp(x)` computes `log(1-exp(x))` for `x <= 0`; `logdiffexp(a, b)` computes
`log(exp(a)-exp(b))` for `a >= b`. They avoid cancellation and intermediate
overflow, support broadcasting, and differentiate on the finite interior.
Zero differences return negative infinity; invalid domains return NaN.

## Continuous diffusion schedule audit

EDM, VE, VP and cosine are checked against their forward marginal/SDE identities,
inverse effective-noise maps, all three training targets and Gaussian-oracle
reverse ODE/SDE sampling. VP now includes the time-domain rescaling in its SDE
coefficients. VP/cosine low-noise calculations avoid float32 cancellation.
Generic schedules use `g² = d(std²)/dt - 2 (scale′/scale) std²`.

EDM preconditioning now normalizes by the signal scale for VP/cosine, including
the denoiser skip connection. Existing VP/cosine checkpoints used different
coefficients and may need retraining. Cosine's singular endpoint is bounded;
use the preset's finite effective-noise bounds for sampling. Numerical checks
with an exact Gaussian denoiser validate solver behavior, not learned sample
quality on arbitrary datasets.

`VSolverConfig` now uses the reverse SDE for `mode="sde"`; previously that path
had zero diffusion and was deterministic. Explicit DDIM remains deterministic.
Categorical linear, cosine, sigmoid and log-SNR schedules are also checked for
monotone retention and normalized, nonnegative transition probabilities.
The categorical linear/sigmoid factory defaults retain about 95% of the initial
signal at terminal time; choose stronger rates when a nearly independent base
prior is required. Their probability checks alone do not establish mixing.

Further sampling fixes: an overridden `t_max` now also selects the base-noise
scale. `sample_with_state` uses the supplied state for both initialization and
integration, and `model_state()` handles Python scalar leaves. Call
`dist.compile()` before wrapping sampling in an outer `jax.jit`; constructing
an export lazily during tracing is currently unsupported.

Exponential AB2 now handles nonuniform step sizes and uses the correct
phi-function weights and midpoint linear coefficient. Gaussian and nonlinear
mixture benchmarks, transformer scaling measurements, and remaining limitations
are recorded in `examples/nn/benchmark_results/diffusion_audit.md`.
