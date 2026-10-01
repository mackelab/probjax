# Migrating to 0.2

This release collects the numerical, inference, neural-network and kernel
refactors since v0.1.1. It includes breaking API changes. The website follows
`main`; use documentation from the matching Git tag when reproducing an older
experiment.

## Environment

Use Python 3.11–3.13 and JAX `>=0.9.0,<0.10`. The package requires Flax
`>=0.12.6` and pins BlackJAX to `1.6.2`. Install accelerator dependencies through
the `cuda13` or `metal` extras and check `jax.devices()`. Accelerator kernels
have additional hardware constraints; CPU tests do not certify GPU performance.

## Required changes

| Previous usage | Current usage |
| --- | --- |
| `AutoregressiveModel` | `Autoregressive`; the old name has no compatibility alias |
| `EDM(net)`, `VP(net)`, `VE(net)` | Supply `event_spec=features` or the shape of one event |
| `LinearFlow(net)`, `LinearMeanFlow(net)` | Supply `event_spec` here too |
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
In particular, adaptive ODEs still use a reverse-mode adjoint; this release does
not introduce a selectable checkpointed adaptive ODE gradient mode. The adaptive
backward path currently has an [open release blocker](../releasing.md#open-release-blocker).

## Sampling traces and kernels

Ordinary distribution sampling no longer emits random-variable primitives.
ProbJax PPL transformations enable tracing automatically. Opt in explicitly only
when inspecting sampling jaxprs or writing a custom interpreter; the flag is
read at tracing time. `PROBJAX_RV_TRACING=1` restores global tracing behavior.

The Pallas migration preserves supported features, with JAX fallbacks where
needed. Multi-tile GPU Mamba uses an associative scan; its GPU runtime has not
been benchmarked. See [kernel coverage](../reference/nn.md#pallas-kernels) before
choosing a backend for an existing experiment.
