# SMC presets, adaptation, and diagnostics

The library reuses BlackJAX 1.7.1 for stationary SMC updates, resampling, ESS,
root finding, waste-free updates, MCMC tuning, and Metropolis acceptance. The
additional probjax code supplies runner/result conventions and temporal composition.

## Batching and waste-free updates

All stationary constructors accept `batch_size`. Zero maps the entire population
at once; a positive value uses BlackJAX's sequential particle batches to reduce
peak memory. Persistent initialization also forwards this option.

`waste_free_smc` retains intermediate MCMC states. Its `p` argument specifies the
number of retained states per chain, and must divide the population size. It
resamples `N/p` seeds and returns `N` particles. These samples are correlated;
population size alone does not measure effective independent information.

```python
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from probjax.inference import SMC, hmc, waste_free_smc

logprior = lambda x: norm.logpdf(x).sum()
loglikelihood = lambda x: norm.logpdf(x, 1., 0.7).sum()
key = jax.random.key(0)
initial_key, run_key = jax.random.split(key)
particles = jax.random.normal(initial_key, (128, 2))
params = {'step_size': jnp.array([0.15]), 'inverse_mass_matrix': jnp.ones((1, 2))}
kernel = waste_free_smc(
    logprior, loglikelihood, num_particles=128, p=4,
    adaptive=True, mcmc_kernel=hmc, num_integration_steps=3, batch_size=32,
)
result = SMC(kernel).run_adaptive(run_key, kernel.init(particles), params)
assert result.completed
assert result.state.particles.shape == particles.shape
```

For custom compositions, use `waste_free_strategy(N, p)` as `update_strategy`, and
set `num_mcmc_steps=None`. The preset does this automatically. This strategy also
works with the existing `PartialPosteriorsPath`; data tempering is already supported.
The adaptive geometric kernel explicitly rejects non-geometric paths.

## Adaptive runs and constant-memory summaries

`SMC.run_adaptive` delegates temperature selection to the existing adaptive kernel
and advances until temperature reaches one. `max_steps` bounds execution;
`completed=False` reports a cap, stalled/nonfinite progress, or exhausted persistent
storage. Persistent samplers need enough `max_iterations` storage at initialization.
Adaptive runs retain no population history. An optional existing `Adaptor` updates
move parameters between stages.

`SMCResult` keeps the original `state`, `params`, and optional `info` trace and adds:

- `log_evidence`: accumulated normalizing-constant estimate, even without a trace.
- `final_info`: the final kernel diagnostics (zero prototype for an adaptive run
  taking no steps; None for an empty fixed schedule).
- `num_steps`, `completed`, and a continuation `key`.

For fixed schedules, `completed` means the supplied schedule was executed; it does
not assert that its last temperature equals one. To resume ordinary SMC, pass the
previous result's `log_evidence` as `initial_log_evidence`; persistent states already
store it. Continuation keys use sequential splitting so chunked runs match batch
runs. This changes the random stream relative to older versions that split all
schedule keys at once. `SMCResult` has additional tuple fields, so prefer named
attributes over positional tuple unpacking. Custom kernels without evidence
diagnostics report NaN evidence instead of a misleading zero.

Persistent sampler constructors bind their own likelihood. Building a second model
cannot overwrite a first model's likelihood; direct low-level initialization must
supply it explicitly.

## BlackJAX tuning adapters

`tuned_smc` wraps BlackJAX `inner_kernel_tuning` using our existing MCMC adapter.
It accepts initial **unbatched shared** MCMC parameter values and a callback
`parameter_update_fn(key, new_smc_state, info)`. The callback returns parameters
with BlackJAX's leading shared (1) or per-particle (N) axis. Tuning happens after
each SMC stage and is stored in `state.parameter_override`.

```python
from probjax.inference import tuned_smc

def update_geometry(key, state, info):
    covariance = jnp.cov(state.particles.T) + 1e-3*jnp.eye(2)
    return {'step_size': jnp.array([0.15]), 'inverse_mass_matrix': covariance[None]}

tuned = tuned_smc(
    logprior, loglikelihood, mcmc_kernel=hmc,
    mcmc_parameters={'step_size': 0.15, 'inverse_mass_matrix': jnp.eye(2)},
    parameter_update_fn=update_geometry, num_mcmc_steps=2,
    num_integration_steps=3, batch_size=32,
)
tuned_result = SMC(tuned).run_adaptive(run_key, tuned.init(particles), {})
assert tuned_result.completed
```

`pretuned_smc` reuses BlackJAX `build_pretune` to take pilot moves, measure movement,
and perturb/reweight a distribution of move parameters before production moves.
Supply `num_particles`, `sigma_parameters`, and initial parameters with explicit
leading batch axes. The default ESJD measure requires a full shared inverse mass
matrix shaped `(1,D,D)`. `positive_parameters` and `natural_parameters` constrain
the perturbed parameters. Our adapter gives pilot and production moves independent
keys and forwards production `batch_size`. BlackJAX's pilot itself currently maps
the population together. It incurs additional computation and should be benchmarked
against the simpler existing geometry/acceptance adaptors.

## Diagnostics beyond weight ESS

- `population_diagnostics`: weight ESS, maximum weight, the exact number of
  distinct particles, and distinct ancestry labels. The diversity calculations
  use sorting.
- `summarize_replicates`: mean and Monte Carlo standard error across independent
  run-level estimates. Do not feed it correlated particles from one run.

Inference over physical time (filtering, SMC²) is covered in the
[temporal guide](temporal-smc.md).
