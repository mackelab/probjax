# SMC presets, adaptation, and diagnostics

The library reuses BlackJAX 1.6.2 for stationary SMC updates, resampling, ESS,
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

## Temporal proposal adaptation and existing MCMC kernels

`temporal_smc(..., adaptive_proposal=True)` supplies a Gaussian random walk using
weighted population covariance and an acceptance-tuned scale. Geometry is frozen
within each sweep. `adaptive_num_steps=True` adjusts the next observation's move
budget toward `target_accepted_moves`, capped by `max_rejuvenation_steps`.
These are practical population adaptation heuristics, not convergence guarantees.
Do not also supply a custom `proposal_fn` or an MCMC kernel with this option.

With an exact likelihood backend, `mcmc_kernel=hmc` (or another compatible probjax
kernel), `mcmc_parameters`, and `mcmc_kernel_kwargs` reuse our existing MCMC adapter.
A masked static scan makes full-prefix likelihood replay differentiable; its
suffix performs no model updates. The returned filter state is recomputed for the
accepted parameter. Stochastic particle likelihoods cannot use this ordinary MCMC
route; they use the dedicated pseudo-marginal proposal path instead.

## Tempering a difficult observation

Set `tempering_ess` in `(0,1)` to choose observation bridge increments using weighted
conditional ESS and BlackJAX's dichotomy solver. The latent filter propagates once;
bridging subsequently reweights/resamples the parameter/filter population and can
rejuvenate at intermediate stages. Prefix replay is still needed for proposals.
`info.num_tempering_steps` and `info.tempering_param` expose progress.

For an exact backend the bridge log target is
`logprior(theta) + logL_previous(theta) + beta*log_increment(theta)`.
For particle likelihoods the same expression defines an **extended-space** target
that retains the auxiliary filter estimate. Candidate replay supplies both the
previous-prefix and current-prefix estimates for the Metropolis ratio. Intermediate
marginals are not claimed to equal the exact fractional marginalized likelihood;
at beta=1 the target is the ordinary SMC² endpoint. Re-estimating just the current
noisy likelihood at every bridge would not implement this construction.

If `max_tempering_steps` is exhausted, the entire observation step is rolled back
and `info.valid=False`. Increase the cap and retry from the unchanged state. Hard
support mismatches/all-impossible observations may still prevent progress.

## Adapting inner particle counts

`likelihood_diagnostics` runs independent likelihood replicates at **one fixed
parameter** and reports log-likelihood variance and the log of the mean likelihood.
It never substitutes the mean log likelihood for a log mean likelihood.
`recommend_particle_count` uses a `variance ~ 1/N` heuristic to select the next
nondecreasing bucket; its `at_capacity` flag reports unmet requested capacity.

`adapt_particle_count` is a host controller: it samples parameter values using the
outer weights, estimates their likelihood noise, selects a bucket, and replays fresh
filters if needed. It caches compiled diagnostic/exchange functions by backend
factory and capacity. The full data prefix is required. After a change, build/use
the temporal kernel associated with `backend_factory(result.particle_count)`.
Its state can be passed directly to that kernel; retain its continuation key.

A particle-count change applies the importance correction `L_new/L_old` to outer
weights and accumulates its evidence correction. Simply swapping filter states
would change the extended target. `exchange_filter_population` exposes this pure,
JIT-compatible operation for a fixed old/new capacity pair. Its output has the
new shape even on failure, so check `info.valid` before committing it. The host
controller raises on failure. This is an importance exchange, not an MH move.

The controller intentionally runs outside JIT because a loop carry cannot change
array shape. Propagation, diagnostics, replay, and exchanges remain compiled.
For a fully compiled application, manage fixed-capacity phases explicitly using
the lower-level functions. Variance estimates from a few replicates can be noisy;
particle growth does not guarantee that parameter proposals mix well.

## Diagnostics beyond weight ESS

- `population_diagnostics`: weight ESS, maximum weight, exact number of distinct
  particles, and distinct ancestry labels. The diversity calculations use sorting.
- `state.lineages` in temporal SMC composes ancestry back to initial parameter
  particles; it can expose collapse hidden by uniform post-resampling weights.
- `likelihood_diagnostics`: independent estimator noise at fixed parameters.
- `summarize_replicates`: mean and Monte Carlo standard error across independent
  run-level estimates. Do not feed it correlated particles from one run.

The [temporal guide](temporal-smc.md) also covers persistent streaming windows and
joint parameter/path sampling. The [notebook](https://github.com/mackelab/probjax/blob/main/examples/inference/temporal_smc.ipynb)
includes exact posterior references and examples of the adaptive controls.
