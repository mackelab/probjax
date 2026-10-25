# Sequential Monte Carlo

SMC moves a population of particles through a sequence of distributions that
ends at the target. Each step reweights the particles towards the next
distribution, resamples them and moves them. In return for the larger state
you get a sample that copes with several modes, and an estimate of the
normalising constant (the evidence) for free.

Both SMC kernels follow the [kernel contract](inference.md): the state is the
whole population, `run` drives them, and an adaptor can tune them as they go.

## Tempering from the prior to the posterior

`smc(logprior, loglikelihood, move=...)` targets
`prior(x) * exp(t * loglikelihood(x))` and raises the temperature `t` from 0 to
1. After every reweighting it resamples and runs `num_mcmc_steps` steps of the
`move` kernel, which can be any MCMC kernel. Its params are the SMC kernel's
params, shared by all particles.

On a fixed schedule each step takes the next temperature as its argument:

```python
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from probjax.inference import hmc, run, smc

logprior = lambda x: norm.logpdf(x).sum()
loglikelihood = lambda x: norm.logpdf(x, 1.0, 0.7).sum()

kernel = smc(logprior, loglikelihood, move=hmc(num_integration_steps=4))
key, init_key = jax.random.split(jax.random.key(0))
state = kernel.init(init_key, jax.random.normal(init_key, (512, 2)))
params = kernel.init_params(state, step_size=0.3)

schedule = jnp.linspace(0.0, 1.0, 11)[1:]
result = run(key, kernel, state, params, 10, args=(schedule,), collect=("ess",))
print(result.state.log_evidence, result.trace["ess"])
```

The particles end equally weighted, and `state.log_evidence` estimates the log
evidence.

## Adaptive temperatures and adaptive moves

With `target_ess` each step picks the largest temperature that keeps that
fraction of effective particles, so the schedule follows the difficulty of the
problem. Once the temperature reaches 1 further steps change nothing, so
`num_steps` is a cap rather than a length.

The kernel's `default_adaptation` refits the move to the population after
every step. A population sampled from the current target is a good estimate of
the next one's shape, so it sets HMC's inverse mass matrix to the particle
variances (`particle_adaptor`). It also scales the step size by
`exp(acceptance_rate - 0.65)` (`acceptance_rate_adaptor`). Both are ordinary
[adaptors](inference.md); compose your own with `compose_adaptors`.

```python
adaptive = smc(
    logprior, loglikelihood, move=hmc(num_integration_steps=4), target_ess=0.7
)
result = run(
    key,
    adaptive,
    state,
    params,
    50,
    adaptor=adaptive.default_adaptation(),
    collect=("temperature",),
)
assert result.state.temperature == 1.0
print(result.trace["temperature"][:6], result.params.inverse_mass_matrix)
```

## Waste-free moves, data tempering and batching

`waste_free=p` keeps every state of the move chains instead of only the last
(Dau and Chopin, 2022). Each step resamples `N / p` particles and runs `p - 1`
moves from each, so all `N` particles come from the same number of target
evaluations as `N / p` long chains. The kept states are correlated, so the
population size overstates the information.

Data tempering adds observations one at a time instead of raising a common
temperature. Let the log-likelihood return one term per observation and use a
vector temperature: `exp(sum(t * loglikelihood(x)))`. A schedule of masks then
switches observations on in turn.

`batch_size` moves the particles in batches of that size, which bounds the
memory of large populations without changing the result.

```python
data = jnp.array([1.0, -0.5]) + 0.7 * jax.random.normal(key, (8, 2))
per_observation = lambda x: norm.logpdf(data, x, 0.7).sum(axis=1)

streaming = smc(
    logprior,
    per_observation,
    move=hmc(num_integration_steps=4),
    waste_free=4,
    batch_size=128,
)
state = streaming.init(init_key, jax.random.normal(init_key, (512, 2)), jnp.zeros(8))
masks = jnp.tril(jnp.ones((8, 8)))  # step i sees the first i + 1 observations
result = run(key, streaming, state, params, 8, args=(masks,))
print(result.state.particles.mean(0))
```

## Diffusion models as proposals

`diffusion_smc(guide, times)` starts particles from the noise of a diffusion
model and follows its reverse SDE down the time grid. A twist looks ahead to
the target and weights the particles; at the last time it is the target
itself, so the sampler is consistent whatever the twist, and a better twist
only needs fewer particles. A `Guide` supplies the reverse SDE and the twist.

- `neural.diffusion_guide(model, V)` steers a pretrained diffusion model
  towards `prior(x) * exp(V(x))`, with `V` evaluated at the model's denoised
  estimate as the twist. This conditions a generative prior on a reward or an
  observation without retraining it. The guide's params are the model's
  weights.
- `score_guide(log_target, schedule, shape)` needs no model. It estimates the
  score and the noised marginal of an unnormalised target by importance
  sampling, at a cost of `num_inner` target evaluations per particle and step.

```python
from probjax.inference import diffusion_smc, score_guide
from probjax.nn.generative.diffusion.config import VPNoiseSchedule

bimodal = lambda x: jnp.logaddexp(
    norm.logpdf(x, -2.0, 0.5).sum(), norm.logpdf(x, 2.0, 0.5).sum()
)
noise = VPNoiseSchedule(t_min=1e-3, t_max=1.0)
guide = score_guide(bimodal, noise, (1,), num_inner=32)
sampler = diffusion_smc(guide, jnp.linspace(1.0, 1e-3, 21))
state = sampler.init(init_key, 256)
result = run(key, sampler, state, None, 20, collect=("ess", "resampled"))
weights = jnp.exp(result.state.log_weights)
print(jnp.sum(weights * (result.state.particles[:, 0] > 0)))  # about one half
```

The [SMC tutorial](../tutorials/smc.md) runs both kernels on harder targets.

## Diagnostics beyond weight ESS

- `population_diagnostics`: weight ESS, maximum weight, the exact number of
  distinct particles, and distinct ancestry labels. The diversity calculations
  use sorting.
- `summarize_replicates`: mean and Monte Carlo standard error across independent
  run-level estimates. Do not feed it correlated particles from one run.

Inference over physical time (filtering, SMC²) is covered in the
[temporal guide](temporal-smc.md).
