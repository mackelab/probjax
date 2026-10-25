# Temporal SMC and trajectory inference

Temporal inference moves forward through physical time, one observation at a
time. With fixed parameters `theta` a filter tracks the latent state
`p(x_t | y_1:t)`; with unknown parameters temporal SMC also targets
`p(theta | y_1:t)`. The static `smc` kernel stays the tool for annealing a
stationary target.

## Writing a model

A `StateSpaceModel` is three functions written for **one** latent state, which may
be any pytree. The library vectorizes them over particles.

```python
import jax
import jax.numpy as jnp
from probjax.stats import norm
from probjax.inference import StateSpaceModel

observation_sd = 0.45

def init(key, theta):  # a draw from the initial law at t0
    return norm(jnp.zeros(1), 1.0).sample(key)

def step(key, theta, x, t_previous, t_next):  # a draw over one interval
    sd = jnp.sqrt(jnp.exp(theta) * (t_next - t_previous))
    return x + sd * jax.random.normal(key, x.shape)

def log_likelihood(theta, x, y, t):  # normalized observation density
    return norm(x, observation_sd).logpdf(y).sum()

model = StateSpaceModel(init, step, log_likelihood)
```

Times are physical, so irregular grids need no special handling. The initial law
is at `t0` and observations come after it; fold any observation at `t0` into
`init` yourself.

A linear-Gaussian model is filtered exactly by a Kalman filter instead. Its
callbacks return `(mean, cov)`, `(Phi, Q)` and `(C, R)`:

```python
from probjax.inference import LinearGaussianModel

linear_model = LinearGaussianModel(
    initial=lambda theta: (jnp.zeros(1), jnp.eye(1)),
    transition=lambda theta, t0, t1: (jnp.eye(1), jnp.exp(theta) * (t1 - t0) * jnp.eye(1)),
    observation=lambda theta, t: (jnp.eye(1), observation_sd**2 * jnp.eye(1)),
)
```

## Filtering at known parameters

```python
from probjax.inference import run_temporal_filter

ts = jnp.array([0.2, 0.7, 1.4, 2.0])
ys = jnp.array([[0.4], [-0.1], [0.8], [0.5]])
key = jax.random.key(0)
theta = jnp.log(0.3)

result = run_temporal_filter(key, model, theta, ts, ys, num_particles=512)
exact = run_temporal_filter(key, linear_model, theta, ts, ys)
print(result.log_likelihood, exact.log_likelihood)
```

`observed=` takes a boolean mask for missing observations (their values may be
NaN); masked steps only predict. `history` controls what is kept:

| `history` | Retained output |
| --- | --- |
| `'full'` (default) | All filter states and step diagnostics |
| `'none'` | Final state, log likelihood and continuation key only |
| Positive integer `L` | A ring buffer of the last `L` intervals |

A windowed trace starts from the filtering state just before its first interval,
so smoothing it conditions on that boundary rather than on discarded data.
To step a filter by hand, build it with `temporal_filter(model, num_particles)`
and call its `init` and `step`.

## Paths

`sample_paths(key, result.trace, model, theta, num_samples=M)` draws joint paths
of shape `(len(ts) + 1, M, D)`, including the initial state:

- A `LinearGaussianModel` trace is sampled exactly, backward through time.
  `smooth_gaussian_path(trace, model, theta)` returns the RTS means and covariances.
- A particle trace uses backward simulation (FFBSi, `O(T*N*M)`) when the model
  has `transition_logdensity(theta, x_next, x_previous, t_previous, t_next)`,
  and otherwise follows the stored ancestry (`O(T*M)`, prone to path collapse).

`particle_gibbs(key, reference_path, model, theta, ts, ys)` performs one
conditional SMC update with ancestor sampling (PGAS). Repeated calls form an MCMC
chain targeting the smoothing law, not independent draws.

## Unknown parameters: temporal SMC and SMC²

```python
from probjax.inference import run_temporal_smc, temporal_smc

prior = norm(jnp.log(0.3), 0.8)
smc = temporal_smc(model, prior.logpdf, num_particles=256)

init_key, run_key, sample_key = jax.random.split(key, 3)
state = smc.init(init_key, prior.sample(init_key, (512,)))
result = jax.jit(run_temporal_smc, static_argnums=0)(smc, run_key, state, ts, ys)
posterior = result.state.posterior_samples(sample_key, 1000)
```

Every parameter particle carries its own filter. At each observation:

1. Each filter advances one interval; its likelihood increment reweights the
   parameter particle, and the weights' log-normalizer adds to the evidence.
2. When the ESS falls below `ess_threshold`, parameters are resampled **together
   with** their filters and likelihood estimates.
3. Each resampled particle then makes `num_moves` Metropolis-Hastings moves. A
   proposal re-filters the observed prefix to get its likelihood.

The default proposal is a Gaussian random walk shaped by the weighted population
covariance, with a scale tuned towards 23.4% acceptance. Pass
`proposal_fn(key, theta) -> (theta', log_q_reverse_minus_forward)` to use your
own; a symmetric proposal returns zero. Parameters may be pytrees.

With a `LinearGaussianModel` the latent path is integrated out exactly. With a
`StateSpaceModel` the particle filter's likelihood estimate (not its logarithm) is
unbiased, which makes this SMC², a pseudo-marginal method. On rejection the
current estimate and filter are kept: refreshing a noisy estimate would change
the target.

`info` reports the ESS before resampling, whether resampling happened, the parent
indices, the move acceptance rate and the evidence increment. Check `info.valid`
after compiled runs: a step whose observation is impossible under every particle
leaves the state unchanged and reports `valid=False`.

### Continuing a run

Call `run_temporal_smc` again with the **whole** dataset so far. Observations the
state has already assimilated are skipped without consuming random keys, so two
calls give the same result as one:

```python
first = run_temporal_smc(smc, run_key, state, ts[:2], ys[:2])
second = run_temporal_smc(smc, first.key, first.state, ts, ys)
```

Moves need the full prefix to re-filter, which is why the earlier data is passed
again. A dataset that does not extend what the state has seen is rejected with
`valid=False`. `state.log_evidence` covers everything assimilated, while
`result.log_likelihood` covers only the current call.

### Joint parameter and path draws

`sample_joint_paths(key, state, model, ts, ys, num_samples=M, num_particles=N)`
draws parameters from the weighted population and then a path for each, by
re-filtering the assimilated data. Paths have shape `(M, T + 1, D)`. Gaussian
paths are exact given each drawn parameter; particle paths add a finite-particle
approximation.

## Cost

Filter storage per step is independent of the time horizon. `history='full'`
stores `O(T*N*D)` particles for one filter. Temporal SMC keeps `N_theta` filters,
`O(N_theta*N*D)`, and full outer history multiplies that by `T`, so the outer
runner defaults to `history='none'`. Moves re-filter the observed prefix, so
their cost grows with elapsed time; `batch_size` bounds how many parameter
particles are processed at once.

The executable [notebook](https://github.com/mackelab/probjax/blob/main/examples/inference/temporal_smc.ipynb)
compares exact Gaussian calculations with Kalman and particle parameter inference,
streaming, joint draws and a nonlinear PGAS example.

Algorithm references: [SMC²](https://arxiv.org/abs/1101.1528) and
[particle Gibbs with ancestor sampling](https://www.jmlr.org/beta/papers/v15/lindsten14a.html).
