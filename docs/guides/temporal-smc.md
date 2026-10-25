# Filtering and temporal SMC

Filtering moves forward through physical time, one observation at a time. With
known parameters `theta` a filter tracks the latent state `p(x_t | y_1:t)` and
the likelihood `p(y_1:t | theta)`; with unknown parameters temporal SMC also
targets `p(theta | y_1:t)`. The static `smc` kernel stays the tool for
annealing a stationary target.

## Writing a model

A model is a set of pure functions of `theta`. Which filter can run it depends
on its form:

| Model | Functions | Filters |
| --- | --- | --- |
| `LinearGaussianModel` | `initial(theta) -> (m, P)`, `transition(theta, t0, t1) -> (Phi, Q)`, `observation(theta, t) -> (C, R)` | `kalman_filter` (exact), `square_root_kalman_filter`, `rank_reduced_kalman_filter` |
| `GaussianModel` | as above, but `transition(theta, x, t0, t1) -> (f(x), Q)` and `observation(theta, x, t) -> (h(x), R)` | `extended_kalman_filter`, `unscented_kalman_filter` |
| `StateSpaceModel` | `init(key, theta)`, `step(key, theta, x, t0, t1)`, `log_likelihood(theta, x, y, t)` for one latent state | `particle_filter` |

A `StateSpaceModel` is written for **one** latent state, which may be any
pytree; the particle filter vectorizes it over particles.

```python
import jax
import jax.numpy as jnp
from probjax.stats import norm
from probjax.inference import LinearGaussianModel, StateSpaceModel

observation_sd = 0.45


def init(key, theta):  # a draw from the initial law at t0
    return norm(jnp.zeros(1), 1.0).sample(key)


def step(key, theta, x, t_previous, t_next):  # a draw over one interval
    sd = jnp.sqrt(jnp.exp(theta) * (t_next - t_previous))
    return x + sd * jax.random.normal(key, x.shape)


def log_likelihood(theta, x, y, t):  # normalized observation density
    return norm(x, observation_sd).logpdf(y).sum()


def transition_logdensity(theta, x_next, x, t_previous, t_next):
    sd = jnp.sqrt(jnp.exp(theta) * (t_next - t_previous))
    return norm(x, sd).logpdf(x_next).sum()


model = StateSpaceModel(init, step, log_likelihood, transition_logdensity)

# The same random walk, written for the Kalman filter.
linear_model = LinearGaussianModel(
    initial=lambda theta: (jnp.zeros(1), jnp.eye(1)),
    transition=lambda theta, t0, t1: (jnp.eye(1), jnp.exp(theta) * (t1 - t0) * jnp.eye(1)),
    observation=lambda theta, t: (jnp.eye(1), observation_sd**2 * jnp.eye(1)),
)
```

Times are physical, so irregular grids need no special handling. The initial
law is at `t0` and observations come after it; fold any observation at `t0`
into the initial law yourself.

## Filtering at known parameters

A filter is a kernel: `init(key, theta, t0=0.0)` and `step(key, state, theta,
t, y, observed=True)`, with `theta` as its params. `run` filters a series in
one compiled scan, taking the times and observations as per-step arguments:

```python
from probjax.inference import kalman_filter, particle_filter, run

ts = jnp.array([0.2, 0.7, 1.4, 2.0])
ys = jnp.array([[0.4], [-0.1], [0.8], [0.5]])
key = jax.random.key(0)
theta = jnp.log(0.3)

kalman = kalman_filter(linear_model)
particles = particle_filter(model, num_particles=512)
exact = run(key, kalman, kalman.init(key, theta), theta, len(ts), args=(ts, ys))
estimate = run(key, particles, particles.init(key, theta), theta, len(ts), args=(ts, ys))
print(exact.state.log_likelihood, estimate.state.log_likelihood)
```

`state.log_likelihood` accumulates `log p(y_1:t | theta)`: exact for the
Kalman filters, an unbiased estimate (of the likelihood, not its log) for the
particle filter. A third argument, a boolean mask, marks missing observations
(their values may be NaN); masked steps only predict.

Because `theta` is an ordinary argument, gradients, `vmap` and outer samplers
see it directly:

```python
def log_likelihood_of(theta):
    state = kalman.init(key, theta)
    return run(key, kalman, state, theta, len(ts), args=(ts, ys)).state.log_likelihood


print(jax.grad(log_likelihood_of)(theta))  # fit theta by gradient ascent
print(jax.vmap(log_likelihood_of)(jnp.linspace(-3.0, 1.0, 5)))  # a profile
```

The particle filter's estimate also differentiates; `unbiased_gradients=True`
passes the gradient of the weights through resampling. Its `proposal=` takes a
guided proposal that sees the next observation, which lowers the variance.

## Smoothing and paths

`collect=("state", "info")` keeps every filter state and step diagnostic.
Smoothing and path sampling take them with the initial state, so their results
cover the initial time and every filtered time:

```python
from probjax.inference import sample_paths, smooth

initial = kalman.init(key, theta)
trace = run(key, kalman, initial, theta, len(ts), args=(ts, ys), collect=("state", "info")).trace
means, covs = smooth(linear_model, theta, initial, trace["state"], trace["info"])
paths = sample_paths(key, linear_model, theta, initial, trace["state"], trace["info"], num_samples=100)
print(means.shape, paths.shape)  # (5, 1) and (5, 100, 1)
```

- A Gaussian filter's output is smoothed by Rauch--Tung--Striebel and sampled
  backward through time: exactly for a `LinearGaussianModel`, linearised for a
  `GaussianModel`.
- A particle filter's output is sampled by backward simulation (FFBSi,
  `O(T*N*M)`) when the model has `transition_logdensity`, and otherwise by
  following the stored ancestry (`O(T*M)`, prone to path collapse).

`particle_gibbs(key, reference_path, model, theta, ts, ys)` performs one
conditional SMC update with ancestor sampling (PGAS). Repeated calls form an
MCMC chain targeting the smoothing law, not independent draws.

## Unknown parameters: temporal SMC and SMC²

```python
from probjax.inference import temporal_smc

prior = norm(jnp.log(0.3), 0.8)
smc = temporal_smc(particles, prior.logpdf, max_observations=len(ts))

init_key, run_key, sample_key = jax.random.split(key, 3)
state = smc.init(init_key, prior.sample(init_key, (256,)), ys[0])
result = run(run_key, smc, state, None, len(ts), args=(ts, ys), collect=("valid",))
posterior = result.state.posterior_samples(sample_key, 1000)
```

Every parameter particle carries its own filter, built for one `theta`. At
each observation:

1. Each filter advances one interval; its likelihood increment reweights the
   parameter particle, and the weights' log-normalizer adds to the evidence.
2. When the ESS falls below `ess_threshold`, parameters are resampled **together
   with** their filters.
3. Each resampled particle then makes `num_moves` Metropolis-Hastings moves. A
   proposal re-filters the observations so far to get its likelihood.

The kernel keeps the observations it has assimilated (up to
`max_observations`) for those moves, so it steps like a filter: one
observation per step, and running more steps continues the run. `init` takes
any observation to size that buffer.

The default proposal is a Gaussian random walk shaped by the weighted population
covariance, with a scale tuned towards 23.4% acceptance. Pass
`proposal_fn(key, theta) -> (theta', log_q_reverse_minus_forward)` to use your
own; a symmetric proposal returns zero. Parameters may be pytrees.

With a Kalman filter the latent path is integrated out exactly. With a particle
filter the likelihood estimate is unbiased, which makes this SMC², a
pseudo-marginal method. On rejection the current estimate and filter are kept:
refreshing a noisy estimate would change the target.

`info` reports the ESS before resampling, whether resampling happened, the
parent indices, the move acceptance rate and the evidence increment. Check
`info.valid` after compiled runs: a step whose observation is impossible under
every particle, whose time does not advance, or that overflows the buffer
leaves the state unchanged and reports `valid=False`.

`sample_joint_paths(key, state, filter, model, num_samples=M)` draws
parameters from the weighted population and then a path for each, by
re-filtering the assimilated data. Paths have shape `(M, T + 1, D)`.

## Cost

A filter's state does not grow with the horizon; `collect=("state", "info")`
stores `O(T*N*D)` for a particle filter. Temporal SMC keeps `N_theta` filters,
`O(N_theta*N*D)`, plus the observation buffer. Moves re-filter the observed
prefix, so their cost grows with elapsed time; `batch_size` bounds how many
parameter particles are processed at once.

The executable [notebook](https://github.com/mackelab/probjax/blob/main/examples/inference/temporal_smc.ipynb)
compares exact Gaussian calculations with Kalman and particle parameter inference,
streaming, joint draws and a nonlinear PGAS example.

Algorithm references: [SMC²](https://arxiv.org/abs/1101.1528) and
[particle Gibbs with ancestor sampling](https://www.jmlr.org/beta/papers/v15/lindsten14a.html).
