# Temporal SMC and trajectory inference

Temporal inference advances a posterior through physical time, targeting
`p(theta, x[t0:t] | y[t0:t])`. The fixed-space `path_smc` API remains useful for
annealing a stationary target. Temporal inference instead propagates a filter
state for each interval and accumulates predictive likelihood increments.

## Incremental filtering

`kalman_backend` wraps the existing Kalman kernel. `particle_backend` wraps the
existing particle-filter kernel and uses discrete systematic resampling. Both
implement:

```python skip
state = backend.init(key, theta, t0)
state, info = backend.step(key, state, theta, t_next, observation, observed=True)
```

The state includes the previous time. The grid must increase strictly; transition
callbacks receive both endpoints, so irregular intervals work without hidden
unit-time assumptions. The initial distribution is at `t0`; observations passed
to the runner are at later times. Incorporate any observation at `t0` in that
initial distribution yourself. Missing observations use `observed=False` and
perform only prediction. A boolean array masks missing observations in a batch;
observation values keep their fixed shape and can be NaN at masked positions.

For a scalar Gaussian random walk:

```python
import jax
import jax.numpy as jnp
from probjax.inference import kalman_backend, run_temporal_filter

backend = kalman_backend(
    lambda theta, t: (jnp.zeros(1), jnp.eye(1)),
    lambda theta, old, new: (jnp.eye(1), jnp.eye(1) * jnp.exp(theta) * (new-old)),
    lambda theta, t: (jnp.eye(1), jnp.eye(1) * 0.2),
)
key = jax.random.key(0)
theta = jnp.array(-1.0)
ts = jnp.array([0.2, 0.7, 1.4])
y = jnp.array([[0.4], [-0.1], [0.8]])
initial_key, run_key = jax.random.split(key)
initial = backend.init(initial_key, theta, 0.0)
run = jax.jit(lambda k, s: run_temporal_filter(backend, k, s, theta, ts, y))
result = run(run_key, initial)
```

`result.log_likelihood` sums normalized predictive increments for this run;
`result.key` permits continuing the same random stream. For identical streaming
results, split `key, step_key = jax.random.split(key)` before each call to `step`.
Only the scan setup is interpreted in Python; compiled updates have fixed-shape
PyTree carries and no growing Python containers.

Storage is explicit:

| `history` | Retained output |
| --- | --- |
| `'none'` | Final filter state, likelihood sum, continuation key |
| `'full'` | All states and step diagnostics |
| Positive integer `L` | A ring buffer for the last `L` intervals of this run |

A windowed trace includes the filtering state immediately before its first
retained interval. Its smoothing distribution is conditional on that boundary
filtering law. It cannot recover discarded states or their full-history smoothed
estimates. Histories describe the current runner invocation; they are not
implicitly concatenated across streaming chunks.

## Joint trajectories

The sampling functions include the trace's initial time in their output, with
shape `(retained_intervals + 1, num_samples, state_dimension)`.

- `smooth_gaussian_path(trace, transition_fn)` returns RTS means and covariances.
- `sample_gaussian_paths(key, trace, transition_fn, num_samples=M)` draws joint
  Gaussian paths using backward conditional distributions. The callback returns
  the transition matrix for two times. Independent marginal draws would lose
  cross-time dependence.
- `sample_particle_paths(..., method='ancestry')` follows stored parent indices.
- `sample_particle_paths(..., method='backward', transition_logdensity_fn=...)`
  reuses the existing FFBSi smoother, now supporting an arbitrary sample count.
  Its callback is `(x_next, x_previous, t_previous, t_next) -> scalar log density`.

Particle traces must use actual discrete ancestry. Optimal-transport combinations
of particles do not supply such ancestry. The temporal particle adapter therefore
uses the existing systematic resampler, rather than accepting arbitrary transport
resampling callbacks. With `N` filter particles and `M` paths, backward simulation
costs `O(T*N*M)`; ancestry reconstruction costs `O(T*M)` but can suffer from path
collapse. State dimension and transition-density evaluation costs are additional.

`particle_gibbs` performs one conditional bootstrap particle-filter update with
optional ancestor sampling (PGAS). Its reference path includes the initial state.
The initial sampler must draw independent samples from the initial law; callbacks
and density conventions are documented in its API. It uses multinomial
resampling at each step because pinning a particle in an ordinary systematic
resampler would not implement valid conditional resampling. Repeated calls are
an MCMC chain targeting the conditional smoothing law, not independent draws.

## Parameters: Kalman SMC and SMC²

```python
from jax.scipy.stats import norm
from probjax.inference import temporal_smc, run_temporal_smc

kernel = temporal_smc(
    backend,
    logprior_fn=lambda theta: norm.logpdf(theta),
    proposal_fn=lambda key, theta: (theta + 0.2 * jax.random.normal(key), 0.0),
    ess_threshold=0.5,
    num_rejuvenation_steps=2,
)
prior_key, init_key, run_key = jax.random.split(key, 3)
parameters = jax.random.normal(prior_key, (64,))
state = kernel.init(init_key, parameters, 0.0)
result = jax.jit(lambda k, s: run_temporal_smc(kernel, k, s, ts, y))(run_key, state)
```

Initialization expects equally weighted draws from the supplied prior. Parameter
particles may be a PyTree, with a shared leading population axis. At each time:

1. Vectorize the backend step over parameters and associated filter states.
2. Multiply outer weights by predictive likelihood increments and accumulate
   the outer evidence increment.
3. If ESS is low, resample the parameters **together with** filter states and
   cumulative likelihood estimates.
4. If configured, rejuvenate using proposals and full-prefix likelihood replay.

This reuses BlackJAX's ESS, systematic resampling, and Metropolis accept/reject
utility. The small temporal outer step retains incremental filter states and
adaptive resampling semantics that the fixed-space SMC step does not provide.
The inner and outer batch APIs share one temporal runner and history mechanism.

The proposal returns `(candidate, log_q_reverse_minus_forward)`. A symmetric
random walk returns zero. With a Kalman backend the latent trajectory is integrated
out exactly. With a particle backend, the likelihood estimate (not its logarithm)
is unbiased, yielding the SMC²/pseudo-marginal construction. On rejection the
current likelihood estimate and filter state are retained. Averaging noisy log
likelihoods or independently refreshing the current estimate would change the
target. Approximate likelihood backends are rejected by the outer constructor.

`info` reports outer ESS before resampling, parent indices, whether resampling
occurred, the realized rejuvenation acceptance fraction (zero when no moves ran),
an evidence increment, and `valid`. Check `valid` after compiled execution:
all-impossible observations give invalid diagnostics and leave the state unchanged.
Insufficient replay data raise eagerly. Under JIT, invalid replay bounds, a
non-increasing current time, or an unfinished observation bridge leave the outer
state unchanged and set `valid=False`.
Replay contents must match the observations previously assimilated; the API
checks bounds/current time, not equality of the entire historical dataset.

When resuming a run with rejuvenation, supply `ReplayData(ts, observations, mask)`
starting at the original `t0`. It must include the full prefix, not just the new
chunk. It may include future data: replay reads only the assimilated prefix.
`state.log_evidence` covers all processed intervals, while
`result.log_likelihood` covers the current invocation. Ordinary updates do not
replay past observations. Parameter proposals generally do, so their cost grows
with elapsed time. Without a proposal, the method is sequential importance
sampling/resampling and cannot rejuvenate a collapsed parameter population.

## Cost and supported scope

Current-state filtering storage is independent of the time horizon. A full trace
uses `O(T*N*D)` particle storage, while a ring buffer uses `O(L*N*D)`. Outer SMC²
stores a population of `N_theta` filters, costing `O(N_theta*N_x*D)` for particle
states. Full outer history multiplies this by the time horizon. Input data storage,
model matrices, and covariance work are additional costs. Use `history='none'`
(the outer default) unless past populations are needed.

`benchmarks/temporal_smc.py` reports compile time separately from warm execution
and compiler-estimated temporary/output memory. On the development CPU, a
256-particle filter over 128 intervals ran in about 1 ms after compilation. Its
retained output was about 2 KB for no history, 29 KB for an eight-interval window,
and 400 KB for full history. These are illustrative local measurements, not
portable performance guarantees. Prefix replay can dominate cost: forcing
rejuvenation at every observation deliberately exposes that expense in the benchmark.

Supported models use a discrete physical-time grid, fixed state/particle shapes,
and normalized observation densities. Continuous dynamics can supply exact or
numerically discretized interval transitions. Observation tempering, adaptive proposals/move counts, and inner-particle capacity
adaptation are described in the [SMC extensions guide](smc.md). Generic nonlinear
Gaussian approximations are not labeled exact/unbiased by this API. The supplied
Kalman backend assumes a linear-Gaussian model.

The executable [notebook](https://github.com/mackelab/probjax/blob/main/examples/inference/temporal_smc.ipynb)
compares exact Gaussian calculations, Kalman and particle parameter inference,
streaming execution, joint parameter/path draws, and a nonlinear PGAS example.

Algorithm references: [SMC²](https://arxiv.org/abs/1101.1528) and
[Particle Gibbs with ancestor sampling](https://www.jmlr.org/beta/papers/v15/lindsten14a.html).

## Persistent streaming windows and joint samples

`init_streaming_window`, `append_streaming_window`, and `streaming_window_trace`
provide an explicit ring-buffer carry that survives separate runner invocations.
Keep it alongside the current filter/SMC state. The returned fixed-shape trace has
a validity mask until capacity is filled. Do not pass padded entries to a smoother;
slice on the host or wait until the buffer is full. Append only successful steps.

`sample_joint_paths` factors a joint draw into a weighted parameter draw and a
conditional trajectory draw, reusing the existing forward runner and backward
samplers. It returns `(parameters, paths, parameter_indices, valid)`, with paths
shaped `(num_samples, num_intervals + 1, state_dimension)`. Replay data must contain
exactly the assimilated prefix. A mismatch raises eagerly; inside JIT it produces
`valid=False` and NaN paths. Gaussian conditionals are exact for each drawn parameter.
Particle smoothing adds a finite-particle approximation; it is not an exact sample
from the stored SMC² auxiliary state.
