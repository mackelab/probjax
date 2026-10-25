# Inference

Kernels are pure and separate from the loops that drive them. A kernel is built
from a log-density, initialised at a position, and then run by `run`:

```python
import jax
import jax.numpy as jnp
from probjax.inference import nuts, run

def logdensity(x):
    return -0.5 * jnp.sum(x**2)

kernel = nuts(logdensity)
state = kernel.init(jax.random.key(0), jnp.zeros(3))
params = kernel.init_params(state)

result = run(jax.random.key(1), kernel, state, params, 500)
draws = result.trace["position"]
```

`run` is one compiled scan. `collect=` names the state or info fields to keep
(default `("position",)`), `thin=` keeps every n-th transition, `args=` feeds
per-step arguments such as minibatches, and `progress=True` shows a progress
bar. `step` takes a single transition, for writing your own loop.

The tutorials show all of this on real problems:
[MCMC in probjax](../tutorials/mcmc.md) (a kernel gallery, every adaptation
mode, many chains, pseudo-marginal MCMC),
[Neural aids for MCMC](../tutorials/neural_mcmc.md) (NeuTra and neural IMH),
[Exact sampling](../tutorials/rejection.md) (adaptive rejection and learned
proposals) and [Bayesian neural networks](../tutorials/bnn.md) (HMC versus stochastic
gradient MCMC).

## Kernels

| Family | Kernels |
| --- | --- |
| Gradient-based | `hmc`, `nuts`, `dynamic_hmc`, `mala`, `mclmc`, `adjusted_mclmc` |
| Gradient-free | `mh`, `gauss_rwmh`, `imh`, `gaussian_imh`, `slice`, `latent_slice`, `elliptical_slice`, `arms`, `a2rms` |
| Stochastic gradient | `sgld`, `sghmc`, `sgnht` |
| Composite | `pseudo_marginal` |
| Exact (iid draws) | `ars`, `rejection` |

Warmup is a separate object, so the same kernel can be tuned different ways:

```python
import jax
import jax.numpy as jnp
from probjax.inference import mala, step_size_adaptor, warmup

kernel = mala(lambda x: -0.5 * jnp.sum(x**2))
state = kernel.init(jax.random.key(0), jnp.zeros(2))
params = kernel.init_params(state)

tuned = warmup(jax.random.key(1), kernel, state, params, 50, step_size_adaptor())
```

`window_warmup`, `pathfinder_warmup` and `mclmc_warmup` cover the usual
strategies; `step_size_adaptor`, `mass_matrix_adaptor` and `covariance_adaptor`
compose for custom ones. Each kernel also carries its recommended policy as
`kernel.default_adaptation` — either an `Adaptor` or a `Warmup`
(`None` when there is nothing to tune), so
`warmup(key, kernel, state, params, n)` just works. Pass `policy=` (an
`Adaptor` or a `Warmup`) to override the default.

An adaptor can also keep adapting while you sample:
`run(..., adaptor=step_size_adaptor())` updates the parameters after every
transition and returns the finalized ones in `result.params`. To adapt several
chains as one, vmap with an axis name and wrap the adaptor in `pooled`:

```python
import jax
import jax.numpy as jnp
from probjax.inference import mala, pooled, run, step_size_adaptor

kernel = mala(lambda x: -0.5 * jnp.sum(x**2))
states = jax.vmap(lambda x: kernel.init(None, x))(jnp.zeros((4, 2)))
params = jax.vmap(kernel.init_params)(states)
keys = jax.random.split(jax.random.key(0), 4)

chains = jax.vmap(
    lambda key, state, params: run(
        key, kernel, state, params, 100, adaptor=pooled(step_size_adaptor())
    ),
    axis_name="chain",
)(keys, states, params)
```

## Exact samplers

`ars` and `rejection` are kernels too, but every step returns one
*independent, exact* draw, so `run` collects iid samples. What improves with
time follows the same rules as MCMC. The `ars` envelope tightens inside the
state with every target evaluation, which costs nothing in exactness. The
`rejection` bound `log M` is a parameter, which its default adaptation
(`bound_adaptor`) learns during warmup:

```python
import jax
import jax.numpy as jnp
from probjax.inference import Density, ars, rejection, run, warmup

normal = ars(lambda x: -0.5 * x**2)
draws = run(jax.random.key(0), normal, normal.init(jax.random.key(1), 0.0), None, 1000)

wide = Density(
    sample=lambda key, scale: scale * jax.random.normal(key, (2,)),
    logpdf=lambda x, scale: jax.scipy.stats.norm.logpdf(x, 0.0, scale).sum(),
)
kernel = rejection(lambda x: -0.5 * jnp.sum(x**2 / 0.5), proposal=wide)
state = kernel.init(jax.random.key(0), jnp.zeros(2))
tuned = warmup(jax.random.key(1), kernel, state, kernel.init_params(state, 2.0), 500)
exact = run(jax.random.key(2), kernel, tuned.state, tuned.params, 1000)
```

A learned proposal (a flow from `neural.density`) trains on the accepted
draws with `fit_adaptor(..., field="proposal")` composed with the bound
adaptor. The accepted draws are exact target samples, so they are the right
training data. The [exact sampling tutorial](../tutorials/rejection.md) shows
both samplers, including a trained flow proposal.

## Sequential Monte Carlo and filtering

SMC kernels follow the same contract, with a whole particle population as the
state, so `run` drives them too. `smc` tempers from the prior to the posterior
with MCMC moves, on a fixed or an adaptive schedule, and estimates the
evidence on the way; its default adaptation fits the move to the population
after every step. `diffusion_smc` follows a reverse diffusion, steered by a
pretrained model (`neural.diffusion_guide`) or by Monte Carlo score estimates
of the target (`score_guide`). The [SMC guide](smc.md) covers both.

Filters are kernels too. `kalman_filter`, `square_root_kalman_filter` and
`rank_reduced_kalman_filter` filter a `LinearGaussianModel` exactly,
`extended_kalman_filter` and `unscented_kalman_filter` a nonlinear
`GaussianModel`, and `particle_filter` any `StateSpaceModel`. The model
parameters `theta` are the filter's params, so `run` filters a series and
`jax.grad` of `state.log_likelihood` differentiates the likelihood of `theta`.
`smooth`, `sample_paths` and `temporal_smc` (SMC²) build on them; the
[filtering guide](temporal-smc.md) covers them.

## Learned components

A neural proposal, variational family or transport map is a pure function plus
a parameter pytree. `probjax.inference.neural` splits a `probjax.nn` flow into
exactly that, so its weights can sit in a kernel's `params`, inside a compiled
scan, `vmap`-ed over chains:

```python
from flax import nnx
from probjax.inference.neural import bijection, density
from probjax.nn import maf

flow = maf(2, 3, rngs=nnx.Rngs(0))
family, flow_params = density(flow)    # Density(sample, logpdf), weights
forward, flow_params = bijection(flow)  # (z, params) -> (x, log|det J|), weights
```

`Density(sample, logpdf)` takes the parameters explicitly:
`sample(key, params)` and `logpdf(x, params)`. Any pair of functions with those
signatures works, not just flows.

Training is an adaptor. `fit_adaptor(loss_fn, optimizer, ...)` keeps a ring
buffer of what the chain produces and takes optimizer steps on `params` (or one
field of it) inside the scan. `every=1, num_fit_steps=1` trains a little after
each transition; `every=250, num_fit_steps=250` alternates sampling and fitting
blocks. With `axis_name=`, chains run under `jax.vmap` average their gradients
and train one shared set of parameters.

### Neural independent Metropolis-Hastings

`imh(logdensity, proposal=family)` proposes from any `Density`. Its default
adaptation trains the proposal by maximum likelihood on the chain's output:

```python
import jax
import jax.numpy as jnp
from flax import nnx
from probjax.inference import imh, run, warmup
from probjax.inference.neural import density
from probjax.nn import maf

def logdensity(x):
    return -0.5 * jnp.sum((x - jnp.array([2.0, -1.0])) ** 2)

proposal, proposal_params = density(maf(2, 3, rngs=nnx.Rngs(0)))
kernel = imh(logdensity, proposal=proposal)
state = kernel.init(jax.random.key(0), jnp.zeros(2))
warm = warmup(jax.random.key(1), kernel, state, proposal_params, 2000)
result = run(jax.random.key(2), kernel, warm.state, warm.params, 1000)
```

Stop adapting before collecting the draws you report: with a fixed proposal
the chain is an exact MCMC kernel.

### Variational inference

`flow_vi` fits a `Density` to an unnormalized target by reparameterised
reverse KL. It is a kernel whose step is one optimizer step, so `run` drives
it like any sampler:

```python
import optax
from probjax.inference import flow_vi

family, flow_params = density(flow)
vi = flow_vi(logdensity, family, optax.adam(1e-3))
fit = run(jax.random.key(0), vi, vi.init(flow_params), None, 2000, collect=("loss",))
fitted_params = fit.state.params
```

`loss` is `mean(log q - log p)`, the negative ELBO up to the target's unknown
log-normaliser, so it goes **down**. Reverse KL is mode-seeking, so a flow
fitted this way tends to under-cover. That is what NeuTra is for.

### NeuTra: better geometry for the samplers

A posterior with strong curvature is hard for HMC because no single step size
suits every direction. `neutra` reparameterises the target through a flow so the
geometry becomes close to isotropic, then any kernel samples the transformed
target:

```python
from probjax.inference import neutra, nuts

forward, _ = bijection(flow)
# fitted_params come from the flow_vi run above, which used the same flow
transform = neutra(logdensity, forward, fitted_params)

kernel = nuts(transform.logdensity)
state = kernel.init(jax.random.key(0), jnp.zeros(2))
result = run(jax.random.key(1), kernel, state, kernel.init_params(state), 200)
draws = transform.forward(result.trace["position"])   # back in the target's space
```

This is a change of variables, not an approximation: **MCMC stays asymptotically
exact however poor the flow is.** A bad flow costs efficiency, never
correctness, which is what makes it safe to pair with a mode-seeking VI fit.
The [neural MCMC tutorial](../tutorials/neural_mcmc.md) runs both methods on a
funnel and a multimodal ring.

## How a kernel is built

Every algorithm in `probjax` — MCMC kernels, SMC kernels, filters, ODE and SDE
solvers — follows the same three steps, so the same habits work everywhere.

| Role | What it is | Example |
| --- | --- | --- |
| **API class** | a namespace holding `init`, `init_params` and `build_step` | `hmc`, `kalman_filter`, `dopri5` |
| **kernel** | a pure, callable `(init, step, ...)` container | `hmc(logdensity)` |
| **runner** | the compiled loop that drives a kernel | `run(key, kernel, ...)` |

An API class is turned into a kernel by calling it with the model's functions.
Options that are known earlier can be bound first with `configure`, which
returns a value you can pass around, store, or hand to another kernel:

```python
import jax.numpy as jnp
from probjax.inference import hmc, smc

# Bound now, applied to the model later.
fast_hmc = hmc.configure(num_integration_steps=10)
kernel = fast_hmc(lambda x: -0.5 * jnp.sum(x**2))

# The same value works wherever a kernel API is expected.
smc_kernel = smc(
    lambda x: -0.5 * jnp.sum(x**2),
    lambda x: -0.5 * jnp.sum((x - 1.0) ** 2),
    move=fast_hmc,
)
```

Bound options are routed to whichever of `init`, `init_params` and `build_step`
accepts them, so one set of options can describe a kernel's whole
configuration. A configured value is hashable and compares by value, so it can
be a static `jit` argument without retracing.

Where the model's function is a single required argument, the shorthand
`hmc(num_integration_steps=10)` returns the same configured value. Filters
take more than one model function, so they use `configure` explicitly.

## Tutorials

Worked examples live in [Tutorials](../tutorials/smc.md): start with
[Kalman filtering](../tutorials/kalman_filter.md) for the linear-Gaussian case,
then [Sequential Monte Carlo](../tutorials/smc.md) for the general one.
