# Inference

Kernels are pure and separate from the loops that drive them. A kernel is built
from a log-density, initialised at a position, and then run by `MCMC` or `SMC`:

```python
import jax
import jax.numpy as jnp
from probjax.inference import MCMC, nuts

def logdensity(x):
    return -0.5 * jnp.sum(x**2)

kernel = nuts(logdensity)
state = kernel.init(jax.random.key(0), jnp.zeros(3))
params = kernel.init_params(state)

result = MCMC(kernel).sample(jax.random.key(1), state, 500, params)
draws = result.samples
```

`MCMC(...).run` keeps the full state and info; `.sample` keeps positions and
supports `thin`. Both are compiled scans.

## Kernels

| Family | Kernels |
| --- | --- |
| Gradient-based | `hmc`, `nuts`, `dynamic_hmc`, `mala`, `mclmc`, `adjusted_mclmc` |
| Gradient-free | `mh`, `gauss_rwmh`, `imh`, `gaussian_imh`, `slice`, `latent_slice`, `elliptical_slice`, `arms`, `a2rms` |
| Stochastic gradient | `sgld`, `sghmc`, `sgnht` |
| Composite | `pseudo_marginal` |

Warmup is a separate object, so the same kernel can be tuned different ways:

```python
import jax
import jax.numpy as jnp
from probjax.inference import MCMC, mala, step_size_adaptor

kernel = mala(lambda x: -0.5 * jnp.sum(x**2))
state = kernel.init(jax.random.key(0), jnp.zeros(2))
params = kernel.init_params(state)

tuned = MCMC(kernel).adapt(jax.random.key(1), step_size_adaptor(), state, params, 50)
```

`window_warmup`, `pathfinder_warmup` and `mclmc_warmup` cover the usual
strategies; `step_size_adaptor`, `mass_matrix_adaptor` and `covariance_adaptor`
compose for custom ones.

## Sequential Monte Carlo and filtering

`SMC` mirrors `MCMC` for particle methods (`smc`, `adaptive_smc`,
`persistent_smc`, `path_smc`). `probjax.inference.filtering` covers
`kalman_filter`, `extended_kalman_filter`, `unscented_kalman_filter`,
`sq_kalman_filter`, `rank_reduced_kalman_filter` and `ParticleFilter`, with
`rauch_tung_stribel_smoother` and `particle_smoother`.

## Variational inference

`flow_vi` fits a normalizing flow to an unnormalized target by reparameterised
reverse KL. It follows the shape of `blackjax.vi.meanfield_vi` — `init`, `step`,
`sample` — and the loop is yours:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from probjax.inference import flow_vi
from probjax.nn import maf

mean = jnp.array([2.0, -1.0])
def logdensity(x):
    return -0.5 * jnp.sum((x - mean) ** 2)

algorithm = flow_vi(logdensity, maf(2, 3, rngs=nnx.Rngs(0)), optax.adam(1e-3))

def one(state, key):
    state, info = algorithm.step(key, state)
    return state, info.elbo

state, objective = jax.lax.scan(
    jax.jit(one), algorithm.init(), jax.random.split(jax.random.key(0), 300)
)
draws = algorithm.sample(jax.random.key(1), state, 256)
```

`info.elbo` holds `mean(log q - log p)` — the quantity being minimised, so it
goes **down**. The name matches blackjax's `MFVIInfo` field; the sign is the
opposite of what it suggests.

Reverse KL is mode-seeking, so a flow fitted this way tends to under-cover. That
is what the next section is for.

## NeuTra: better geometry for the samplers

A posterior with strong curvature is hard for HMC because no single step size
suits every direction. `neutra` reparameterises the target through a flow so the
geometry becomes close to isotropic, then any kernel samples the transformed
target:

```python
import jax
import jax.numpy as jnp
from flax import nnx
from probjax.inference import MCMC, neutra, nuts
from probjax.nn import maf

def logdensity(x):
    return -0.5 * jnp.sum((x - jnp.array([2.0, -1.0])) ** 2)

flow = maf(2, 3, rngs=nnx.Rngs(0))        # normally fitted with flow_vi first
transform = neutra(logdensity, flow)

kernel = nuts(transform.logdensity)
state = kernel.init(jax.random.key(0), jnp.zeros(2))
result = MCMC(kernel).sample(
    jax.random.key(1), state, 200, kernel.init_params(state)
)
draws = transform.forward(result.samples)   # back in the target's space
```

This is a change of variables, not an approximation: **MCMC stays asymptotically
exact however poor the flow is.** A bad flow costs efficiency, never
correctness — which is what makes it safe to pair with a mode-seeking VI fit. On
Neal's funnel, NUTS on the transformed target reached an effective sample size
of 601 against 168 for NUTS on the target directly, at the same budget.

Because `neutra` returns a plain log-density and a map, it composes with every
kernel, warmup and runner above without any special integration.
