# Density estimation

All the generative families share one interface: construct with `nnx.Rngs`, call
`fit` to train, and `as_dist()` for a frozen distribution with `logpdf` and
`sample`. Swapping a flow for an autoregressive model or a diffusion model means
a common training workflow. Sampling options, event shapes and the cost or
availability of likelihood evaluation still depend on the family.

## Normalizing flows

```python
import jax
import jax.numpy as jnp
from flax import nnx
from probjax.nn import nsf

data = jax.random.normal(jax.random.key(0), (512, 2)) * jnp.array([1.5, 0.5])

flow = nsf(2, 4, rngs=nnx.Rngs(0))
losses = flow.fit(jax.random.key(1), data, num_steps=200, batch_size=128)

distribution = flow.as_dist()
log_density = distribution.logpdf(data[:8])
draws = distribution.sample(jax.random.key(2), (16,))
```

`maf`, `nsf`, `naf`, `unaf`, `sospf`, `bpf` and `gf` are all available, along
with coupling variants. They differ in the bijector they compose; the interface
does not change.

Flows standardise their input inside `fit`, so data with a large mean or scale
does not need pre-scaling.

## Conditional densities

Pass `context_features` at construction and `context=` to `fit`. This gives
`q(x | c)`, which is what amortized inference is built on:

```python
import jax
from flax import nnx
from probjax.nn import maf

context = jax.random.normal(jax.random.key(0), (256, 1))
data = jax.random.normal(jax.random.key(1), (256, 2)) + context

flow = maf(2, 3, rngs=nnx.Rngs(0), context_features=1)
losses = flow.fit(jax.random.key(2), data, context=context, num_steps=100)
```

## Families

Every family below shares the `fit` / `as_dist` interface above; they differ in
the model class and the density they represent:

- [Autoregressive models](autoregressive-models.md) — `MADE`, `MixtureAutoregressive`,
  `SplineAutoregressive`, `HistogramAutoregressive`, `CategoricalAutoregressive`
- [Diffusion models](diffusion-models.md) — `EDM`, `VP`, `VE`, `CosineDM`,
  `MultinomialDiffusion`
- [Flow matching](flow-matching.md) — `FlowMatcher`, `MeanFlowMatcher`,
  `LinearFlow`, `LinearMeanFlow`

## Training

`fit` takes either the whole dataset or an iterable of batches. Without general
callbacks, updates run in a compiled `jax.lax.scan`; iterable batches are fetched
through host callbacks. Array-only fitting can be wrapped in JIT:

```python
import jax
from flax import nnx
from probjax.nn import maf

raw = jax.random.normal(jax.random.key(0), (512, 2))

# whole array
flow = maf(2, 2, rngs=nnx.Rngs(0))
flow.fit(jax.random.key(1), raw, num_steps=50, batch_size=128)

# or an iterable of batches, for data that does not fit in memory
batches = [raw[i:i + 128] for i in range(0, 512, 128)]
streamed = maf(2, 2, rngs=nnx.Rngs(0))
streamed.fit(jax.random.key(1), batches, num_steps=50)
```

A list, tuple, generator or `DataLoader` is read as a sequence of batches; a
bare array or a dict of arrays is one batch. With an iterable, every batch must
have the same shapes as the first, and `batch_size` belongs to the loader rather
than to `fit`.

To watch a long run or stop it early, pass `on_step`:

```python
import jax
from flax import nnx
from probjax.nn import maf

seen = []
flow = maf(2, 2, rngs=nnx.Rngs(0))
flow.fit(
    jax.random.key(1),
    jax.random.normal(jax.random.key(0), (256, 2)),
    num_steps=100,
    batch_size=64,
    on_step=lambda step, loss: seen.append((step, loss)),
    log_every=25,
)
```

Returning `False` from `on_step` stops training. The hook runs during training;
the returned loss history becomes available when `fit` finishes.

For EMA, validation, checkpoints or additional diagnostics, use `ema_decay`,
`return_result=True`, and `callbacks=[...]`. General callbacks receive
`(state, info)` between compiled chunks. Resume from `initial_state=result.state`
to preserve optimizer, RNG and EMA state. Under JIT, general Python callbacks
require explicit `callback_mode="io"`; this path does not support autodiff or
`vmap`. See [Training and EMA](../reference/nn.md#training-and-ema) for executable
examples, callback cadence and padded histories after early stopping.
