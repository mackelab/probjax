# Autoregressive models

An autoregressive model factorises `p(x) = prod_i p(x_i | x_<i)` and takes any
univariate family from `probjax.stats` as its conditional. Like every generative
family in `probjax.nn`, it trains with `fit` and exposes `as_dist()` for a frozen
distribution with `logpdf` and `sample`.

```python
import jax
from flax import nnx
from probjax.nn import MADE, MixtureAutoregressive

data = jax.random.normal(jax.random.key(0), (256, 3))

gaussian_head = MADE(3, rngs=nnx.Rngs(0))
flexible_head = MixtureAutoregressive(3, rngs=nnx.Rngs(0))

losses = flexible_head.fit(jax.random.key(1), data, num_steps=100)
```

`SplineAutoregressive`, `HistogramAutoregressive` and
`CategoricalAutoregressive` cover flexible continuous and discrete conditionals,
while `Autoregressive` accepts an `ARFamily` head directly.

- [Density estimation](density-estimation.md) — the shared `fit` / `as_dist` interface
- [Reference](../reference/nn.md) — every autoregressive family
