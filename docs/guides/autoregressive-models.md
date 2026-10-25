# Autoregressive models

An autoregressive model factorises `p(x) = prod_i p(x_i | x_<i)`. Two choices
make one: the *head*, which says what each conditional is, and the
*conditioner*, the causal network that predicts the head's parameters. Like
every generative family in `probjax.nn`, it trains with `fit` and exposes
`as_dist()` for a distribution with `logpdf` and `sample`.

```python
import jax
from flax import nnx
from probjax.nn import Autoregressive, MixtureHead

data = jax.random.normal(jax.random.key(0), (256, 3))

made = Autoregressive(3, rngs=nnx.Rngs(0))  # NormalHead: MADE
flexible = Autoregressive(3, MixtureHead(10), rngs=nnx.Rngs(0))

losses = flexible.fit(jax.random.key(1), data, num_steps=100).losses
```

The heads are small explicit classes:

| Head | Conditional |
| --- | --- |
| `NormalHead()` | Gaussian (the default, i.e. MADE) |
| `LocScaleHead(dist, fixed)` | any location-scale family, e.g. `LocScaleHead(laplace)` or `LocScaleHead(t, {"df": 5.0})` |
| `MixtureHead(num_components, kernel)` | Gaussian or logistic mixture, the flexible KDE-like head |
| `HistogramHead(num_bins, low, high, tails)` | piecewise constant, with optional exponential tails |
| `SplineHead(num_bins, bound)` | spline-warped normal |
| `CategoricalHead(num_classes)` | integer data, one-hot encoded for the conditioner |

A new family is a `Head` subclass that sets `dist` and `num_params` and maps
the raw outputs to the family's parameters in `params(raw)`.

## Transformer conditioner defaults

`TransformerARConditionerConfig()` uses model width 256, eight 32D attention
heads, four layers, and 4× FFN expansion. Use explicit width 128 for a compact
profile; smaller test dimensions and callable width rules remain configurable.
On CPU, `attention_fn=dot_product_attention` selects the dense softmax kernel.
See [profile timings and examples](../benchmarks/nn_defaults.md).

Spline distribution heads also damp boundary logits with
`tail_logit_scale=0.1`; pass one to restore undamped predictions. Initial slopes
are one and remain learnable. This matches the spline-flow tail convention.

- [Density estimation](density-estimation.md) — the shared `fit` / `as_dist` interface
- [Reference](../reference/nn.md) — every head and conditioner
