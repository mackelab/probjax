# Flow matching

Flow matching learns a velocity field that transports noise to data along a
probability path. Like every generative family in `probjax.nn`, matchers train
with `fit` and expose `as_dist()` for a frozen distribution with `logpdf` and
`sample`. They require an [`event_spec`](../glossary.md#event_spec) — the shape
of one event, excluding sample and batch axes.

```python
import jax
from flax import nnx
from probjax.nn import MLP, LinearFlow


class Velocity(nnx.Module):
    def __init__(self, rngs):
        self.net = MLP([3, 32, 3], rngs=rngs)

    def __call__(self, t, x, **kwargs):
        return self.net(x)


data = jax.random.normal(jax.random.key(0), (256, 3))
matcher = LinearFlow(Velocity(nnx.Rngs(0)), event_spec=3)
losses = matcher.fit(jax.random.key(1), data, num_steps=50)

view = matcher.as_dist()
samples = view.sample(jax.random.key(2), (4,))
assert samples.shape == (4, 3)
```

`FlowMatcher` and `MeanFlowMatcher` are the training objectives; `LinearFlow` and
`LinearMeanFlow` pair them with a linear probability path. Mean-flow training
defaults to Improved MeanFlow (`imf=True`). See
[Diffusion and flow matching](../reference/nn.md#diffusion-and-flow-matching)
for event-spec overrides and defaults.

- [Density estimation](density-estimation.md) — the shared `fit` / `as_dist` interface
- [Reference](../reference/nn.md#diffusion-and-flow-matching) — every flow-matching family
