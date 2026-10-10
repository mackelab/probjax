# Diffusion models

Diffusion models learn a denoising process that maps noise to data. Like every
generative family in `probjax.nn`, they train with `fit` and expose `as_dist()`
for a frozen distribution with `logpdf` and `sample`. Continuous and
categorical diffusion require an [`event_spec`](../glossary.md#event_spec) — the
shape of one event, excluding sample and batch axes.

```python
import jax
from flax import nnx
from probjax.nn import EDM

key = jax.random.key(0)


class Denoiser(nnx.Module):
    def __call__(self, t, x):
        return 0.1 * x


diffusion = EDM(Denoiser(), event_spec=3, num_steps=3)
view = diffusion.as_dist()
samples = view.sample(key, (5,))
assert samples.shape == (5, 3)

# Override the event shape for one distribution view.
sequence_view = diffusion.as_dist(event_spec=(8, 3))
assert sequence_view.sample(key, (2,)).shape == (2, 8, 3)

# Change the default for future views; existing views keep their shapes.
diffusion.set_event_spec((16, 3))
assert diffusion.as_dist().event_shape == (16, 3)
assert view.event_shape == (3,)
```

`EDM`, `VP`, `VE` and `CosineDM` provide continuous denoising-diffusion
families. `MultinomialDiffusion` and its presets cover categorical data: their
event shapes count token positions and exclude the one-hot class axis, and they
use `int32` events. See
[Diffusion and flow matching](../reference/nn.md#diffusion-and-flow-matching)
for the full event-spec semantics.

- [Density estimation](density-estimation.md) — the shared `fit` / `as_dist` interface
- [Reference](../reference/nn.md#diffusion-and-flow-matching) — every diffusion family
