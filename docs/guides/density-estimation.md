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
losses = flow.fit(
    jax.random.key(1), data, num_steps=200, batch_size=128,
).losses

distribution = flow.as_dist()
log_density = distribution.logpdf(data[:8])
draws = distribution.sample(jax.random.key(2), (16,))
```

`maf`, `nsf`, `naf`, `unaf`, `sospf`, `bpf` and `gf` are all available, along
with coupling variants. They differ in the bijector they compose; the interface
does not change.

Flows standardise their input inside `fit`, so data with a large mean or scale
does not need pre-scaling.

### Combining feature mixers

Use `CompositeMixingConfig` to apply several mixers between flow transforms.
The configs run in the order listed, so a fixed permutation can precede or
follow learnable circulant–diagonal mixing:

```python
from probjax.nn import (
    CirculantDiagonalMixingConfig,
    CompositeMixingConfig,
    PermuteMixingConfig,
    nsf,
)

mixing = CompositeMixingConfig((
    PermuteMixingConfig(permutation=(2, 0, 3, 1)),
    CirculantDiagonalMixingConfig(num_diagonals=2),
))
mixed_flow = nsf(4, 3, mixing=mixing, rngs=nnx.Rngs(0))
mixed_samples = mixed_flow.as_dist().sample(jax.random.key(3), (8,))
assert mixed_samples.shape == (8, 4)
assert jnp.all(jnp.isfinite(mixed_flow.as_dist().logpdf(mixed_samples)))
```

`PermuteMixingConfig()` without explicit indices draws a random permutation
once for each mixing layer at construction. Those permutations remain fixed
during training. Supplying indices uses the same ordering in every flow block.
Each block has its own learned circulant–diagonal parameters; reversing the
config order generally gives a different transform.

Circulant–diagonal mixing alternates per-feature scaling with circular
convolution, evaluated through FFTs. `num_diagonals=2` applies a diagonal,
a circulant, and a second diagonal in that order. Inversion reverses those
operations. The permutation contributes zero log-absolute-determinant, while
the learned factors contribute their scale and Fourier-amplitude terms.
The mixer initializes near the identity; `init_std=0` starts exactly at it.
Per-factor log-scales are bounded by `max_log_scale`, but products of many
factors can still become ill-conditioned. Small factor counts restrict the
linear maps it can represent. See [CDFlow](https://arxiv.org/abs/2510.25323)
for the structured-matrix approach; this implementation uses positive diagonal
scales and positive real DC/Nyquist eigenvalues.

`CompositeMixingConfig` also works with flips, rotations, and custom configs
implementing `MixingConfigProtocol`. `NoMixingConfig` entries are omitted.

### Learnable spline tails

Spline presets accept domain bounds and `tail_logit_scale` to control boundary
slope predictions. For example:

```python
tail_controlled_flow = nsf(
    4, 3, rngs=nnx.Rngs(1), mixing=mixing,
    x_min=-6.0, x_max=6.0, y_min=-6.0, y_max=6.0,
    tail_logit_scale=0.1,
)
```

The default domain is `[-5, 5]`; wider bounds put more values inside the spline
but reduce knot resolution at a fixed bin count. `tail_logit_scale` defaults
to 0.1. It damps only the two boundary-slope logits, including their
bias contributions. The tails still start at unit slope and remain learnable;
their asymptotic slope limits do not change. Boundary slopes affect the edge
bins as well as extrapolation outside the domain. These are tuning options,
not a guarantee that a stack of learned inverses remains well-conditioned.
Use `tail_logit_scale=1.0` to restore undamped boundary predictions, or a smaller
positive value for stronger damping.

The [mixing/tail study](../benchmarks/flow_mixing.md) and
[representative workloads](../benchmarks/flow_mixing_realistic.md) include
held-out likelihoods, runtime, and tail diagnostics. Small reproducible commands
are in the [benchmark index](../benchmarks/index.md).

## Conditional densities

Pass `context_features` at construction and put the context in the `data` dict
of `fit`. This gives
`q(x | c)`, which is what amortized inference is built on:

```python
import jax
from flax import nnx
from probjax.nn import maf

context = jax.random.normal(jax.random.key(0), (256, 1))
data = jax.random.normal(jax.random.key(1), (256, 2)) + context

flow = maf(2, 3, rngs=nnx.Rngs(0), context_features=1)
batch = {"data": data, "context": context}
losses = flow.fit(jax.random.key(2), batch, num_steps=100,).losses
```

## Families

Every family below shares the `fit` / `as_dist` interface above; they differ in
the model class and the density they represent:

- [Autoregressive models](autoregressive-models.md) — `Autoregressive` with
  `NormalHead`, `MixtureHead`, `SplineHead`, `HistogramHead`, `CategoricalHead`
- [Diffusion models](diffusion-models.md) — `EDM`, `VP`, `VE`, `CosineDM`,
  `MultinomialDiffusion`
- [Flow matching](flow-matching.md) — `FlowMatcher`, `MeanFlowMatcher`,
  `LinearFlow`, `LinearMeanFlow`

## Training

`fit` takes either the whole dataset or an iterable of batches. An array (or a
dict of arrays) is minibatched on the device in a compiled `jax.lax.scan`; any
other iterable is a stream, pulled one batch per update:

```python
import jax
from flax import nnx
from probjax.nn import maf

raw = jax.random.normal(jax.random.key(0), (512, 2))

# whole array
flow = maf(2, 2, rngs=nnx.Rngs(0))
flow.fit(jax.random.key(1), raw, num_steps=50, batch_size=128)

# or an iterable of batches, for data that does not fit in memory
batches = [{"data": raw[i:i + 128]} for i in range(0, 512, 128)]
streamed = maf(2, 2, rngs=nnx.Rngs(0))
streamed.fit(jax.random.key(1), batches, num_steps=4)
```

A list, generator or `DataLoader` is a stream; a dict, array or tuple is one
batch. A stream is the caller's iterator: it is advanced once per step and
never restarted, so it has to hold at least as many batches as the run takes
steps. The batch size belongs to the loader rather than to `fit`.

To watch a long run or stop it early, pass a callback:

```python
import jax
from flax import nnx
from probjax.nn import maf

def watch(state, info):
    print(int(state.step), float(info.loss))
    return True  # False stops the run

flow = maf(2, 2, rngs=nnx.Rngs(0))
flow.fit(
    jax.random.key(1),
    jax.random.normal(jax.random.key(0), (256, 2)),
    num_steps=100, batch_size=64, callback=watch, every=25,
)
```

The callback receives `(state, info)` every `every` updates and runs on the
host, so it cannot be used inside JIT. For EMA, validation, checkpoints or
additional diagnostics, use `ema_decay`, `use_ema` and `callback=`. Resume
from `state=result.state` to preserve optimizer, RNG and EMA state. See
[Training and EMA](../reference/nn.md#training-and-ema) for executable examples
and the callback cadence.
