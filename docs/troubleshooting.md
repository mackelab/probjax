# Troubleshooting

Every snippet on this page is executed by the test suite, so what is written
here is what actually happens.

## An inverse came back as NaN

Program inversion reports failure by returning NaN rather than raising. This is
almost always one of the unsupported patterns:

```python
import jax.numpy as jnp
from probjax.core import inverse

assert jnp.isnan(inverse(lambda x: x + jnp.tanh(x))(jnp.asarray(1.0)))
```

The full list is in [Program inversion](guides/program-inversion.md#what-can-be-inverted).
The short version: a variable used more than once **nonlinearly**, `fori_loop`,
a `scan` with a non-invertible carry, or `inverse(inverse(f))`. Affine reuse
such as `3 * x - x` is solved automatically. Check `jnp.isfinite` on results you
have not seen before.

If the NaN is only in *some* elements, that is a guard firing — the inverse does
not exist at those values, which is different from being unsupported. Turn it
into an error with `inverse_checks` plus `checkify`, as shown in the guide.

## "no usable INVERSE_LOGDET rule for ..."

The primitive inverts but has no log-determinant rule, and one is not guessed:
differentiating an inverse elementwise is only valid for elementwise maps.
Either avoid the primitive on the inverted path, or register a rule —
`register_rearrangement_inverse_logdet` if it only moves elements around.

`inverse` alone still works in this case; only `inverse_and_logabsdet` refuses.

## "Cannot abstractly evaluate a checkify.check"

`checkify.check` cannot be staged out by a plain `jit`. If you enabled
`inverse_checks()`, the call must be wrapped in `checkify.checkify`:

```python
import jax.numpy as jnp
from jax.experimental import checkify
from probjax.core import inverse
from probjax.core.registry import inverse_checks

f = lambda x: x * jnp.float32(0.0)
with inverse_checks():
    error, _ = checkify.checkify(inverse(f))(jnp.float32(5.0))
```

## "custom_inverse ...: dynamic argument N is a Foo"

An argument that is not an array or a pytree of arrays reached the primitive.
Either register the type as a JAX pytree so its arrays become visible, or list
its position in `static_argnums`.

The related message "*is a traced value, so it cannot be held as a static
parameter*" is the reverse: something traced was passed where a compile-time
constant was expected. Pass it as a dynamic positional argument instead.

## "definv was called twice"

A second registration of the same kind discards the first. Usually a module
imported twice, or a decorator applied in a loop. Registering `definv` and then
`definv_and_logdet` is fine and does not warn — that is the intended way to
supply both.

## Training loss is NaN

`fit` warns rather than failing silently, and returns the parameters as they
are — once a NaN gradient has been applied the run is dead:

```python
import jax
import jax.numpy as jnp
from probjax.stats import fit

def loss_fn(params, rng, batch):
    del rng
    return jnp.mean((batch["data"] - params["w"]) ** 2)

data = jax.random.normal(jax.random.key(0), (64, 2))
params, losses = fit(
    loss_fn, {"w": jnp.zeros(2)}, jax.random.key(1), {"data": data},
    num_steps=20, learning_rate=1e-2,
)
assert jnp.all(jnp.isfinite(losses))
```

Lower the learning rate, tighten `clip_norm` (10.0 by default), or look for an
unbounded transform in the model. Flows bound their scales precisely to avoid
this, so a NaN from a stock flow is worth reporting.

## Losses only arrive at the end of `fit`

That is by design. The loop is a single `jax.lax.scan` and never returns to
Python, which is what makes it compile once regardless of `num_steps`. Use
`on_step` to watch progress live, and return `False` from it to stop early.

## A list of arrays was treated as several batches

`fit` reads a list, tuple, generator or `DataLoader` as a **sequence of
batches**, and a bare array or a dict of arrays as **one batch**. Both readings
are pytrees of arrays, so this cannot be inferred — it is a fixed rule. A single
batch that groups several arrays must be a dict:

```python
import jax
import jax.numpy as jnp
from probjax.stats import is_batch_stream

array = jnp.zeros((4, 2))
assert not is_batch_stream(array)
assert not is_batch_stream({"data": array, "context": array})
assert is_batch_stream([array, array])
```

## "batch N has a different shape or dtype than the first one"

One compiled step serves every batch from an iterable, so they must agree. Pass
`drop_last=True` to the loader, or pad the final batch.

## JAX picks the wrong backend

```python
import jax
print(jax.devices())
```

Force one with `JAX_PLATFORMS=cpu` (or `metal,cpu`) before importing JAX. Metal
covers fewer primitives than CPU and CUDA; if something fails only there,
confirm against `JAX_PLATFORMS=cpu` before reporting it.

## `TracerArrayConversionError`

A traced value was used where Python needed a concrete one — `int()`, `float()`,
`if`, or `.item()` inside a `jit`. Move the conversion outside the jitted
function, or use `jax.lax.cond` and `jnp.where` for data-dependent branching.

## Shape errors inside a model

`jax.eval_shape` gives the shapes without running anything, which is usually
faster to read than a traceback:

```python
import jax
import jax.numpy as jnp
from flax import nnx
from probjax.nn import MLP

model = MLP([4, 16, 2], rngs=nnx.Rngs(0))
print(jax.eval_shape(model, jnp.ones((3, 4))))
```
