# Program inversion

ProbJax can invert a JAX function. It traces the function to a jaxpr and walks
it backwards, replacing each primitive with its registered inverse, so the
result is ordinary JAX code — no interpreter remains at runtime, and the output
is `jit`-able and `vmap`-able like anything else.

```python
import jax.numpy as jnp
from probjax.core import inverse, inverse_and_logabsdet

def forward(x):
    return 2.0 * jnp.exp(x) + 1.0

y = forward(jnp.asarray(0.4))

x = inverse(forward)(y)
x, log_det = inverse_and_logabsdet(forward)(y)
```

`log_det` is the log-determinant of the **inverse** map, `log|d(inv)/dy|`, which
is the term a change of variables needs.

## What can be inverted

The interpreter works one equation at a time, so it inverts a *tree* of
operations. It is not a solver. These are the limits, and they are worth knowing
because most of them fail **silently, by returning NaN**:

| Pattern | Result |
| --- | --- |
| A variable used twice — `3 * x - x`, `exp(x) * exp(x)`, a residual `x + f(x)` | `nan` |
| `lax.fori_loop`, or `lax.scan` carrying anything not itself invertible | `nan` |
| `inverse(inverse(f))` | `nan` |
| `lax.while_loop` | raises |
| `lax.scan` with an invertible carry, `lax.cond` | works |

Each of the NaN cases is invertible in principle; recovering `x` from
`3 * x - x` just means solving an equation rather than applying rules backwards.
Check `jnp.isfinite` on the result if you are inverting something you have not
inverted before.

```python
import jax.numpy as jnp
from probjax.core import inverse

fanned_out = inverse(lambda x: 3.0 * x - x)(jnp.asarray(4.0))
assert jnp.isnan(fanned_out)          # invertible in principle, unsupported here
```

## Guards: when an inverse only exists for some values

Some inverses are exact wherever they are defined and need no checking — a
domain violation surfaces as NaN through IEEE for free, since `log(-1)` and
`atanh(2)` are already NaN.

Others depend on the value at runtime. Dividing by something that is only zero
sometimes, or a forward map whose image is bounded, cannot be settled while
tracing. Those rules carry a guard, and produce NaN exactly where the inverse
does not exist:

```python
import jax.numpy as jnp
from probjax.core import inverse

scale = jnp.asarray([0.0, 2.0])
recovered = inverse(lambda x: x * scale)(jnp.asarray([5.0, 4.0]))
# element 0 has no preimage; element 1 does
assert jnp.isnan(recovered[0]) and jnp.allclose(recovered[1], 2.0)
```

Guards that can be settled while tracing cost nothing — `x * 2.0` has a literal
multiplier, so no runtime check is emitted at all, and the generated inverse is
exactly the arithmetic you would write by hand.

To turn a guard into a real error rather than a NaN, pair `inverse_checks` with
`jax.experimental.checkify`. It is opt-in because `checkify.check` cannot be
staged out by a plain `jit`:

```python
import jax.numpy as jnp
from jax.experimental import checkify
from probjax.core import inverse
from probjax.core.registry import inverse_checks

zero_scale = lambda x: x * jnp.float32(0.0)
with inverse_checks():
    error, _ = checkify.checkify(inverse(zero_scale))(jnp.float32(5.0))
assert "no inverse at this value" in str(error.get())
```

## Supplying your own inverse

`custom_inverse` attaches an analytic inverse to a function, which the
interpreter uses instead of trying to work one out. This is how the normalizing
flows get their exact, cheap inverses:

```python
import jax.numpy as jnp
from probjax.core import custom_inverse, inverse_and_logabsdet

@custom_inverse
def affine(x, scale, shift):
    return x * scale + shift

@affine.definv_and_logdet
def _(y, scale, shift):
    return (y - shift) / scale, -jnp.log(jnp.abs(scale))

x, log_det = inverse_and_logabsdet(affine)(jnp.asarray(7.0), 2.0, 1.0)
```

`inv_argnum` selects which argument is inverted when it is not the first, and
`static_argnums` marks arguments that are configuration rather than data.
Negative indices count from the end.

The registered inverse is checked against the argument it claims to invert: if
it returns the wrong pytree, shape, or dtype kind, you get an error naming the
mismatch rather than a quietly wrong result.

## Log-determinants

`inverse_and_logabsdet` needs a log-determinant rule for every primitive on the
path. Elementwise primitives have closed forms; rearrangements like `reshape`,
`concatenate` and `slice` contribute zero because they move elements without
scaling them.

A primitive that inverts but has no log-determinant rule **raises**, naming
itself and the rule to add. It is not approximated — differentiating an inverse
elementwise is only valid for elementwise maps, and guessing produced numbers
that were not log-determinants:

```python
import jax.numpy as jnp
from probjax.core import inverse, inverse_and_logabsdet

# dynamic_slice inverts, but recovers only its window, so the input is left
# partially known and there is no square Jacobian to take a determinant of.
import jax
windowed = lambda x: jax.lax.dynamic_slice(jnp.exp(x), (1,), (2,))
assert jnp.all(jnp.isfinite(inverse(windowed)(jnp.ones(2))))

try:
    inverse_and_logabsdet(windowed)(jnp.ones(2))
except NotImplementedError as error:
    assert "dynamic_slice" in str(error)
```
