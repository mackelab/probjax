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
is the term a change of variables needs. When the program is structurally
volume-preserving — rearrangements, translations, sign flips, `±1` scalings —
the log-det is proven zero from the jaxpr and nothing is staged for it, so the
compiled inverse matches a hand-written one equation for equation. The same
holds piecemeal: a volume-preserving stretch inside a larger program (a
negation or shift around an `exp`, a reversal) contributes no log-det equations
of its own — only the parts that genuinely scale stage arithmetic.

## What can be inverted

The interpreter works one equation at a time, so it inverts a *tree* of
operations. It is not a solver. These are the limits, and they are worth knowing
because most of them fail **silently, by returning NaN**:

| Pattern | Result |
| --- | --- |
| A tree of operations — each value used once | works |
| `lax.scan` with an invertible carry, `lax.cond` | works |
| A variable used twice, **affinely** — `3 * x - x`, `A @ x + b`, `sum(x) - x` | works, by linear solve |
| An overdetermined affine map — `tile`, `concat([x, x])`, padding | works, with `input_template` (least squares; log-det is `nan`) |
| A joint system over several arguments, `invertible_arg=(0, 1)` | works |
| `jnp.fft.fft` / `ifft`, scatter-add, `maximum`/`minimum` on the active side | works |
| A variable used twice, **nonlinearly** — `x * x`, a residual `x + f(x)` | `nan` |
| `lax.fori_loop`, or `lax.scan` carrying anything not itself invertible | `nan` |
| `inverse(inverse(f))` | `nan` |
| `lax.while_loop` | raises |

A variable used twice is what breaks local propagation: in `3 * x - x` the
subtraction has two unknown operands, and the bivariate rules need exactly one.
When the stalled program is affine in the target the inverse is a linear solve,
so that case is handled automatically:

```python
import jax.numpy as jnp
from probjax.core import inverse, inverse_and_logabsdet

x = inverse(lambda t: 3.0 * t - t)(jnp.array([4.0, 6.0]))
assert jnp.allclose(x, jnp.array([2.0, 3.0]))

# including coupling between components, and the log-determinant
x, log_det = inverse_and_logabsdet(lambda t: jnp.sum(t) * jnp.ones(2) - t)(
    jnp.array([1.0, 2.0])
)
assert jnp.allclose(x, jnp.array([2.0, 1.0]))
```

Affinity is decided from the jaxpr rather than sampled, so it is a proof: a
nonlinear fan-out cannot slip through by happening to look linear at a few
points.

```python
import jax.numpy as jnp
from probjax.core import inverse

assert jnp.isnan(inverse(lambda t: t + jnp.tanh(t))(jnp.asarray(1.0)))
```

A residual *is* invertible when its branch is a contraction — by fixed-point
iteration, `x <- y - f(x)` — but nothing in a jaxpr states a Lipschitz bound, so
that guarantee has to come from you. Register it with `custom_inverse` below.

Two caveats on the affine path. Pointwise maps (`x + x`, `3 * x - x`) invert
elementwise in O(n); small coupled maps build the matrix with one vmapped sweep;
only large coupled maps pay for an iterative solve — and there the log-det comes
back NaN, since it needs the dense matrix. For very large coupled inputs a
hand-written `custom_inverse` is still better. And the path runs only after
propagation fails, so ordinary inverses are untouched: `2 * x + 1` still goes
through the rules and emits just a `sub` and a `div`.

Check `jnp.isfinite` on the result if you are inverting something you have not
inverted before.

## Tracing with the wrong shape: `input_template`

`inverse` traces your function with the outputs you pass in, which is exact
whenever inputs and outputs share their structure. When they do not — `tile`,
padding, `split` — pass an example input so the true program is staged:

```python
import jax.numpy as jnp
from probjax.core import inverse

x = jnp.array([1.0, 2.0])
assert jnp.allclose(
    inverse(lambda t: jnp.tile(t, 2), input_template=x)(jnp.tile(x, 2)), x
)
```

The same idea solves several arguments jointly: `invertible_arg=(0, 1)`
inverts `(x, y) -> (x + y, x - y)` back to `(x, y)`. Outputs the templated
program could not have produced still report NaN rather than a value.

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
