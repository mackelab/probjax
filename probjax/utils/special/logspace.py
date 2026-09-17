"""Stable subtraction in log space."""

import jax
import jax.numpy as jnp


@jax.jit
def log1mexp(x):
    """Compute log(1-exp(x)) for x <= 0; positive inputs return NaN.

    Zero returns -inf and -inf returns zero. Inactive branch arguments are
    guarded to keep interior gradients finite near zero.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(x, 1.0))
    left = x < -jnp.log(2.0)
    return jnp.where(
        left,
        jnp.log1p(-jnp.exp(jnp.where(left, x, -1.0))),
        jnp.log(-jnp.expm1(jnp.where(left, -1.0, x))),
    )


@jax.jit
def logdiffexp(a, b):
    """Compute log(exp(a)-exp(b)), broadcasting a >= b.

    Equal finite inputs and (-inf, -inf) return -inf. Invalid domains,
    including (+inf, +inf), return NaN. Gradients require finite a > b.
    """
    dtype = jnp.result_type(a, b, 1.0)
    a, b = jnp.broadcast_arrays(jnp.asarray(a, dtype), jnp.asarray(b, dtype))
    both_zero = jnp.isneginf(a) & jnp.isneginf(b)
    difference = jnp.where(both_zero, -1.0, b - a)
    return jnp.where(both_zero, -jnp.inf, a + log1mexp(difference))
