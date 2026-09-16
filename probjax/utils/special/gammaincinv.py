"""Regularized gamma inverses with implicit differentiation."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaincc, gammaln, ndtri, xlogy


def _solve(a, p, upper):
    p = jnp.clip(p, 0.0, 1.0)
    valid = (a > 0) & jnp.isfinite(a)
    interior = valid & (p > 0) & (p < 1)
    shape = jnp.where(interior, a, 1.0)
    probability = jnp.where(interior, p, 0.5)
    upper_tail = probability < 0.5 if upper else probability > 0.5
    target = jnp.where(
        upper_tail,
        probability if upper else 1 - probability,
        1 - probability if upper else probability,
    )
    log_lower = jnp.where(upper_tail, jnp.log1p(-target), jnp.log(target))
    small = jnp.exp((log_lower + gammaln(shape + 1)) / shape)
    z = jnp.where(upper_tail, -ndtri(target), ndtri(target))
    normal = (
        shape * jnp.maximum(1 - 1 / (9 * shape) + z / (3 * jnp.sqrt(shape)), 0) ** 3
    )
    # Revert the small-x gamma series; useful when the CDF underflows during
    # refinement and for probabilities far below the normal approximation.
    c2 = 1 / (shape + 1)
    c3 = (3 * shape + 5) / (2 * (shape + 1) ** 2 * (shape + 2))
    c4 = (8 * shape**2 + 33 * shape + 31) / (
        3 * (shape + 1) ** 3 * (shape + 2) * (shape + 3)
    )
    series = small * (1 + small * (c2 + small * (c3 + small * c4)))
    guess = jnp.where((shape >= 1) & (normal > 0), normal, small)
    guess = jnp.where(small < 0.1 * (shape + 1), series, guess)
    guess = jnp.where(
        (shape < 1) & upper_tail, jnp.maximum(small, -jnp.log(target)), guess
    )
    tiny, eps = jnp.finfo(p.dtype).tiny, jnp.finfo(p.dtype).eps
    guess = jnp.maximum(guess, tiny)

    def residual(x):
        return jnp.where(
            upper_tail, target - gammaincc(shape, x), gammainc(shape, x) - target
        )

    def expand(state):
        i, hi = state
        return i + 1, jnp.where(residual(hi) < 0, 2 * hi, hi)

    _, hi = jax.lax.while_loop(
        lambda s: (s[0] < 32) & jnp.any(residual(s[1]) < 0),
        expand,
        (0, jnp.maximum(guess * 2, shape + 1)),
    )

    def step(state):
        i, x, lo, hi, done = state
        error = residual(x)
        lo = jnp.where(error < 0, x, lo)
        hi = jnp.where(error >= 0, x, hi)
        pdf = jnp.exp(xlogy(shape - 1, x) - x - gammaln(shape))
        delta = error / pdf
        correction = jnp.clip(0.5 * delta * ((shape - 1) / x - 1), -0.5, 0.5)
        candidate = x - delta / (1 - correction)
        candidate = jnp.where(
            jnp.isfinite(candidate) & (candidate > lo) & (candidate < hi),
            candidate,
            lo + (hi - lo) / 2,
        )
        converged = (jnp.abs(error) <= 8 * eps * target) | (
            jnp.abs(delta) <= 8 * eps * x
        )
        done = done | converged
        return i + 1, jnp.where(done, x, candidate), lo, hi, done

    _, x, _, _, _ = jax.lax.while_loop(
        lambda s: (s[0] < 64) & jnp.any(~s[4]),
        step,
        (0, guess, jnp.zeros_like(p), hi, ~interior),
    )
    boundary = jnp.where(p == 0, jnp.inf if upper else 0.0, 0.0 if upper else jnp.inf)
    return jnp.where(valid & ~jnp.isnan(p), jnp.where(interior, x, boundary), jnp.nan)


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _core(a, p, upper):
    return _solve(a, p, upper)


def _core_jvp(upper, primals, tangents):
    a, p = primals
    da, dp = tangents
    x = _core(a, p, upper)
    boundary = (p <= 0) | (p >= 1)
    safe_x = jnp.where(boundary, 1.0, x)
    inv_pdf = jnp.exp(safe_x - xlogy(a - 1, safe_x) + gammaln(a))
    shape_grad = jnp.where(boundary, 0.0, -jax.lax.igamma_grad_a(a, safe_x) * inv_pdf)
    endpoint_pdf = jnp.exp(x - xlogy(a - 1, x) + gammaln(a))
    endpoint_pdf = jnp.where(jnp.isposinf(x), jnp.inf, endpoint_pdf)
    prob_grad = jnp.where(boundary, endpoint_pdf, inv_pdf) * (-1 if upper else 1)
    prob_grad = jnp.where((p < 0) | (p > 1), 0.0, prob_grad)
    tangent = jnp.zeros_like(x)
    if not isinstance(da, jax.custom_derivatives.SymbolicZero):
        tangent = tangent + shape_grad * da
    if not isinstance(dp, jax.custom_derivatives.SymbolicZero):
        tangent = tangent + prob_grad * dp
    return x, tangent


_core.defjvp(_core_jvp, symbolic_zeros=True)


def _inverse(a, p, upper):
    dtype = jnp.result_type(a, p, 1.0)
    a, p = jnp.broadcast_arrays(jnp.asarray(a, dtype), jnp.asarray(p, dtype))
    return _core(a, p, upper)


@jax.jit
def gammaincinv(a, p):
    """Invert regularized lower gamma; finite a > 0, p clipped to [0, 1]."""
    return _inverse(a, p, False)


@jax.jit
def gammainccinv(a, q):
    """Invert regularized upper gamma directly, preserving tiny q.

    Finite a > 0 is required. Probabilities are clipped to [0, 1]; q=0
    returns infinity and q=1 returns zero. Invalid shapes return NaN.
    Supports implicit forward and reverse differentiation.
    """
    return _inverse(a, q, True)
