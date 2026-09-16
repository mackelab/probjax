"""Regularized beta inverses with cached solvers and implicit reverse AD."""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, betaln, ndtri, xlog1py, xlogy


def _solve(a, b, p, budget, upper):
    p = jnp.clip(p, 0.0, 1.0)
    valid = (a > 0) & (b > 0) & jnp.isfinite(a) & jnp.isfinite(b)
    interior = valid & (p > 0) & (p < 1)
    aa, bb = jnp.where(interior, a, 1.0), jnp.where(interior, b, 1.0)
    probability = jnp.where(interior, p, 0.5)
    # Solve in the smaller coordinate, rather than reflecting solely by p.
    midpoint = betainc(bb, aa, 0.5) if upper else betainc(aa, bb, 0.5)
    reflect = probability < midpoint if upper else probability > midpoint
    ar, br = jnp.where(reflect, bb, aa), jnp.where(reflect, aa, bb)
    complemented = jnp.logical_xor(reflect, upper)
    q = jnp.where(complemented, probability, 1 - probability)
    direct_upper = q < 0.01
    lower_target = jnp.where(complemented, 1 - probability, probability)
    target = jnp.where(direct_upper, q, lower_target)
    log_beta = betaln(ar, br)
    lower_guess = jnp.exp((jnp.log(ar) + log_beta + jnp.log(lower_target)) / ar)
    mean = ar / (ar + br)
    std = jnp.sqrt(ar * br / ((ar + br) ** 2 * (ar + br + 1)))
    normal = mean + std * ndtri(lower_target)
    lower_guess = jnp.where((ar >= 2) & (br >= 2) & (normal > 0), normal, lower_guess)
    upper_guess = -jnp.expm1((jnp.log(br) + log_beta + jnp.log(target)) / br)
    upper_guess = jnp.where(
        (ar >= 2) & (br >= 2),
        jnp.minimum(mean - std * ndtri(target), upper_guess),
        upper_guess,
    )
    guess = jnp.where(direct_upper & (upper_guess > 0), upper_guess, lower_guess)
    tiny, eps = jnp.finfo(p.dtype).tiny, jnp.finfo(p.dtype).eps
    guess = jnp.clip(guess, tiny, 0.5)

    def step(state):
        i, x, lo, hi, done = state
        error = jax.lax.cond(
            jnp.any(direct_upper),
            lambda: jnp.where(
                direct_upper,
                target - betainc(br, ar, 1 - x),
                betainc(ar, br, x) - target,
            ),
            lambda: betainc(ar, br, x) - target,
        )
        lo = jnp.where(error < 0, x, lo)
        hi = jnp.where(error >= 0, x, hi)
        pdf = jnp.exp(xlogy(ar - 1, x) + xlog1py(br - 1, -x) - log_beta)
        delta = error / pdf
        correction = jnp.clip(
            0.5 * delta * ((ar - 1) / x - (br - 1) / (1 - x)), -0.5, 0.5
        )
        candidate = x - delta / (1 - correction)
        candidate = jnp.where(
            jnp.isfinite(candidate) & (candidate > lo) & (candidate < hi),
            candidate,
            lo + (hi - lo) / 2,
        )
        done = (
            done
            | (jnp.abs(error) <= 4 * eps * target)
            | (jnp.abs(delta) <= 4 * eps * x)
        )
        return i + 1, jnp.where(done, x, candidate), lo, hi, done

    _, x, _, _, _ = jax.lax.while_loop(
        lambda s: (s[0] < budget) & jnp.any(~s[4]),
        step,
        (0, guess, jnp.zeros_like(p), jnp.full_like(p, 0.5), ~interior),
    )
    x = jnp.where(reflect, 1 - x, x)
    boundary = jnp.where(p == 0, 1.0 if upper else 0.0, 0.0 if upper else 1.0)
    return jnp.where(valid & ~jnp.isnan(p), jnp.where(interior, x, boundary), jnp.nan)


@lru_cache(maxsize=32)
def _make_betaincinv_core(max_halley_steps, max_bisection_steps, upper=False):
    budget = 2 + max_halley_steps + max_bisection_steps

    @jax.custom_vjp
    def core(a, b, p):
        return _solve(a, b, p, budget, upper)

    def forward(a, b, p):
        x = _solve(a, b, p, budget, upper)
        return x, (a, b, p, x)

    def backward(residuals, g):
        a, b, p, x = residuals
        boundary = (p <= 0) | (p >= 1)
        safe_x = jnp.where(boundary, 0.5, x)
        inverse_pdf = jnp.exp(
            betaln(a, b) - xlogy(a - 1, safe_x) - xlog1py(b - 1, -safe_x)
        )
        scale = jnp.finfo(x.dtype).eps ** (1 / 3)
        da, db = jnp.minimum(scale * a, a / 2), jnp.minimum(scale * b, b / 2)
        sign = -1 if upper else 1

        def cdf(aa, bb):
            return betainc(bb, aa, 1 - safe_x) if upper else betainc(aa, bb, safe_x)

        ca = sign * (cdf(a + da, b) - cdf(a - da, b)) / (2 * da)
        cb = sign * (cdf(a, b + db) - cdf(a, b - db)) / (2 * db)
        endpoint_pdf = jnp.exp(betaln(a, b) - xlogy(a - 1, x) - xlog1py(b - 1, -x))
        gp = sign * jnp.where(boundary, endpoint_pdf, inverse_pdf)
        gp = jnp.where((p < 0) | (p > 1), 0.0, gp)
        return (
            jnp.where(boundary, 0.0, -ca * inverse_pdf) * g,
            jnp.where(boundary, 0.0, -cb * inverse_pdf) * g,
            gp * g,
        )

    core.defvjp(forward, backward)
    return jax.jit(core)


def _inverse(a, b, p, max_halley_steps, max_bisection_steps, upper):
    for steps in (max_halley_steps, max_bisection_steps):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("Solver step budgets must be nonnegative integers")
    dtype = jnp.result_type(a, b, p, 1.0)
    a, b, p = jnp.broadcast_arrays(*(jnp.asarray(v, dtype) for v in (a, b, p)))
    return _make_betaincinv_core(max_halley_steps, max_bisection_steps, upper)(a, b, p)


def betaincinv(a, b, p, *, max_halley_steps=6, max_bisection_steps=15):
    """Invert regularized lower beta for finite positive shapes.

    Probabilities are clipped to [0, 1]; invalid shapes return NaN. The total
    safeguarded Halley/bisection budget is 2 plus both step settings.
    Reverse AD uses the inverse density and central differences for shapes;
    forward AD and higher shape derivatives are not supported.
    """
    return _inverse(a, b, p, max_halley_steps, max_bisection_steps, False)


def betainccinv(a, b, q, *, max_halley_steps=6, max_bisection_steps=15):
    """Invert regularized upper beta directly, preserving tiny q.

    q=0 returns one; q=1 returns zero. Domain, step settings and differentiation
    have the same semantics as betaincinv.
    """
    return _inverse(a, b, q, max_halley_steps, max_bisection_steps, True)
