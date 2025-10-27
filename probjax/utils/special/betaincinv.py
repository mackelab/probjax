import jax
import jax.numpy as jnp
import jax.scipy
from jax.scipy.special import betainc, gammaln

# ------------------------------------------------------------
# Helper math utilities (mostly your code, lightly tidied)
# ------------------------------------------------------------


def _compute_initial_guess(a, b, p):
    """Heuristics for starting x."""
    large_ab = (a >= 2.0) & (b >= 2.0)

    x0_large = _initial_guess_large_ab(a, b, p)
    x0_small = _initial_guess_small_ab(a, b, p)

    x0 = jnp.where(large_ab, x0_large, x0_small)

    # For extreme left tail, override with asymptotic near 0
    x0 = jnp.where(((p < 0.3) | (b <= 1.0)), asymptotic_guess_p_to_0(a, b, p), x0)

    return jnp.clip(x0, 1e-3, 1 - 1e-3)


def bracket_x(a, b, p):
    """Return (lo, hi) such that betainc(a,b,lo) <= p <= betainc(a,b,hi)."""
    beta_ab = jax.scipy.special.beta(a, b)

    lower_bound = (p * a * beta_ab) ** (1.0 / a)
    lower_bound = jnp.where(b >= 1.0, lower_bound, 0.0)

    upper_bound1 = 1.0 - ((1.0 - p) * b * beta_ab) ** (1.0 / b)
    upper_bound2 = 1.0
    upper_bound = jnp.where(a < 1.0, upper_bound2, upper_bound1)

    lower_bound = jnp.clip(lower_bound, 1e-10, 1.0 - 1e-7)
    upper_bound = jnp.clip(upper_bound, 1e-10, 1.0 - 1e-7)
    return lower_bound, upper_bound


def _initial_guess_small_ab(a, b, p):
    """Initial guess when a or b is small."""
    eps = 1e-7
    ln_p = jnp.log(p + eps)
    ln_1mp = jnp.log1p(-p + eps)

    return jnp.where(
        p < 0.5,
        jnp.exp(ln_p / a),
        1.0 - jnp.exp(ln_1mp / b),
    )


def _initial_guess_large_ab(a, b, p):
    """Initial guess from normal approximation when a,b are both >= ~2."""
    mu = a / (a + b)
    sigma = jnp.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))

    p_clip = jnp.clip(p, 1e-7, 1.0 - 1e-7)
    z = jax.scipy.stats.norm.ppf(p_clip)
    x_approx = mu + sigma * z
    return jnp.clip(x_approx, 1e-10, 1.0 - 1e-10)


def asymptotic_guess_p_to_1(a, b, p):
    log_Bab = gammaln(a) + gammaln(b) - gammaln(a + b)
    # 1 - x ~ [b * B(a,b) * (1-p)]^(1/b)
    log_1mx = (1.0 / b) * (jnp.log(b) + log_Bab + jnp.log1p(-p))
    one_minus_x = jnp.exp(log_1mx)
    return 1.0 - one_minus_x


def asymptotic_guess_p_to_0(a, b, p):
    log_Bab = gammaln(a) + gammaln(b) - gammaln(a + b)
    # x ~ [a * B(a,b) * p]^(1/a)
    log_x = (1.0 / a) * (jnp.log(a) + log_Bab + jnp.log(p))
    return jnp.exp(log_x)


# ------------------------------------------------------------
# Root solver: Halley refinement + safeguarded bisection
# ------------------------------------------------------------


def _safe_betaincinv_solve(a, b, p, x_init, max_halley_steps=6, max_bisection_steps=15):
    """
    Halley-like refinement with bracketing plus bisection fallback.
    We maintain [lo, hi] that brackets the root, and keep clamping x.
    """

    eps = 1e-7

    def halley_step(carry, _):
        x, lo, hi = carry
        f_x = betainc(a, b, x)
        err = f_x - p

        # first derivative wrt x (beta pdf)
        log_pdf = (
            (a - 1.0) * jnp.log(x + eps)
            + (b - 1.0) * jnp.log1p(-x + eps)
            - gammaln(a)
            - gammaln(b)
            + gammaln(a + b)
        )
        deriv = jnp.exp(log_pdf)

        # second derivative wrt x
        second_deriv = deriv * ((a - 1.0) / (x + eps) - (b - 1.0) / (1.0 - x + eps))

        # Halley's update
        step = err / (deriv + eps)
        correction = 0.5 * step * (second_deriv / (deriv + eps))
        update = step / (1.0 - correction)
        x_new = x - update

        # stay inside the bracket
        x_new = jnp.clip(x_new, lo, hi)

        f_x_new = betainc(a, b, x_new)

        # shrink bracket
        lo_new = jnp.where(f_x_new < p, x_new, lo)
        hi_new = jnp.where(f_x_new >= p, x_new, hi)

        return (x_new, lo_new, hi_new), None

    def bisection_step(carry, _):
        x, lo, hi = carry
        x_new = 0.5 * (lo + hi)
        f_x_new = betainc(a, b, x_new)

        lo_new = jnp.where(f_x_new < p, x_new, lo)
        hi_new = jnp.where(f_x_new >= p, x_new, hi)

        return (x_new, lo_new, hi_new), None

    lo0, hi0 = bracket_x(a, b, p)
    x0 = jnp.clip(x_init, lo0, hi0)

    # A couple of Halley warmup steps
    (xH, loH, hiH), _ = jax.lax.scan(
        halley_step,
        (x0, lo0, hi0),
        xs=None,
        length=2,
        unroll=2,
    )

    # Some bisection cleanup to be safe / monotone
    (xB, loB, hiB), _ = jax.lax.scan(
        bisection_step,
        (xH, loH, hiH),
        xs=None,
        length=max_bisection_steps,
        unroll=2,
    )

    # Final Halley polish
    (xF, loF, hiF), _ = jax.lax.scan(
        halley_step,
        (xB, loB, hiB),
        xs=None,
        length=max_halley_steps,
        unroll=2,
    )

    return xF


# ------------------------------------------------------------
# Pure forward "body" (no custom_vjp here!)
# ------------------------------------------------------------


def _betaincinv_body(a, b, p, max_halley_steps, max_bisection_steps):
    """
    Numerically invert betainc(a,b,x)=p for x in [0,1].
    Returns x satisfying betainc(a,b,x)=p.
    """

    # 1. clip p, handle trivial
    p = jnp.clip(p, 0.0, 1.0)
    trivial = (p == 0.0) | (p == 1.0) | (a <= 0.0)
    x0_or_1 = jnp.where((p <= 0.0) | (a <= 0.0), 0.0, 1.0)

    # 2. symmetry trick for p>0.5 to improve conditioning
    reflect = p > 0.5
    p_ = jnp.where(reflect, 1.0 - p, p)
    a_ = jnp.where(reflect, b, a)
    b_ = jnp.where(reflect, a, b)

    # 3. initial guess
    x_init = _compute_initial_guess(a_, b_, p_)
    x_init = jnp.where(trivial, x0_or_1, x_init)

    # 4. refine using safeguarded root solve
    x_refined = _safe_betaincinv_solve(
        a_,
        b_,
        p_,
        x_init,
        max_halley_steps=max_halley_steps,
        max_bisection_steps=max_bisection_steps,
    )

    # 5. undo reflection
    x_final = jnp.where(reflect, 1.0 - x_refined, x_refined)

    # 6. exact 0 or 1 for boundary probs
    return jnp.where(trivial, x0_or_1, x_final)


# ------------------------------------------------------------
# Custom VJP wrapper (captures solver settings in a closure)
# ------------------------------------------------------------


def _make_betaincinv_core(max_halley_steps, max_bisection_steps):
    """
    Returns a differentiable function core(a,b,p) with custom_vjp.
    The step limits live in the closure, so JAX never sees them as args.
    """

    @jax.custom_vjp
    def _betaincinv_core(a, b, p):
        # forward primal value only
        return _betaincinv_body(a, b, p, max_halley_steps, max_bisection_steps)

    # fwd rule: must return (primal_out, residuals_for_bwd)
    # residuals can be anything we'll need in bwd. We'll keep a,b,p,y.
    def _betaincinv_core_fwd(a, b, p):
        y = _betaincinv_body(a, b, p, max_halley_steps, max_bisection_steps)
        return y, (a, b, p, y)

    # bwd rule: takes (residuals, g) and must return one cotangent
    # per original argument of _betaincinv_core (so: grad_a, grad_b, grad_p)
    # We also include "None" for nondiff stuff if we had extra args, but here
    # _betaincinv_core only has (a,b,p), so we just return 3 values. :contentReference[oaicite:1]{index=1}
    def _betaincinv_core_bwd(res, g):
        a, b, p, y = res
        eps = 1e-7

        # dbetainc/dy = Beta(a,b) pdf at x=y
        log_pdf = (
            (a - 1.0) * jnp.log(jnp.maximum(y, eps))
            + (b - 1.0) * jnp.log1p(-jnp.minimum(y, 1.0 - eps))
            - gammaln(a)
            - gammaln(b)
            + gammaln(a + b)
        )
        dbetainc_dy = jnp.exp(log_pdf)

        # dy/dp = 1 / (dbetainc/dy)
        dy_dp = 1.0 / (dbetainc_dy + eps)

        # partials wrt a,b via finite diff (coarse but serviceable)
        delta = 1e-5
        dbetainc_da = (betainc(a + delta, b, y) - betainc(a, b, y)) / delta
        dbetainc_db = (betainc(a, b + delta, y) - betainc(a, b, y)) / delta

        # implicit differentiation:
        # betainc(a,b,y) = p, F(a,b,y,p)=0
        # dy/da = - (dF/da)/(dF/dy) = - (∂betainc/∂a) / (∂betainc/∂y)
        grad_a = -dbetainc_da * dy_dp * g
        grad_b = -dbetainc_db * dy_dp * g
        grad_p = dy_dp * g

        return (grad_a, grad_b, grad_p)

    _betaincinv_core.defvjp(_betaincinv_core_fwd, _betaincinv_core_bwd)
    return _betaincinv_core


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def betaincinv(a, b, p, *, max_halley_steps=6, max_bisection_steps=15):
    """
    Inverse of the regularized incomplete beta function:
    returns x in [0,1] s.t. betainc(a,b,x) = p.

    a, b > 0
    p in [0,1]

    max_halley_steps, max_bisection_steps control solver refinement;
    they are treated as static "tuning knobs", not differentiable inputs.
    """
    core = _make_betaincinv_core(max_halley_steps, max_bisection_steps)
    # We can jit the core. The closure constants (step counts) are static,
    # and the JIT only sees (a,b,p) so there's no weird kwarg plumbing.
    core_jit = jax.jit(core)
    return core_jit(a, b, p)
