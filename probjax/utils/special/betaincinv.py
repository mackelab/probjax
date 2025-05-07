import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, gammaln

# Scipy stats implementation missing in JAX


# Inverse beta cdf


# -------------------------------------------------------------------
# Core betaincinv function implementation
# -------------------------------------------------------------------


def _betaincinv_impl(a, b, p):
    """
    Implementation of the inverse of the regularized incomplete beta function.
    Returns x in (0,1) such that betainc(a, b, x) = p.

    Args:
        a (jnp.ndarray): Shape parameter (a > 0).
        b (jnp.ndarray): Shape parameter (b > 0).
        p (jnp.ndarray): Probability in [0, 1].

    Returns:
        jnp.ndarray: The value x in [0, 1] which satisfies
                     betainc(a, b, x) = p.
    """
    # Clip p to [0,1] and handle trivial cases
    p = jnp.clip(p, 0.0, 1.0)
    x0_or_1 = jnp.where((p <= 0.0) | (a <= 0.0), 0.0, 1.0)
    trivial = (p == 0.0) | (p == 1.0) | (a == 0)

    # Reflect if p > 0.5
    reflect = p > 0.5
    p_ = jnp.where(reflect, 1.0 - p, p)
    a_, b_ = jnp.where(reflect, b, a), jnp.where(reflect, a, b)

    # Initial guess
    x_init = _compute_initial_guess(a_, b_, p_)
    x_init = jnp.where(trivial, x0_or_1, x_init)

    # Safe solve (Newton + bisection)
    x_solved = _safe_betaincinv_solve(a_, b_, p_, x_init)

    # Reflect back if needed
    x_final = jnp.where(reflect, 1.0 - x_solved, x_solved)

    # Return x0_or_1 if p is exactly 0 or 1
    return jnp.where(trivial, x0_or_1, x_final)


def _betaincinv_fwd(a, b, p):
    """Forward pass for betaincinv custom vjp."""
    y = _betaincinv_impl(a, b, p)
    # Store values needed for backward pass
    return y, (a, b, p, y)


def _betaincinv_bwd(res, g):
    """Backward pass for betaincinv using the implicit function theorem.

    For inverse function y = betaincinv(a, b, p) where betainc(a, b, y) = p,
    we use implicit differentiation:

    If F(a, b, y, p) = betainc(a, b, y) - p = 0, then:
    dy/dp = -∂F/∂p / ∂F/∂y = 1 / ∂betainc(a,b,y)/∂y

    For a and b derivation, we similarly use:
    dy/da = -∂F/∂a / ∂F/∂y = -∂betainc(a,b,y)/∂a / ∂betainc(a,b,y)/∂y
    dy/db = -∂F/∂b / ∂F/∂y = -∂betainc(a,b,y)/∂b / ∂betainc(a,b,y)/∂y
    """
    a, b, p, y = res

    # Compute the derivative of betainc with respect to x at y
    # This is the PDF of the beta distribution
    eps = 1e-7
    # Compute the PDF more numerically stable way using logs
    log_pdf = (
        (a - 1) * jnp.log(jnp.maximum(y, eps))
        + (b - 1) * jnp.log1p(-jnp.minimum(y, 1 - eps))
        - gammaln(a)
        - gammaln(b)
        + gammaln(a + b)
    )
    dbetainc_dy = jnp.exp(log_pdf)

    # Compute dp/dy and its reciprocal dy/dp
    dy_dp = 1.0 / (dbetainc_dy + eps)  # Add eps for numerical stability

    # Compute partial derivatives of betainc with respect to a and b
    # Use finite differences or more accurate methods
    delta = 1e-5
    dbetainc_da = (betainc(a + delta, b, y) - betainc(a, b, y)) / delta
    dbetainc_db = (betainc(a, b + delta, y) - betainc(a, b, y)) / delta

    # Compute gradients using the chain rule
    grad_a = -dbetainc_da * dy_dp * g
    grad_b = -dbetainc_db * dy_dp * g
    grad_p = dy_dp * g

    return grad_a, grad_b, grad_p


# Create the custom VJP version of betaincinv
betaincinv = jax.custom_vjp(_betaincinv_impl)
betaincinv.defvjp(_betaincinv_fwd, _betaincinv_bwd)

# Apply JIT to the public function
betaincinv = jax.jit(betaincinv)


# -------------------------------------------------------------------
# Newton + Bisection Solver
# -------------------------------------------------------------------
def _safe_betaincinv_solve(a, b, p, x_init, max_halley_steps=6, max_bisection_steps=15):
    """
    Safe solver using Halley's method with bracket tracking and fallback bisection.
    """

    def halley_step(carry, _):
        x, lo, hi, f_x = carry
        err = f_x - p

        # Derivative of the regularized incomplete beta function
        deriv = jnp.exp(
            (a - 1) * jnp.log(x + 1e-7)
            + (b - 1) * jnp.log1p(-x + 1e-7)
            - jax.scipy.special.gammaln(a)
            - jax.scipy.special.gammaln(b)
            + jax.scipy.special.gammaln(a + b)
        )

        # Second derivative
        second_deriv = deriv * ((a - 1) / (x + 1e-7) - (b - 1) / (1 - x + 1e-7))

        # Halley's update step
        step = err / (deriv + 1e-7)
        correction = 0.5 * step * (second_deriv / (deriv + 1e-7))
        update = step / (1.0 - correction)

        # Propose a new x, clamp to [lo, hi]
        x_new = x - update
        # If large than upper bound -> replace with midpoint between x and upper bound
        # Fi smaller than lower bound -> replace with midpoint between x and lower bound
        x_new = jnp.clip(x_new, lo, hi)

        # Evaluate at new x
        f_x_new = jax.scipy.special.betainc(a, b, x_new)

        # Update bounds
        lo = jnp.where(f_x_new < p, x_new, lo)
        hi = jnp.where(f_x_new >= p, x_new, hi)

        return (x_new, lo, hi, f_x_new), None

    def bisection_step(carry, _):
        x, lo, hi, f_x = carry
        x_new = 0.5 * (lo + hi)
        f_x_new = betainc(a, b, x_new)
        lo = jnp.where(f_x_new < p, x_new, lo)
        hi = jnp.where(f_x_new >= p, x_new, hi)
        return (x_new, lo, hi, f_x_new), None

    # Ensure initial bounds bracket the root
    lo, hi = bracket_x(a, b, p)

    # Initial function value
    f_x = betainc(a, b, x_init)
    x = jnp.clip(x_init, lo, hi)

    # jax.debug.print("x={}. lo={}, hi={}", x, lo, hi)

    # Run Halley's method using scan
    (x, lo, hi, f_x), _ = jax.lax.scan(
        halley_step,
        (x_init, lo, hi, f_x),
        None,
        length=2,
        unroll=2,
    )

    # jax.debug.print("x={}. lo={}, hi={}", x, lo, hi)

    (x, lo, hi, f_x), _ = jax.lax.scan(
        bisection_step,
        (x_init, lo, hi, f_x),
        None,
        length=max_bisection_steps,
        unroll=2,
    )

    # jax.debug.print("x={}. lo={}, hi={}", x, lo, hi)

    # Run Halley's method using scan
    (x, lo, hi, f_x), _ = jax.lax.scan(
        halley_step,
        (x_init, lo, hi, f_x),
        None,
        length=max_halley_steps,
        unroll=2,
    )

    # jax.debug.print("x={}. lo={}, hi={}", x, lo, hi)

    return x


# -------------------------------------------------------------------
# Initial Guess Functions
# -------------------------------------------------------------------


def _compute_initial_guess(a, b, p):
    """
    Computes an initial guess for betaincinv using different strategies
    depending on the size of a and b.
    """
    large_ab = (a >= 2.0) & (b >= 2.0)

    x0 = jnp.where(
        large_ab,
        _initial_guess_large_ab(a, b, p),
        _initial_guess_small_ab(a, b, p),
    )

    x0 = jnp.where(((p < 0.3) | (b <= 1.0)), asymptotic_guess_p_to_0(a, b, p), x0)
    # x0 = jnp.where((p >= 0.8), asymptotic_guess_p_to_1(a, b, p), x0)

    return jnp.clip(x0, 1e-3, 1 - 1e-3)


def bracket_x(a, b, p):
    # Compute the beta function value
    beta_ab = jax.scipy.special.beta(a, b)

    # Lower bound for x
    lower_bound = (p * a * beta_ab) ** (1 / a)
    lower_bound = jnp.where(b >= 1.0, lower_bound, 0)

    # Upper bound for x
    upper_bound1 = 1 - ((1 - p) * b * beta_ab) ** (1 / b)
    upper_bound2 = 1.0
    upper_bound = jnp.where(a < 1, upper_bound2, upper_bound1)
    # Ensure bounds are within [0, 1]
    lower_bound = jnp.clip(lower_bound, 1e-10, 1.0 - 1e-7)
    upper_bound = jnp.clip(upper_bound, 1e-10, 1.0 - 1e-7)

    return lower_bound, upper_bound


# ---------------------------------------------
# Method 1: Small a or b (Edge Cases)
# ---------------------------------------------
def _initial_guess_small_ab(a, b, p):
    """
    Initial guess when a or b is small (e.g., < 1.0).
    """
    epsilon = 1e-7
    ln_p = jnp.log(p + epsilon)  # Avoid log(0)
    ln_1mp = jnp.log1p(-p + epsilon)  # Avoid log1p(0)

    return jnp.where(
        p < 0.5,
        jnp.exp(ln_p / a),
        1.0 - jnp.exp(ln_1mp / b),
    )


# ---------------------------------------------
# Method 2: Large a, b (Asymptotic Case)
# ---------------------------------------------
def _initial_guess_large_ab(a, b, p):
    """
    Initial guess using asymptotic expansion when a and b are large.
    """
    mu = a / (a + b)
    sigma = jnp.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    # Approximate the quantile using the normal distribution
    p = jnp.clip(p, 1e-7, 1 - 1e-7)
    z = jax.scipy.stats.norm.ppf(p)
    x_approx = mu + sigma * z

    # Clip to [0, 1] to avoid overflow
    return jnp.clip(x_approx, 1e-10, 1 - 1e-10)


def asymptotic_guess_p_to_1(a, b, p):
    # p = jnp.clip(p, 1e-10, 1 - 1e-7)
    # log(B(a,b)) = ln Γ(a) + ln Γ(b) - ln Γ(a+b)
    log_Bab = gammaln(a) + gammaln(b) - gammaln(a + b)
    # leading term: 1 - x ~ [b * B(a,b) * (1-p)]^(1/b)
    log_1mx = (1.0 / b) * (jnp.log(b) + log_Bab + jnp.log1p(-p))
    one_minus_x = jnp.exp(log_1mx)
    x = 1.0 - one_minus_x
    return x


def asymptotic_guess_p_to_0(a, b, p):
    # log(B(a,b)) = ln Γ(a) + ln Γ(b) - ln Γ(a+b)
    log_Bab = gammaln(a) + gammaln(b) - gammaln(a + b)
    # leading term: x ~ [a * B(a,b) * p]^(1/a)
    log_x = (1.0 / a) * (jnp.log(a) + log_Bab + jnp.log(p))
    x = jnp.exp(log_x)
    return x
