import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, logsumexp

# Scipy stats implementation missing in JAX


# Inverse beta cdf


# -------------------------------------------------------------------
# Core betaincinv function with reflection and fallback
# -------------------------------------------------------------------


@jax.jit
def betaincinv(a, b, p):
    """
    Inverse of the regularized incomplete beta function (betainc).
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
    x0_or_1 = jnp.where(p <= 0.0, 0.0, 1.0)
    trivial = (p == 0.0) | (p == 1.0)

    # Reflect if p > 0.5
    reflect = p > 0.5
    p_ = jnp.where(reflect, 1.0 - p, p)
    a_, b_ = jnp.where(reflect, b, a), jnp.where(reflect, a, b)

    # Initial guess
    x_init = _compute_initial_guess(p_, a_, b_)
    x_init = jnp.where(trivial, x0_or_1, x_init)

    # Safe solve (Newton + bisection)
    x_solved = _safe_betaincinv_solve(a_, b_, p_, x_init)

    # Reflect back if needed
    x_final = jnp.where(reflect, 1.0 - x_solved, x_solved)

    # Return x0_or_1 if p is exactly 0 or 1
    return jnp.where(trivial, x0_or_1, x_final)


# -------------------------------------------------------------------
# Newton + Bisection Solver
# -------------------------------------------------------------------


def _safe_betaincinv_solve(a, b, p, x_init, max_newton_steps=12, max_bisect_steps=6):
    """
    Safe solver using Newton iteration with bracket tracking and fallback bisection.
    """

    def newton_step(carry, _):
        x, lo, hi, f_x = carry
        err = f_x - p
        deriv = jnp.exp(
            (a - 1) * jnp.log(x)
            + (b - 1) * jnp.log1p(-x)
            - jax.scipy.special.betaln(a, b)
        )
        step = jnp.where(jnp.abs(deriv) > 1e-8, err / (deriv + 1e-30), 0.0)

        x_new = jnp.clip(x - step, jnp.maximum(lo, 0.0), jnp.minimum(hi, 1.0))
        f_x_new = betainc(a, b, x_new)

        # Update bounds
        lo = jnp.where(f_x_new < p, x_new, lo)
        hi = jnp.where(f_x_new >= p, x_new, hi)

        return (x_new, lo, hi, f_x_new), None

    def bisection_step(carry, _):
        lo, hi, mid, mid_cdf = carry
        mid = 0.5 * (lo + hi)
        mid_cdf = betainc(a, b, mid)
        lo = jnp.where(mid_cdf < p, mid, lo)
        hi = jnp.where(mid_cdf >= p, mid, hi)
        return (lo, hi, mid, mid_cdf), None

    # Ensure initial bounds bracket the root
    lo = jnp.zeros_like(x_init)
    hi = jnp.ones_like(x_init)

    # Initial function value
    f_x_init = betainc(a, b, x_init)

    # Run Newton's method using scan
    (x, lo, hi, f_x), _ = jax.lax.scan(
        newton_step,
        (x_init, lo, hi, f_x_init),
        None,
        length=max_newton_steps // 2,
        unroll=2,
    )

    # Initial mid and mid_cdf for bisection
    mid = x
    mid_cdf = f_x

    # Run bisection refinement using scan
    (lo, hi, mid, _), _ = jax.lax.scan(
        bisection_step, (lo, hi, mid, mid_cdf), None, length=max_bisect_steps
    )

    # Final Newton refinement
    (x, _, _, _), _ = jax.lax.scan(
        newton_step,
        (mid, lo, hi, mid_cdf),
        None,
        length=max_newton_steps // 2,
        unroll=2,
    )

    return x


# -------------------------------------------------------------------
# Initial Guess Functions
# -------------------------------------------------------------------


def _compute_initial_guess(p, a, b):
    """
    Computes an initial guess for betaincinv using different strategies
    depending on the size of a and b.
    """
    small_ab = (a < 1.0) | (b < 1.0)
    large_ab = (a >= 10.0) & (b >= 10.0)

    return jnp.where(
        small_ab,
        _initial_guess_small_ab(a, b, p),
        jnp.where(
            large_ab,
            _initial_guess_large_ab(a, b, p),
            _initial_guess_balanced(a, b, p),
        ),
    )


# ---------------------------------------------
# Method 1: Small a or b (Edge Cases)
# ---------------------------------------------
def _initial_guess_small_ab(a, b, p):
    """
    Initial guess when a or b is small (e.g., < 1.0).
    """
    ln_p = jnp.log(p + 1e-10)  # Avoid log(0)
    ln_1mp = jnp.log1p(-p)

    return jnp.where(
        p < 0.5,
        jnp.exp(ln_p / a),
        1.0 - jnp.exp(ln_1mp / b),
    )


# ---------------------------------------------
# Method 2: Balanced a ≈ b (Symmetrical Case)
# ---------------------------------------------
def _initial_guess_balanced(a, b, p):
    """
    Initial guess for balanced a and b (when a ≈ b).
    """
    logit_p = jnp.log(p) - jnp.log1p(-p)  # Logit transformation
    correction = (a - b) / (a + b) / 6.0  # Correction term for skew

    return 1 / (1 + jnp.exp(-(logit_p + correction)))


# ---------------------------------------------
# Method 3: Large a, b (Asymptotic Case)
# ---------------------------------------------
def _initial_guess_large_ab(a, b, p):
    """
    Initial guess using asymptotic expansion when a and b are large.
    """
    mu = a / (a + b)
    sigma = jnp.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    # Approximate the quantile using the normal distribution
    z = jax.scipy.stats.norm.ppf(p)
    x_approx = mu + sigma * z

    # Clip to [0, 1] to avoid overflow
    return jnp.clip(x_approx, 1e-10, 1 - 1e-10)


# Estimate the differential entropy of a continuous random variable. -------------------
# Main function to compute differential entropy using various methods


# Main function to compute differential entropy using various methods
def differential_entropy(values, window_length=None, base=None, axis=0, method="auto"):
    """
    Computes the differential entropy of a dataset using one of several methods.

    Args:
        values (array-like): Input array.
        window_length (int, optional): Window length for entropy estimation.
        base (float, optional): Logarithm base for entropy. Defaults to natural log is
            None.
        axis (int, optional): Axis along which to compute entropy.
        method (str, optional): Entropy estimation method ('vasicek', 'van es', 'correa'
            ,'ebrahimi', or 'auto').

    Returns:
        float: Estimated differential entropy.
    """
    values = jnp.asarray(values)
    values = jnp.moveaxis(values, axis, -1)
    n = values.shape[-1]

    if window_length is None:
        window_length = int(jnp.sqrt(n) + 0.5)

    if not (2 <= 2 * window_length < n):
        raise ValueError(
            f"Window length ({window_length}) must be at least 2 and less than half"
            "the sample size ({n})."
        )

    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or None.")

    # Sort the data along the last axis
    sorted_data = jnp.sort(values, axis=-1)

    method = method.lower()
    if method == "auto":
        method = _select_auto_method(n)

    if method not in methods:
        raise ValueError(
            f"`method` must be one of {set(methods.keys())}, but got '{method}'."
        )

    # Compute entropy using the selected method
    entropy = methods[method](sorted_data, window_length, n)

    if base is not None:
        entropy /= jnp.log(base)

    return entropy


def _select_auto_method(n):
    """
    Selects an entropy estimation method based on sample size.
    """
    if n <= 10:
        return "van es"
    elif n <= 1000:
        return "ebrahimi"
    else:
        return "vasicek"


# Helper function to pad data along the last axis
def _pad_along_last_axis(X, m):
    shape = X.shape[:-1] + (m,)
    X_left = jnp.broadcast_to(X[..., :1], shape)
    X_right = jnp.broadcast_to(X[..., -1:], shape)
    return jnp.concatenate([X_left, X, X_right], axis=-1)


# Vasicek entropy estimation
def _vasicek_entropy(X, m, n):
    X = _pad_along_last_axis(X, m)
    differences = X[..., 2 * m :] - X[..., : -2 * m]
    logs = jnp.log(n / (2 * m) * differences)
    return jnp.mean(logs, axis=-1)


# Van Es entropy estimation
def _van_es_entropy(X, m, n):
    differences = X[..., m:] - X[..., :-m]
    term1 = jnp.mean(jnp.log((n + 1) / m * differences), axis=-1)
    harmonic_sum = logsumexp(-jnp.log(jnp.arange(m, n + 1)))
    return term1 + harmonic_sum + jnp.log(m) - jnp.log(n + 1)


# Corrected Ebrahimi entropy estimation
def _ebrahimi_entropy(X, m, n):
    """
    Ebrahimi entropy estimator based on differences between order statistics.
    """
    differences = X[..., 1:] - X[..., :-1]  # Consecutive differences
    ci = jnp.where(
        jnp.arange(1, n) <= m,
        1 + (jnp.arange(1, n) - 1) / m,
        1 + (n - jnp.arange(1, n)) / m,
    )
    logs = jnp.log(n * differences / (ci * m))
    return jnp.mean(logs, axis=-1)


# Correa entropy estimation
def _correa_entropy(X, m, n):
    i = jnp.arange(1, n + 1, dtype=jnp.int32)
    dj = jnp.arange(-m, m + 1)[:, None]
    j = i + dj
    j0 = j + m - 1
    Xibar = jnp.mean(X[..., j0], axis=-2, keepdims=True)
    difference = X[..., j0] - Xibar
    num = jnp.sum(difference * dj, axis=-2)
    den = n * jnp.sum(difference**2, axis=-2)
    return -jnp.mean(jnp.log(num / den), axis=-1)


# Mapping methods to functions
methods = {
    "vasicek": _vasicek_entropy,
    "van es": _van_es_entropy,
    "correa": _correa_entropy,
    "ebrahimi": _ebrahimi_entropy,
}
