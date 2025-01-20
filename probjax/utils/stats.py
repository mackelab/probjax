import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, logsumexp, digamma, gammaln

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
    x_init = _compute_initial_guess(a_, b_, p_)
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
def _safe_betaincinv_solve(a, b, p, x_init, max_halley_steps=4, max_bisection_steps=12):
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


# Inverse gamma cdf


# -------------------------------------------------------------------
# Core gammaincinv function with reflection and fallback
# -------------------------------------------------------------------


@jax.jit
def gammaincinv(a, p):
    """
    Inverse of the *regularized* lower incomplete gamma function.
    Solves for x >= 0 such that gammainc(a, x) = p.

    Args:
        a (jnp.ndarray): Shape parameter (a > 0).
        p (jnp.ndarray): Probability in [0, 1].

    Returns:
        jnp.ndarray: x in [0, ∞) satisfying gammainc(a, x) = p.
    """

    # Clip p to [0,1]
    p = jnp.clip(p, 0.0, 1.0)

    # Handle trivial cases
    trivial_low = p == 0.0
    trivial_high = p == 1.0
    # By convention: gammainc(a, x=0) = 0 => inverse at p=0 => x=0
    #                gammainc(a, x->∞) = 1 => inverse at p=1 => x=∞
    x_trivial = jnp.where(trivial_low, 0.0, jnp.inf)
    trivial = trivial_low | trivial_high

    # Initial guess
    x_init = _compute_initial_guess_gammaincinv(a, p)
    x_init = jnp.where(trivial, x_trivial, x_init)

    # Refine with safe solver
    x_sol = _safe_gammaincinv_solve(a, p, x_init)

    return jnp.where(trivial, x_trivial, x_sol)


# -------------------------------------------------------------------
# Safe Newton + Bisection Solver
# -------------------------------------------------------------------


def _safe_gammaincinv_solve(a, p, x_init, max_halley_steps=6):
    """
    Safe solver for gammaincinv using a mix of Halley's method
    and bracketed bisection steps.
    """

    def halley_step(carry, _):
        """
        One iteration of Halley's method, with bracket updates
        to ensure we remain in [lo, hi] that brackets p.
        """
        x, lo, hi, f_x = carry
        err = f_x - p

        # Derivative of regularized gammainc(a, x) w.r.t. x:
        #   d/dx [gammainc(a, x)] = x^(a-1) * e^(-x) / Gamma(a)
        deriv = jnp.where(
            x > 0.0,
            jnp.exp((a - 1.0) * jnp.log(x) - x - jax.scipy.special.gammaln(a)),
            0.0,
        )

        # Second derivative of regularized gammainc(a, x) w.r.t. x:
        #   d^2/dx^2 [gammainc(a, x)] = (a-1) * x^(a-2) * e^(-x) / Gamma(a) - x^(a-1) * e^(-x) / Gamma(a)
        second_deriv = jnp.where(
            x > 0.0,
            deriv * ((a - 1.0) / x - 1.0),
            0.0,
        )

        # Halley's update step
        step = err / (deriv + 1e-10)
        correction = 0.5 * step * (second_deriv / (deriv + 1e-10))

        # Propose a new x, clamp to [lo, hi]
        x_new = jnp.clip(
            x - step / (1.0 - correction), jnp.maximum(lo, 0.0), jnp.minimum(hi, 1e10)
        )

        # Evaluate at new x
        f_x_new = jax.scipy.special.gammainc(a, x_new)

        # Update bracket if needed
        lo = jnp.where(f_x_new < p, x_new, lo)
        hi = jnp.where(f_x_new >= p, x_new, hi)

        return (x_new, lo, hi, f_x_new), None

    # Initialize bracket
    lo, hi = _bracket_gamma_inverse(a, p)
    x_init = jnp.clip(x_init, lo, hi)

    # Evaluate at x_init
    f_x_init = jax.scipy.special.gammainc(a, x_init)

    # --- 1) First half of Halley's steps ---
    (x_n, lo, hi, f_n), _ = jax.lax.scan(
        halley_step,
        (x_init, lo, hi, f_x_init),
        None,
        length=max_halley_steps,
        unroll=2,
    )

    return x_n


# -------------------------------------------------------------------
# Initial Guess Function and bracket
# -------------------------------------------------------------------
def _bracket_gamma_inverse(a, p):
    """
    Return lower- and upper-bracket xL, xU such that:
        gammainc(a, xL) <= p*gammainc(a, np.inf)  (i.e. P(a, xL) <= p)
        gammainc(a, xU) >= p*gammainc(a, np.inf)  (i.e. P(a, xU) >= p)
    for a in (0,1).
    """
    # 1) LOWER BOUND
    #    Using incGamma(a, x) ~ x^a / a for small x => P(a, x) ~ x^a / Gamma(a+1).
    #    So xL = ( p * Gamma(a+1) )^(1/a).
    #    Optionally "safety factor" for p near 1:
    gamma_a1 = jax.scipy.special.gamma(a + 1)
    denom = jnp.where(a < 1, a, a - 1)
    x_l = (p * gamma_a1) ** (1 / denom)

    # 2) UPPER BOUND
    #    We can do an asymptotic-based guess: xU = (a - 1) + ln(Gamma(a)/(1-p)).
    #    Then *verify* it actually brackets. If it doesn't, we'll bump it up.
    #    However, if p is extremely small, that log might be large negative. So we
    #    do a piecewise approach:

    # Start with an "asymptotic" guess if p is not too small:
    # We'll clamp p to avoid log(0) or negative arguments.
    p_clip = jnp.maximum(1e-15, jnp.minimum(1.0 - 1e-15, p))
    x_u_guess = (a - 1.0) + jnp.log(jax.scipy.special.gamma(a) / (1.0 - p_clip))

    # If that guess is below x_l, or not finite, pick something bigger than x_l:
    x_u = jnp.maximum(x_u_guess, x_l + 1.0)

    # # If some calls might be vector, we'd do it with a loop or so.
    # # For now assume p is scalar in this snippet:
    y_u = jax.scipy.special.gammainc(a, x_u)
    x_u = jnp.where(y_u < p, x_u * 1.5, x_u)

    return x_l, x_u


def _compute_initial_guess_gammaincinv(a, p):
    """
    Computes an initial guess for x s.t. gammainc(a, x) ~ p.

    Partition the problem into:
      - small a (< 1.0)
      - large a (>= 10.0)
      - balanced region (1 <= a < 10)
    """
    small_a = a < 1.0
    large_a = a >= 20.0
    small_p = p <= a

    x0 = jnp.where(
        small_a,
        _initial_guess_small_a(a, p),
        jnp.where(
            large_a,
            _initial_guess_large_a(a, p),
            _initial_guess_a_ge_1_smaller_30(a, p),
        ),
    )

    x0 = jnp.where(small_p, _initial_approx_small_p(a, p), x0)
    return x0


# ---------------------------------------------
# Method 1: Small a (e.g. a < 1)
# ---------------------------------------------
def _initial_guess_small_a(a, p):
    """
    Improved initial guess for a < 1.

    Near x=0, we can approximate:
      gammainc(a,x) ~ x^a / (a * Gamma(a)).

    => x^a / (a Gamma(a)) = p  =>  x = [a Gamma(a) p]^(1/a)

    For p close to 1, a crude fallback is x ~ -ln(1 - p).
    """
    # For small p
    gamma_a = jnp.exp(jax.scipy.special.gammaln(a))  # = Gamma(a)
    x_small = jnp.power(a * gamma_a * (p + 1e-30), 1.0 / a)

    # For p near 1
    x_big = -jnp.log1p(-p + 1e-30)  # ~ -ln(1 - p)

    # Switch around p=0.5 (you can tweak the threshold)
    return jnp.where(p < 0.5, x_small, x_big)


# ------------------------------------------------------
# Method 2:Small p (e.g. p < 1.)  generally good
# ------------------------------------------------------


def _initial_approx_small_p(a, p):
    """
    Compute the initial approximation of the inverse incomplete gamma function
    for small values of p using the asymptotic series expansion.

    Parameters:
    p : float
        The value of the regularized incomplete gamma function (0 <= p <= 1).
    a : float
        The shape parameter of the gamma function (a > 0).

    Returns:
    x : float
        The initial approximation of the inverse incomplete gamma function.
    """

    # Compute r as the initial approximation
    # Handle small p values separately to avoid numerical instability
    small_p_threshold = 1e-10
    x_small_p = (p * jax.scipy.special.gamma(1 + a)) ** (1 / a)

    # Compute r as the initial approximation
    r = jnp.where(
        p < small_p_threshold,
        x_small_p,
        (p * jax.scipy.special.gamma(1 + a)) ** (1 / a),
    )

    # Coefficients for the series expansion
    c2 = 1.0 / (a + 1.0)
    c3 = (3 * a + 5) / (2 * (a + 1) ** 2 * (a + 2))
    c4 = (8 * a**2 + 33 * a + 31) / (3 * (a + 1) ** 3 * (a + 2) * (a + 3))
    c5 = (125 * a**4 + 1179 * a**3 + 3971 * a**2 + 5661 * a + 2888) / (
        24 * (a + 1) ** 4 * (a + 2) ** 2 * (a + 3) * (a + 4)
    )

    # Initial approximation using the first few terms of the series
    x = r + c2 * r**2 + c3 * r**3 + c4 * r**4 + c5 * r**5

    return x


# ---------------------------------------------
# Method 2: Balanced 1 <= a < 10
# ---------------------------------------------
# ---------------------------------------------
# Method 2: Wilson-Hilferty for a >= 1
# ---------------------------------------------
def _initial_guess_a_ge_1_smaller_30(a, p):
    """
    For a >= 1, use the Wilson–Hilferty transform:
        Z = ( (X/a)^(1/3) - (1 - 2/(9a)) ) / sqrt(2/(9a)) ~ N(0,1)
    =>  X ~ a * [ (1 - 2/(9a)) + sqrt(2/(9a))*Z ]^3
    with Z = Phi^{-1}(p).

    This is often better than the simple mean ± sqrt(a) approach
    for moderate a.
    """
    # Standard normal quantile
    z = jax.scipy.stats.norm.ppf(p)

    alpha = 1.0 - (2.0 / (9.0 * a))
    beta = jnp.sqrt(2.0 / (9.0 * a))

    w = alpha + beta * z
    # Avoid negative inside the cube if p << 0.5, or if a is moderate
    w = jnp.maximum(w, 1e-7)
    x_approx = a * (w**3)

    return jnp.clip(x_approx, 1e-10, a + 6 * jnp.sqrt(a))


# ---------------------------------------------
# Method 3: Large a (Asymptotic / CLT region)
# ---------------------------------------------
def _initial_guess_large_a(a, p):
    """
    For large a, approximate via normal around mean = a, std = sqrt(a).
    x_approx = a + sqrt(a)*z,  where z = Phi^{-1}(p).
    """
    z = jax.scipy.stats.norm.ppf(p)
    x_approx = a + jnp.sqrt(a) * z
    return jnp.clip(x_approx, 1e-20, a + 6 * jnp.sqrt(a))
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


def mutual_information(x, y, method="kraskov", **kwargs):
    """
    Compute the mutual information between two continuous random variables.

    Args:
        x (array-like): First variable.
        y (array-like): Second variable.
        method (str, optional): Estimation method ('knn' or 'kraskov'). Defaults to
            'knn'.

    Returns:
        float: Estimated mutual information.
    """
    x, y = jnp.asarray(x), jnp.asarray(y)
    n = x.shape[0]

    if method == "kraskov":
        return mutual_information_knn_jax(x, y)
    else:
        raise ValueError(f"Unknown method '{method}'.")


# Function to compute the mutual information using k-NN
def mutual_information_knn_jax(x, y, k=3):
    # Reshape if needed
    x = jnp.atleast_2d(x).T if x.ndim == 1 else x
    y = jnp.atleast_2d(y).T if y.ndim == 1 else y

    # Combine x and y into joint space
    xy = jnp.concatenate([x, y], axis=1)

    # Compute pairwise distances in the joint space
    d_xy = jax.vmap(lambda row: jnp.linalg.norm(xy - row, axis=1))(xy)
    kth_distance = jnp.sort(d_xy, axis=1)[:, k]

    # Compute marginal distances
    d_x = jax.vmap(lambda row: jnp.linalg.norm(x - row, axis=1))(x)
    d_y = jax.vmap(lambda row: jnp.linalg.norm(y - row, axis=1))(y)

    # Count neighbors within the k-th distance
    nx = jnp.sum(d_x < kth_distance[:, None], axis=1)
    ny = jnp.sum(d_y < kth_distance[:, None], axis=1)

    # Kraskov's MI estimator
    n = x.shape[0]
    mi = digamma(k) + digamma(n) - (1 / n) * jnp.sum(digamma(nx) + digamma(ny))
    return mi
