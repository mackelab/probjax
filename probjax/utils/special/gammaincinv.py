import jax
import jax.numpy as jnp

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
