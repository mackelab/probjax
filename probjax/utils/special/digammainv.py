from typing import Union

import jax.numpy as jnp
from jax import lax
from jax.scipy.special import digamma, polygamma


def digammainv(
    y: Union[float, jnp.ndarray], maxiter: int = 5, tol: float = 1e-14
) -> Union[float, jnp.ndarray]:
    """
    Inverse of the digamma function using Newton's method with asymptotic approximations.

    Implementation follows the approach described in the literature:
    For Ψ(x) = y, use Newton's method with the update:
    x_new = x_old - (Ψ(x) - y) / Ψ'(x)

    NOTE: Digamm is only invertible for x > 0. and this function assumes that y is in the domain of invertibility.

    Args:
        y: The value to find the inverse digamma for
        maxiter: Maximum number of Newton iterations (5 iterations typically sufficient for 14 digits)
        tol: Tolerance for convergence

    Returns:
        x such that digamma(x) = y
    """

    # Initial guess based on asymptotic formulas (eq. 149 in the paper)
    def initial_guess(y):
        gamma = jnp.euler_gamma  # Euler-Mascheroni constant

        # Use asymptotic approximations
        # For y ≥ -2.22: x ≈ exp(y) + 1/2
        # For y < -2.22: x ≈ -1/(y + γ)
        x_init = jnp.where(y >= -2.22, jnp.exp(y) + 0.5, -1.0 / (y + gamma))

        return x_init

    # Define Newton step function (eq. 146 in the paper)
    def newton_step(x_old):
        # Calculate Ψ(x) - y and Ψ'(x)
        psi_x = digamma(x_old)
        psi_prime_x = polygamma(1, x_old)  # First derivative of digamma

        # Newton update
        x_new = x_old - (psi_x - y) / psi_prime_x

        return x_new

    # Initialize with the asymptotic approximation
    x = initial_guess(y)

    # Run fixed number of Newton iterations using lax.fori_loop
    def body_fun(i, x):
        return newton_step(x)

    # The paper states 5 iterations are sufficient for 14 digits of precision
    x = lax.fori_loop(0, maxiter, body_fun, x)

    return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
