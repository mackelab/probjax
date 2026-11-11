"""SMC effective sample size, vendored from unreleased BlackJAX main.

The ``ess``/``log_ess`` of ``blackjax.smc.ess`` with the numerical fixes from
``6482e5e4`` (subtract the max log-weight before the log-sum-exp difference;
ESS is invariant to a common offset, without this large negative offsets
underflow and corrupt the result) and ``e06d8ec8`` (stop the gradient through
that max-subtraction so differentiating ESS stays correct).

Remove this module and re-export from BlackJAX once the pin reaches a release
containing those fixes.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from probjax.utils.typing import Array

__all__ = ["ess", "log_ess"]


def ess(log_weights: Array) -> Array:
    """Compute the effective sample size.

    Parameters
    ----------
    log_weights: Array
        Log-weights of the sample, shape (n_particles,).

    Returns
    -------
    ess: float | Array
        The effective sample size.
    """
    return jnp.exp(log_ess(log_weights))


def log_ess(log_weights: Array) -> Array:
    """Compute the logarithm of the effective sample size.

    Parameters
    ----------
    log_weights: Array
        Log-weights of the sample, shape (n_particles,).

    Returns
    -------
    log_ess: float | Array
        The logarithm of the effective sample size.
    """
    # ESS is invariant to a common offset; remove it before subtracting log sums.
    # When every weight is -inf the max is -inf and the difference is NaN, so
    # clamp the offset with nansum-safe masking: no mass means ESS is zero.
    max_log_weight = jax.lax.stop_gradient(jnp.max(log_weights))
    finite = jnp.isfinite(max_log_weight)
    offset = jnp.where(finite, max_log_weight, 0.0)
    log_weights = log_weights - offset
    log_ess_value = 2 * logsumexp(log_weights) - logsumexp(2 * log_weights)
    return jnp.where(finite, log_ess_value, -jnp.inf)
