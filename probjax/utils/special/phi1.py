"""phi_1(z) = (exp(z) - 1) / z, the exponential-integrator weight."""

import jax.numpy as jnp

from probjax.utils.typing import Array


def phi1_scalar(z: Array) -> Array:
    """Compute phi_1(z) = (exp(z) - 1) / z with numerical stability."""
    small = jnp.abs(z) < 1e-4
    series = 1.0 + 0.5 * z + (z * z) / 6.0 + (z * z * z) / 24.0
    return jnp.where(small, series, jnp.expm1(z) / z)
