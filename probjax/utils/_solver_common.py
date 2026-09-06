"""Shared scaffolding for ODE/SDE solver implementations.

Both ``odeutil`` and ``sdeutil`` solvers repeat the same small patterns
(state init, Wiener sampling, grid integration, method registries). They
live here so the two families cannot drift apart. Import paths of the
existing ``odeutil.solvers.base`` / ``sdeutil.base`` modules are unchanged.
"""

import jax
import jax.numpy as jnp


def make_trivial_init(state_cls):
    """State init for solvers whose state is just ``(t0, y0)``."""

    def init(t0, y0, *args, **kwargs):
        del args, kwargs
        return state_cls(t0=jnp.asarray(t0), y0=jnp.asarray(y0))

    return init


def sample_wiener_increment(rng, g0, default_dim, dt, configured_dim=None):
    """Sample a Wiener increment, inferring the noise dimension from ``g0``."""
    from probjax.utils.sdeutil.base import infer_noise_dim

    dim = infer_noise_dim(g0, default_dim, configured_dim)
    return jax.random.normal(rng, (dim,)) * jnp.sqrt(jnp.abs(dt))
