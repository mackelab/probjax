from typing import Dict, Optional

import jax
import jax.numpy as jnp
from blackjax.smc.tuning import from_kernel_info, from_particles


def tune_from_particles(
    params: Dict,
    particles,
    *,
    updates: Optional[Dict[str, str]] = None,
) -> Dict:
    """Tune MCMC parameters from particles.

    updates maps parameter names to: "mass_matrix", "stds", "means".
    """
    if updates is None:
        updates = {
            "inverse_mass_matrix": "inverse_mass_matrix",
            "scale": "stds",
            "mean": "means",
        }
    tuned = dict(params)
    for key, mode in updates.items():
        if key not in tuned:
            continue
        if mode == "inverse_mass_matrix":
            fn = getattr(
                from_particles,
                "inverse_mass_matrix_from_particles",
                getattr(from_particles, "mass_matrix_from_particles", None),
            )
            if fn is not None:
                val = fn(particles)
                val = jnp.atleast_1d(val)
                if val.ndim == 1 and val.shape[0] == 1:
                    val = val[None, ...]
                tuned[key] = val
        elif mode == "stds":
            tuned[key] = jnp.atleast_1d(from_particles.particles_stds(particles))
        elif mode == "means":
            tuned[key] = jnp.atleast_1d(from_particles.particles_means(particles))
    return tuned


def _get_acceptance_rates(info):
    if info is None:
        return None
    if hasattr(info, "acceptance_rate"):
        return info.acceptance_rate
    if hasattr(info, "acceptance_rates"):
        return info.acceptance_rates
    if hasattr(info, "p_accept"):
        return info.p_accept
    return None


def tune_from_kernel_info(
    params: Dict,
    info,
    *,
    target_acceptance_rate: float = 0.234,
    scale_key: str = "step_size",
) -> Dict:
    """Tune MCMC parameters using kernel acceptance rates."""
    rates = _get_acceptance_rates(info)
    if rates is None or scale_key not in params:
        return params
    tuned = dict(params)
    tuned[scale_key] = from_kernel_info.update_scale_from_acceptance_rate(
        jnp.asarray(tuned[scale_key]), jnp.asarray(rates), target_acceptance_rate
    )
    return tuned
