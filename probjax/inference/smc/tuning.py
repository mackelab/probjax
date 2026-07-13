from typing import Dict, Optional

import jax.numpy as jnp
from blackjax.smc.tuning import from_kernel_info, from_particles

from probjax.inference.base import Adaptor
from probjax.inference.smc.base import _ensure_param_batch, _params_to_dict


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
                old = jnp.asarray(tuned[key])
                val = fn(particles)
                val = jnp.atleast_1d(val)
                # Preserve the original shape: if input was diagonal
                # (e.g. [1, d]), keep diagonal; if full ([1, d, d]), keep full.
                if old.ndim == 2 and val.ndim == 2:
                    # old is [batch, d] (diagonal), val is [d, d] (full)
                    val = jnp.diag(val)
                    if old.shape[0] == 1:
                        val = val[None, ...]
                else:
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
    scale = jnp.asarray(tuned[scale_key])
    rates = jnp.asarray(rates)
    if scale.size == 1:
        rates = jnp.mean(rates)
    tuned[scale_key] = from_kernel_info.update_scale_from_acceptance_rate(
        scale, rates, target_acceptance_rate
    )
    return tuned


def particle_adaptor(*, updates: Optional[Dict[str, str]] = None) -> Adaptor:
    """Update MCMC geometry from the current SMC particle population."""

    def init(_state, _params):
        return ()

    def update(state, _info, adaptor_state, params):
        params = tune_from_particles(
            _params_to_dict(params), state.particles, updates=updates
        )
        return adaptor_state, _ensure_param_batch(params, shared=True), None

    def finalize(_adaptor_state, params):
        return params, None

    return Adaptor(init, update, finalize)


def acceptance_rate_adaptor(
    *,
    target_acceptance_rate: float = 0.234,
    scale_key: str = "step_size",
    info_fn=lambda info: info.update_info,
) -> Adaptor:
    """Update an SMC move-kernel scale from its acceptance diagnostics."""

    def init(_state, _params):
        return ()

    def update(_state, info, adaptor_state, params):
        params = tune_from_kernel_info(
            _params_to_dict(params),
            info_fn(info),
            target_acceptance_rate=target_acceptance_rate,
            scale_key=scale_key,
        )
        return adaptor_state, _ensure_param_batch(params, shared=True), None

    def finalize(_adaptor_state, params):
        return params, None

    return Adaptor(init, update, finalize)
