from typing import Callable, NamedTuple, Optional, Tuple

import blackjax
import jax.numpy as jnp
from blackjax.mcmc.hmc import HMCInfo, HMCState
from jax.flatten_util import ravel_pytree
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey

from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel


class HMCParams(NamedTuple):
    step_size: ArrayLike
    inverse_mass_matrix: ArrayLike


def _init_hmc_like_params(state: PyTree, inverse_mass_matrix: Optional[Array]) -> Array:
    position = state.position if hasattr(state, "position") else state
    flat_position, _ = ravel_pytree(position)
    dim = flat_position.shape[0]
    if inverse_mass_matrix is None:
        return jnp.ones((dim,))
    if inverse_mass_matrix.shape[0] != dim:
        raise ValueError(
            "The dimension of the inverse mass matrix must match the dimension"
            " of the position."
        )
    return inverse_mass_matrix


def _scale_step_size_by_grad(
    state: PyTree, step_size: float, inverse_mass_matrix: Optional[Array] = None
) -> ArrayLike:
    grad = getattr(state, "logdensity_grad", None)
    if grad is None:
        return step_size
    flat_grad, _ = ravel_pytree(grad)
    if inverse_mass_matrix is not None:
        inv = inverse_mass_matrix
        if inv.ndim == 0:
            grad_norm = jnp.sqrt(inv) * jnp.linalg.norm(flat_grad)
        elif inv.ndim == 1:
            grad_norm = jnp.sqrt(jnp.sum(inv * flat_grad**2))
        elif inv.ndim == 2:
            grad_norm = jnp.sqrt(flat_grad @ (inv @ flat_grad))
        else:
            raise ValueError("Invalid inverse_mass_matrix shape")
    else:
        grad_norm = jnp.linalg.norm(flat_grad)
    return step_size / jnp.where(grad_norm == 0, 1.0, grad_norm)


def init_params(
    state: PyTree,
    step_size: float = 0.5,
    inverse_mass_matrix: Optional[Array] = None,
) -> HMCParams:
    """Initialize the parameters for the HMC kernel.

    Args:
        state (PyTree): Position of the chain.
        step_size (float): Default step size for the HMC kernel. Defaults to 0.5.
        inverse_mass_matrix (Optional[Array], optional): Inverse mass matrix.
            Defaults to None i.e. identity matrix (as diagonal!).

    Raises:
        ValueError: If the dimension of the inverse mass matrix does not match
            the dimension of the position.

    Returns:
        HMCParams: Parameters for the HMC kernel.
    """
    inverse_mass_matrix = _init_hmc_like_params(state, inverse_mass_matrix)
    step_size = _scale_step_size_by_grad(state, step_size, inverse_mass_matrix)
    return HMCParams(
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
    )


def build_step(
    logdensity_fn: Callable,
    num_integration_steps: int = 10,
    integrator: Callable = blackjax.mcmc.integrators.velocity_verlet,
    divergence_threshold: float = 1000.0,
):
    """Build the HMC kernel.

    Args:
        logdensity_fn (Callable): The log density function.
        num_integration_steps (int, optional): Number of integration steps.
            Defaults to 10.
        integrator (Callable, optional): The integrator to use. Defaults to
            blackjax.integrators.velocity_verlet.
        divergence_threshold (float, optional): The threshold for the divergence
            check. Defaults to 1000.0.

    """
    kernel_builder = lambda: blackjax.hmc.build_kernel(integrator, divergence_threshold)
    return make_step_from_kernel(
        logdensity_fn,
        kernel_builder,
        call_defaults={"num_integration_steps": num_integration_steps},
    )


def build_hmc_family_adaption(
    algorithm,
    logdensity_fn: Callable,
    integrator: Callable = blackjax.mcmc.integrators.velocity_verlet,
    **kwargs,
):
    """Build parameter adaption method for the HMC kernel."""

    def fit_params(
        key: RngKey,
        state: Array,
        init_params,
        num_steps: int,
        method: str = "window",
        target_acceptance_rate: float = 0.8,
        **_,
    ):
        position = state.position if hasattr(state, "position") else state
        if method == "window":
            adaption_alg = blackjax.window_adaptation(
                algorithm,
                logdensity_fn,
                initial_step_size=init_params.step_size,
                target_acceptance_rate=target_acceptance_rate,
                adaptation_info_fn=lambda *args, **kwargs: None,
                integrator=integrator,
                **kwargs,
            )
        elif method == "pathfinder":
            adaption_alg = blackjax.pathfinder_adaptation(
                algorithm,
                logdensity_fn,
                initial_step_size=init_params.step_size,
                target_acceptance_rate=target_acceptance_rate,
                adaptation_info_fn=lambda *args, **kwargs: None,
                integrator=integrator,
                **kwargs,
            )
        else:
            raise ValueError(f"Adaption method {method} not supported")

        adaption_state, _ = adaption_alg.run(key, position, num_steps)
        state = adaption_state.state
        params = HMCParams(
            step_size=adaption_state.parameters["step_size"],
            inverse_mass_matrix=adaption_state.parameters["inverse_mass_matrix"],
        )
        return state, params

    return fit_params


hmc = make_kernel_api(
    name="hmc",
    init_fn=blackjax.hmc.init,
    init_params_fn=init_params,
    build_step_fn=build_step,
    build_adaptation_fn=lambda *args, **kwargs: build_hmc_family_adaption(
        blackjax.hmc, *args, **kwargs
    ),
)


def build_kernel_nuts(
    logdensity_fn: Callable,
    max_num_doublings: int = 10,
    divergence_threshold: float = 1000.0,
    integrator: Callable = blackjax.mcmc.integrators.velocity_verlet,
):
    kernel_builder = lambda: blackjax.nuts.build_kernel(
        divergence_threshold=divergence_threshold, integrator=integrator
    )
    return make_step_from_kernel(
        logdensity_fn,
        kernel_builder,
        call_defaults={"max_num_doublings": max_num_doublings},
    )


nuts = make_kernel_api(
    name="nuts",
    init_fn=blackjax.hmc.init,
    init_params_fn=init_params,
    build_step_fn=build_kernel_nuts,
    build_adaptation_fn=lambda *args, **kwargs: build_hmc_family_adaption(
        blackjax.nuts, *args, **kwargs
    ),
)
