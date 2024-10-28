from typing import Callable, NamedTuple, Optional, Tuple

import blackjax
import jax.numpy as jnp
from blackjax.mcmc.hmc import HMCInfo, HMCState
from chex import PRNGKey
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
from jaxtyping import Array, PyTree

from probjax.inference.kernels.base import MarkovKernelAPI


class HMCParams(NamedTuple):
    step_size: ArrayLike
    inverse_mass_matrix: ArrayLike


def init_params(
    position: PyTree,
    step_size: float = 0.5,
    inverse_mass_matrix: Optional[Array] = None,
) -> HMCParams:
    """Initialize the parameters for the HMC kernel.

    Args:
        position (PyTree): Position of the chain.
        step_size (float): Default step size for the HMC kernel. Defaults to 0.5.
        inverse_mass_matrix (Optional[Array], optional): Inverse mass matrix.
            Defaults to None i.e. identity matrix (as diagonal!).

    Raises:
        ValueError: If the dimension of the inverse mass matrix does not match
            the dimension of the position.

    Returns:
        HMCParams: Parameters for the HMC kernel.
    """
    flat_position, _ = ravel_pytree(position)
    dim = flat_position.shape[0]
    if inverse_mass_matrix is None:
        inverse_mass_matrix = jnp.ones((dim,))
    else:
        if inverse_mass_matrix.shap[0] != dim:
            raise ValueError(
                "The dimension of the inverse mass matrix must match the dimension"
                " of the position."
            )
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
    kernel = blackjax.hmc.build_kernel(integrator, divergence_threshold)

    def step(
        key: PRNGKey,
        state: HMCState,
        params: HMCParams,
    ) -> Tuple[HMCState, HMCInfo]:
        return kernel(
            key,
            state,
            logdensity_fn,
            step_size=params.step_size,
            inverse_mass_matrix=params.inverse_mass_matrix,
            num_integration_steps=num_integration_steps,
        )

    return step


def build_hmc_family_adaption(
    algorithm,
    logdensity_fn: Callable,
    integrator: Callable = blackjax.mcmc.integrators.velocity_verlet,
    **kwargs,
):
    """Build parameter adaption method for the HMC kernel."""

    def adapt_params(
        key: PRNGKey,
        position: Array,
        init_params,
        num_steps: int,
        method: str = "window",
        target_acceptance_rate: float = 0.8,
    ):
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

    return adapt_params


class HMC(MarkovKernelAPI):
    init = blackjax.hmc.init
    init_params = init_params
    build_step = build_step
    build_adaptation = lambda *args, **kwargs: build_hmc_family_adaption(
        blackjax.hmc, *args, **kwargs
    )


def build_kernel_nuts(
    logdensity_fn: Callable,
    max_num_doublings: int = 10,
    divergence_threshold: float = 1000.0,
    integrator: Callable = blackjax.mcmc.integrators.velocity_verlet,
):
    kernel = blackjax.nuts.build_kernel(
        divergence_threshold=divergence_threshold, integrator=integrator
    )

    def step(
        key: PRNGKey,
        state: HMCState,
        params: HMCParams,
    ) -> Tuple[HMCState, HMCInfo]:
        return kernel(
            key,
            state,
            logdensity_fn,
            step_size=params.step_size,
            inverse_mass_matrix=params.inverse_mass_matrix,
            max_num_doublings=max_num_doublings,
        )

    return step


class NUTS(HMC):
    build_step = build_kernel_nuts
    build_adaptation = lambda *args, **kwargs: build_hmc_family_adaption(
        blackjax.nuts, *args, **kwargs
    )
