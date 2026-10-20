from __future__ import annotations

from typing import Callable, NamedTuple

import blackjax
from blackjax.mcmc import integrators

from probjax.inference.mcmc.adaptation import window_warmup
from probjax.inference.mcmc.base import (
    MarkovKernelAPI,
    init_hmc_like_params,
    make_step_from_kernel,
    scale_step_size_by_grad,
)
from probjax.utils.typing import Array, ArrayLike, PyTree, Scalar


class HMCParams(NamedTuple):
    step_size: Scalar
    inverse_mass_matrix: ArrayLike


def init_params(
    state: PyTree,
    step_size: float = 0.5,
    inverse_mass_matrix: Array | None = None,
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
    inverse_mass_matrix = init_hmc_like_params(state, inverse_mass_matrix)
    scaled_step_size = scale_step_size_by_grad(state, step_size, inverse_mass_matrix)
    return HMCParams(
        step_size=scaled_step_size,
        inverse_mass_matrix=inverse_mass_matrix,
    )


def build_step(
    logdensity_fn: Callable,
    num_integration_steps: int = 10,
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000.0,
) -> Callable:
    kernel_builder = lambda: blackjax.hmc.build_kernel(integrator, divergence_threshold)
    return make_step_from_kernel(
        logdensity_fn,
        kernel_builder,
        call_defaults={"num_integration_steps": num_integration_steps},
    )


class hmc(MarkovKernelAPI):
    """Hamiltonian Monte Carlo with a fixed number of integration steps.

    Tunable parameters come from ``kernel.init_params(state, step_size=0.5,
    inverse_mass_matrix=None)`` (see :class:`HMCParams`); the step size is
    scaled by the initial gradient norm. Default adaptation:
    :func:`window_warmup`.

    Args:
        logdensity_fn: Log density of the target.
        num_integration_steps: Number of integrator steps per proposal.
        integrator: Symplectic integrator from ``blackjax.mcmc.integrators``.
        divergence_threshold: Energy error above which a transition is
            flagged as divergent.
    """

    init = staticmethod(blackjax.hmc.init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_step)
    default_adaptation_fn = staticmethod(window_warmup)


def build_kernel_nuts(
    logdensity_fn: Callable,
    max_num_doublings: int = 10,
    divergence_threshold: float = 1000.0,
    integrator: Callable = integrators.velocity_verlet,
) -> Callable:
    kernel_builder = lambda: blackjax.nuts.build_kernel(
        divergence_threshold=divergence_threshold, integrator=integrator
    )
    return make_step_from_kernel(
        logdensity_fn,
        kernel_builder,
        call_defaults={"max_num_doublings": max_num_doublings},
    )


class nuts(MarkovKernelAPI):
    """No-U-Turn sampler: HMC with an adaptively chosen trajectory length.

    Tunable parameters are the same as for :class:`hmc`
    (``step_size``, ``inverse_mass_matrix``). Default adaptation:
    :func:`window_warmup`.

    Args:
        logdensity_fn: Log density of the target.
        max_num_doublings: Maximum tree depth; a trajectory has at most
            ``2**max_num_doublings`` integrator steps.
        divergence_threshold: Energy error above which a transition is
            flagged as divergent.
        integrator: Symplectic integrator from ``blackjax.mcmc.integrators``.
    """

    init = staticmethod(blackjax.hmc.init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_kernel_nuts)
    default_adaptation_fn = staticmethod(window_warmup)
