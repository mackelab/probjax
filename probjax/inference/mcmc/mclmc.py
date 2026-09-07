from typing import Callable, NamedTuple, Optional, Tuple

import blackjax
import jax.numpy as jnp
from blackjax.mcmc.adjusted_mclmc_dynamic import trajectory_length
from blackjax.mcmc.dynamic_hmc import DynamicHMCState
from blackjax.mcmc.hmc import HMCInfo, HMCState
from blackjax.mcmc.integrators import IntegratorState
from blackjax.mcmc.mclmc import MCLMCInfo

from probjax.inference.mcmc._dynamic_stepping import (
    get_dynamic_stepping as _resolve_dynamic_stepping,
)
from probjax.inference.mcmc._dynamic_stepping import (
    halton_trajectory_length_fns as _halton_fns,
)
from probjax.inference.mcmc._dynamic_stepping import (
    init_dynamic_arg,
)
from probjax.inference.mcmc._dynamic_stepping import (
    random_trajectory_length_fns as _random_fns,
)
from probjax.inference.mcmc.base import make_kernel_api
from probjax.inference.mcmc.hmc import _init_hmc_like_params, _scale_step_size_by_grad
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey


class MCLMCParams(NamedTuple):
    step_size: float
    L: float
    inverse_mass_matrix: ArrayLike


def init_mclmc_params(
    state: PyTree,
    step_size: float = 0.1,
    L: float = 1.0,
    inverse_mass_matrix: Optional[Array] = None,
) -> MCLMCParams:
    inverse_mass_matrix = _init_hmc_like_params(state, inverse_mass_matrix)
    step_size = _scale_step_size_by_grad(state, step_size)
    return MCLMCParams(
        step_size=step_size, L=L, inverse_mass_matrix=inverse_mass_matrix
    )


def build_mclmc_step(
    logdensity_fn: Callable,
    integrator: Callable = blackjax.mcmc.integrators.isokinetic_mclachlan,
) -> Callable:
    # blackjax >= 1.6 takes logdensity_fn and inverse_mass_matrix on the kernel
    # rather than the builder, so the kernel no longer closes over per-step
    # parameters and can be built once here.
    kernel = blackjax.mclmc.build_kernel(integrator=integrator)

    def step(
        key: RngKey,
        state: IntegratorState,
        params: MCLMCParams,
    ) -> Tuple[IntegratorState, MCLMCInfo]:
        return kernel(
            key,
            state,
            logdensity_fn,
            inverse_mass_matrix=params.inverse_mass_matrix,
            L=params.L,
            step_size=params.step_size,
        )

    return step


mclmc = make_kernel_api(
    name="mclmc",
    init_fn=blackjax.mclmc.init,
    init_params_fn=init_mclmc_params,
    build_step_fn=build_mclmc_step,
)


class AdjustedMCLMCParams(NamedTuple):
    step_size: float
    num_integration_steps: int
    inverse_mass_matrix: ArrayLike
    L_proposal_factor: float


def init_params(
    state: PyTree,
    step_size: float = 0.5,
    num_integration_steps: int = 10,
    inverse_mass_matrix: Optional[Array] = None,
    L_proposal_factor: float = jnp.inf,
) -> AdjustedMCLMCParams:
    inverse_mass_matrix = _init_hmc_like_params(state, inverse_mass_matrix)
    step_size = _scale_step_size_by_grad(state, step_size)
    return AdjustedMCLMCParams(
        step_size=step_size,
        num_integration_steps=num_integration_steps,
        inverse_mass_matrix=inverse_mass_matrix,
        L_proposal_factor=L_proposal_factor,
    )


def build_step(
    logdensity_fn: Callable,
    integrator: Callable = blackjax.mcmc.integrators.isokinetic_mclachlan,
    divergence_threshold: float = 1000.0,
) -> Callable:
    kernel = blackjax.adjusted_mclmc.build_kernel(
        integrator=integrator,
        divergence_threshold=divergence_threshold,
    )

    def step(
        key: RngKey,
        state: HMCState,
        params: AdjustedMCLMCParams,
    ) -> Tuple[HMCState, HMCInfo]:
        return kernel(
            key,
            state,
            logdensity_fn,
            step_size=params.step_size,
            # blackjax >= 1.6 passes the step count through a tuple that the
            # kernel unpacks, so that it can be adapted without a rebuild.
            integration_steps_params=(params.num_integration_steps,),
            inverse_mass_matrix=params.inverse_mass_matrix,
            L_proposal_factor=params.L_proposal_factor,
        )

    return step


adjusted_mclmc = make_kernel_api(
    name="adjusted_mclmc",
    init_fn=blackjax.adjusted_mclmc.init,
    init_params_fn=init_params,
    build_step_fn=build_step,
)


class AdjustedMCLMCDynamicParams(NamedTuple):
    step_size: float
    inverse_mass_matrix: ArrayLike
    L_proposal_factor: float


def init_dynamic_params(
    state: PyTree,
    step_size: float = 0.5,
    inverse_mass_matrix: Optional[Array] = None,
    L_proposal_factor: float = jnp.inf,
) -> AdjustedMCLMCDynamicParams:
    inverse_mass_matrix = _init_hmc_like_params(state, inverse_mass_matrix)
    step_size = _scale_step_size_by_grad(state, step_size)
    return AdjustedMCLMCDynamicParams(
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        L_proposal_factor=L_proposal_factor,
    )


def halton_trajectory_length_fns(average_integration_steps: int):
    return _halton_fns(trajectory_length, average_integration_steps)


def random_trajectory_length_fns(average_integration_steps: int):
    return _random_fns(average_integration_steps)


def get_dynamic_stepping(integration_steps_sequence, average_integration_steps):
    return _resolve_dynamic_stepping(
        integration_steps_sequence,
        average_integration_steps,
        length_fn=trajectory_length,
    )


def build_dynamic_step(
    logdensity_fn: Callable,
    average_integration_steps: int = 10,
    integration_steps_sequence: str = "halton",
    divergence_threshold: float = 1000.0,
    integrator: Callable = blackjax.mcmc.integrators.isokinetic_mclachlan,
    next_random_arg_fn: Optional[Callable] = None,
    integration_steps_fn: Optional[Callable] = None,
) -> Callable:
    if integration_steps_fn is None:
        random_arg_next_fn, integration_steps_fn = get_dynamic_stepping(
            integration_steps_sequence, average_integration_steps
        )
    else:
        random_arg_next_fn = next_random_arg_fn

    if random_arg_next_fn is None:
        raise ValueError(
            "next_random_arg_fn must be provided when integration_steps_fn is set"
        )

    kernel = blackjax.adjusted_mclmc_dynamic.build_kernel(
        integration_steps_fn=integration_steps_fn,
        integrator=integrator,
        divergence_threshold=divergence_threshold,
        next_random_arg_fn=random_arg_next_fn,
    )

    def step(
        key: RngKey,
        state: DynamicHMCState,
        params: AdjustedMCLMCDynamicParams,
    ) -> Tuple[DynamicHMCState, HMCInfo]:
        return kernel(
            key,
            state,
            logdensity_fn,
            step_size=params.step_size,
            L_proposal_factor=params.L_proposal_factor,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )

    return step


def init_dynamic(
    position,
    logdensity_fn,
    rng_key: RngKey,
    integration_steps_sequence: str = "halton",
):
    random_generator_arg = init_dynamic_arg(rng_key, integration_steps_sequence)
    return blackjax.adjusted_mclmc_dynamic.init(
        position, logdensity_fn, random_generator_arg=random_generator_arg
    )


adjusted_mclmc_dynamic = make_kernel_api(
    name="adjusted_mclmc_dynamic",
    init_fn=init_dynamic,
    init_params_fn=init_dynamic_params,
    build_step_fn=build_dynamic_step,
)
