from typing import Callable, NamedTuple, Optional, Tuple

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.adjusted_mclmc_dynamic import trajectory_length
from blackjax.mcmc.dynamic_hmc import DynamicHMCState
from blackjax.mcmc.hmc import HMCInfo, HMCState
from blackjax.mcmc.integrators import IntegratorState
from blackjax.mcmc.mclmc import MCLMCInfo
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
    return MCLMCParams(step_size=step_size, L=L, inverse_mass_matrix=inverse_mass_matrix)


def build_mclmc_step(
    logdensity_fn: Callable,
    integrator: Callable = blackjax.mcmc.integrators.isokinetic_mclachlan,
) -> Callable:
    def step(
        key: RngKey,
        state: IntegratorState,
        params: MCLMCParams,
    ) -> Tuple[IntegratorState, MCLMCInfo]:
        kernel = blackjax.mclmc.build_kernel(
            logdensity_fn,
            inverse_mass_matrix=params.inverse_mass_matrix,
            integrator=integrator,
        )
        return kernel(key, state, params.L, params.step_size)

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
    def step(
        key: RngKey,
        state: HMCState,
        params: AdjustedMCLMCParams,
    ) -> Tuple[HMCState, HMCInfo]:
        params_dict = params._asdict()
        inverse_mass_matrix = params_dict.pop("inverse_mass_matrix")
        kernel = blackjax.adjusted_mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=integrator,
            divergence_threshold=divergence_threshold,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        return kernel(key, state, **params_dict)

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
    def halton_next_random_arg_fn(index: Array):
        return jnp.array(index + 1, dtype=jnp.int32)

    def halton_next_integration_steps_fn(random_arg: Array, **kwargs):
        return trajectory_length(random_arg, average_integration_steps)

    return (
        halton_next_random_arg_fn,
        halton_next_integration_steps_fn,
    )


def random_trajectory_length_fns(average_integration_steps: int):
    def random_next_random_arg_fn(random_arg: Array):
        return jax.random.split(random_arg)[1]

    def random_next_integration_steps_fn(random_arg: Array, **kwargs):
        return jax.random.randint(
            random_arg, shape=(), minval=1, maxval=2 * average_integration_steps
        )

    return (
        random_next_random_arg_fn,
        random_next_integration_steps_fn,
    )


def get_dynamic_stepping(integration_steps_sequence, average_integration_steps):
    if isinstance(integration_steps_sequence, str):
        if integration_steps_sequence == "halton":
            random_arg_next_fn, integration_steps_fn = halton_trajectory_length_fns(
                average_integration_steps
            )
        elif integration_steps_sequence == "random":
            random_arg_next_fn, integration_steps_fn = random_trajectory_length_fns(
                average_integration_steps
            )
        else:
            raise ValueError(
                "integration_steps_sequence must be 'halton', 'random', or a tuple"
            )
    else:
        random_arg_next_fn, integration_steps_fn = integration_steps_sequence

    return random_arg_next_fn, integration_steps_fn


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

    def step(
        key: RngKey,
        state: DynamicHMCState,
        params: AdjustedMCLMCDynamicParams,
    ) -> Tuple[DynamicHMCState, HMCInfo]:
        params_dict = params._asdict()
        inverse_mass_matrix = params_dict.pop("inverse_mass_matrix")
        kernel = blackjax.adjusted_mclmc_dynamic.build_kernel(
            integration_steps_fn=integration_steps_fn,
            integrator=integrator,
            divergence_threshold=divergence_threshold,
            next_random_arg_fn=random_arg_next_fn,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        return kernel(
            key,
            state,
            logdensity_fn=logdensity_fn,
            **params_dict,
        )

    return step


def init_dynamic(
    position,
    logdensity_fn,
    rng_key: RngKey,
    integration_steps_sequence: str = "halton",
):
    if integration_steps_sequence == "random":
        random_generator_arg = rng_key
    else:
        random_generator_arg = jax.random.randint(
            rng_key, shape=(), minval=0, maxval=2**31 - 1, dtype=jnp.int32
        )
    return blackjax.adjusted_mclmc_dynamic.init(
        position, logdensity_fn, random_generator_arg=random_generator_arg
    )


adjusted_mclmc_dynamic = make_kernel_api(
    name="adjusted_mclmc_dynamic",
    init_fn=init_dynamic,
    init_params_fn=init_dynamic_params,
    build_step_fn=build_dynamic_step,
)
