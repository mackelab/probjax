from typing import Callable, Tuple

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.dynamic_hmc import DynamicHMCState, halton_trajectory_length
from blackjax.mcmc.hmc import HMCInfo
from chex import PRNGKey
from jaxtyping import Array

from probjax.inference.kernels.hmc import (
    HMC,
    HMCParams,
    build_hmc_family_adaption,
    init_params,
)


def halton_trajectory_length_fns(average_trajectory_length: float):
    def halton_next_random_arg_fn(index: Array):
        return jnp.array(index + 1, dtype=jnp.int32)

    def halton_next_integration_steps_fn(random_arg: Array, **kwargs):
        return halton_trajectory_length(random_arg, average_trajectory_length)

    return (
        halton_next_random_arg_fn,
        halton_next_integration_steps_fn,
    )


def random_trajectory_length_fns(average_trajectory_length: int):
    def random_next_random_arg_fn(random_arg: Array):
        return jax.random.split(random_arg)[1]

    def random_next_integration_steps_fn(random_arg: Array, **kwargs):
        return jax.random.randint(
            random_arg, shape=(), minval=1, maxval=2 * average_trajectory_length
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
        random_arg_next_fn, integration_steps_fn = integration_steps_sequence

    return random_arg_next_fn, integration_steps_fn


def build_dynamic_hmc_step(
    logdensity_fn: Callable,
    average_integration_steps: int = 10,
    integration_steps_sequence: str = "halton",
    divergence_threshold: float = 1000.0,
    integrator: Callable = blackjax.mcmc.integrators.velocity_verlet,
) -> Callable:
    random_arg_next_fn, integration_steps_fn = get_dynamic_stepping(
        integration_steps_sequence, average_integration_steps
    )

    kernel = blackjax.dynamic_hmc.build_kernel(
        next_random_arg_fn=random_arg_next_fn,
        integration_steps_fn=integration_steps_fn,
        integrator=integrator,
        divergence_threshold=divergence_threshold,
    )

    def step(
        key: PRNGKey,
        state: DynamicHMCState,
        params: HMCParams,
    ) -> Tuple[DynamicHMCState, HMCInfo]:
        return kernel(
            key,
            state,
            logdensity_fn,
            step_size=params.step_size,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )

    return step


def build_adaption(
    logdensity_fn: Callable,
    average_integration_steps: int = 10,
    integration_steps_sequence: str = "halton",
    integrator: Callable = blackjax.mcmc.integrators.velocity_verlet,
) -> Callable:
    random_arg_next_fn, integration_steps_fn = get_dynamic_stepping(
        integration_steps_sequence, average_integration_steps
    )

    _object = blackjax.dynamic_hmc.copy()
    _object.build_kernel = lambda integrator: blackjax.dynamic_hmc.build_kernel(
        next_random_arg_fn=random_arg_next_fn,
        integration_steps_fn=integration_steps_fn,
        integrator=integrator,
    )

    return build_hmc_family_adaption(
        _object,
        logdensity_fn,
        average_integration_steps=average_integration_steps,
        next_random_arg_fn=random_arg_next_fn,
        integration_steps_fn=integration_steps_fn,
        integrator=integrator,
    )


class dHMC(HMC):
    init = blackjax.dynamic_hmc.init
    init_params = init_params
    build_step = build_dynamic_hmc_step
    build_adaption = build_adaption
