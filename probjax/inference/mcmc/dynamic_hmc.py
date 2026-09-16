from typing import Callable

import blackjax
from blackjax.mcmc.dynamic_hmc import halton_trajectory_length

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
from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel
from probjax.inference.mcmc.hmc import init_params
from probjax.utils.typing import RngKey


def halton_trajectory_length_fns(average_trajectory_length: float):
    return _halton_fns(halton_trajectory_length, average_trajectory_length)


def random_trajectory_length_fns(average_trajectory_length: int):
    return _random_fns(average_trajectory_length)


def get_dynamic_stepping(integration_steps_sequence, average_integration_steps):
    return _resolve_dynamic_stepping(
        integration_steps_sequence,
        average_integration_steps,
        length_fn=halton_trajectory_length,
    )


def init(
    position,
    logdensity_fn,
    rng_key: RngKey,
    integration_steps_sequence: str = "halton",
):
    random_generator_arg = init_dynamic_arg(rng_key, integration_steps_sequence)
    return blackjax.dynamic_hmc.init(position, logdensity_fn, random_generator_arg)


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

    kernel_builder = lambda: blackjax.dynamic_hmc.build_kernel(
        next_random_arg_fn=random_arg_next_fn,
        integration_steps_fn=integration_steps_fn,
        integrator=integrator,
        divergence_threshold=divergence_threshold,
    )
    return make_step_from_kernel(
        logdensity_fn,
        kernel_builder,
    )


dynamic_hmc = make_kernel_api(
    name="dynamic_hmc",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=build_dynamic_hmc_step,
)
