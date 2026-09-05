from typing import Callable

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.dynamic_hmc import halton_trajectory_length

from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel
from probjax.inference.mcmc.hmc import init_params
from probjax.utils.typing import Array, RngKey


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


def init(
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
