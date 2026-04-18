from typing import Callable, Tuple

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.dynamic_hmc import DynamicHMCState, halton_trajectory_length
from blackjax.mcmc.hmc import HMCInfo
from probjax.utils.typing import Array, RngKey

from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel
from probjax.inference.mcmc.hmc import HMCParams, build_hmc_family_adaption, init_params


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


def build_adaptation(
    logdensity_fn: Callable,
    average_integration_steps: int = 10,
    integration_steps_sequence: str = "halton",
    integrator: Callable = blackjax.mcmc.integrators.velocity_verlet,
) -> Callable:
    """Build parameter adaptation method for the Dynamic HMC kernel."""

    def fit_params(
        key: RngKey,
        state: Array,
        init_params,
        num_steps: int,
        method: str = "window",
        target_acceptance_rate: float = 0.8,
        is_mass_matrix_diagonal: bool = True,
        **_,
    ):
        if method != "window":
            raise ValueError(f"Adaption method {method} not supported for dynamic_hmc")

        position = state.position if hasattr(state, "position") else state

        # Generate random_generator_arg for dynamic_hmc.init
        if integration_steps_sequence == "random":
            random_generator_arg = key
        else:
            random_generator_arg = jax.random.randint(
                key, shape=(), minval=0, maxval=2**31 - 1, dtype=jnp.int32
            )

        # Build the kernel with extra parameters for dynamic_hmc
        random_arg_next_fn, integration_steps_fn = get_dynamic_stepping(
            integration_steps_sequence, average_integration_steps
        )

        mcmc_kernel = blackjax.dynamic_hmc.build_kernel(
            next_random_arg_fn=random_arg_next_fn,
            integration_steps_fn=integration_steps_fn,
            integrator=integrator,
        )

        # Get adaptation functions from blackjax
        adapt_init, adapt_step, adapt_final = (
            blackjax.adaptation.window_adaptation.base(
                is_mass_matrix_diagonal,
                target_acceptance_rate=target_acceptance_rate,
            )
        )

        # Initialize adaptation state
        init_adaptation_state = adapt_init(position, init_params.step_size)

        def one_step(carry, xs):
            _, rng_key, adaptation_stage = xs
            state, adaptation_state = carry

            new_state, info = mcmc_kernel(
                rng_key,
                state,
                logdensity_fn,
                adaptation_state.step_size,
                adaptation_state.inverse_mass_matrix,
            )
            new_adaptation_state = adapt_step(
                adaptation_state,
                adaptation_stage,
                new_state.position,
                info.acceptance_rate,
            )

            return (
                (new_state, new_adaptation_state),
                None,
            )

        # Build the schedule for adaptation stages
        schedule = blackjax.adaptation.window_adaptation.build_schedule(num_steps)

        # Initialize the state with random_generator_arg
        init_state = blackjax.dynamic_hmc.init(
            position, logdensity_fn, random_generator_arg
        )

        # Run adaptation
        start_state = (init_state, init_adaptation_state)
        keys = jax.random.split(key, num_steps)

        last_state, _ = jax.lax.scan(
            one_step,
            start_state,
            (jnp.arange(num_steps), keys, schedule),
        )

        last_chain_state, last_warmup_state, *_ = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        params = HMCParams(
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        return last_chain_state, params

    return fit_params


dynamic_hmc = make_kernel_api(
    name="dynamic_hmc",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=build_dynamic_hmc_step,
    build_adaptation_fn=build_adaptation,
)
