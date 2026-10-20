from __future__ import annotations

from typing import Callable

import blackjax
from blackjax.mcmc import integrators
from blackjax.mcmc.dynamic_hmc import halton_trajectory_length

from probjax.inference.mcmc._dynamic_stepping import (
    get_dynamic_stepping,
    init_dynamic_arg,
)
from probjax.inference.mcmc.adaptation import window_warmup
from probjax.inference.mcmc.base import MarkovKernelAPI, make_step_from_kernel
from probjax.inference.mcmc.hmc import init_params
from probjax.utils.typing import RngKey


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
    integrator: Callable = integrators.velocity_verlet,
) -> Callable:
    random_arg_next_fn, integration_steps_fn = get_dynamic_stepping(
        integration_steps_sequence,
        average_integration_steps,
        length_fn=halton_trajectory_length,
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


class dynamic_hmc(MarkovKernelAPI):
    """HMC whose number of integration steps changes from step to step.

    The step count follows a deterministic or random sequence with the given
    average, which avoids the periodicity of fixed-length HMC. Tunable
    parameters are the same as for :class:`hmc`. Default adaptation:
    :func:`window_warmup`.

    Args:
        logdensity_fn: Log density of the target.
        average_integration_steps: Mean number of integrator steps.
        integration_steps_sequence: How the step counts are drawn:
            ``"halton"`` (low-discrepancy sequence), ``"random"`` (uniform in
            ``[1, 2 * average)``) or a ``(next_random_arg_fn,
            integration_steps_fn)`` pair.
        divergence_threshold: Energy error above which a transition is
            flagged as divergent.
        integrator: Symplectic integrator from ``blackjax.mcmc.integrators``.
    """

    init = staticmethod(init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_dynamic_hmc_step)
    default_adaptation_fn = staticmethod(window_warmup)
