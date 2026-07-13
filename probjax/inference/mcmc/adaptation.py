from typing import NamedTuple

import jax
import jax.numpy as jnp
from blackjax.adaptation.mass_matrix import (
    mass_matrix_adaptation,
    welford_algorithm,
)
from blackjax.adaptation.step_size import dual_averaging_adaptation
from blackjax.adaptation.window_adaptation import base as window_adaptation_base
from blackjax.adaptation.window_adaptation import build_schedule

from probjax.inference.adaptation import get_param, replace_params
from probjax.inference.base import Adaptor, Warmup, WarmupResult


def step_size_adaptor(
    target: float = 0.8,
    *,
    target_from_info_fn=lambda info: info.acceptance_rate,
    t0: int = 10,
    gamma: float = 0.05,
    kappa: float = 0.75,
) -> Adaptor:
    """Local dual-averaging rule backed by BlackJAX's adaptation primitive."""
    init_da, update_da, final_da = dual_averaging_adaptation(
        target, t0=t0, gamma=gamma, kappa=kappa
    )

    def init(_state, params):
        return init_da(get_param(params, "step_size"))

    def update(_state, info, adaptor_state, params):
        adaptor_state = update_da(adaptor_state, target_from_info_fn(info))
        params = replace_params(params, step_size=jnp.exp(adaptor_state.log_step_size))
        return adaptor_state, params, None

    def finalize(adaptor_state, params):
        return replace_params(params, step_size=final_da(adaptor_state)), None

    return Adaptor(init, update, finalize)


def mass_matrix_adaptor(*, diagonal: bool = True) -> Adaptor:
    """Local mass-matrix estimator backed by BlackJAX Welford adaptation."""
    init_mm, update_mm, final_mm = mass_matrix_adaptation(diagonal)

    def init(state, params):
        position, _ = jax.flatten_util.ravel_pytree(state.position)
        get_param(params, "inverse_mass_matrix")
        return init_mm(position.size)

    def update(state, _info, adaptor_state, params):
        return update_mm(adaptor_state, state.position), params, None

    def finalize(adaptor_state, params):
        adaptor_state = final_mm(adaptor_state)
        return replace_params(
            params, inverse_mass_matrix=adaptor_state.inverse_mass_matrix
        ), None

    return Adaptor(init, update, finalize)


class CovarianceAdaptorState(NamedTuple):
    welford_state: object


def covariance_adaptor(
    *,
    field: str = "scale",
    diagonal: bool = True,
    output: str = "standard_deviation",
) -> Adaptor:
    """Local proposal-geometry estimator backed by BlackJAX Welford updates."""
    init_cov, update_cov, final_cov = welford_algorithm(diagonal)
    if output not in {"variance", "covariance", "standard_deviation", "cholesky"}:
        raise ValueError(f"Unsupported covariance output: {output}")

    def init(state, params):
        position, _ = jax.flatten_util.ravel_pytree(state.position)
        get_param(params, field)
        return CovarianceAdaptorState(init_cov(position.size))

    def update(state, _info, adaptor_state, params):
        position, _ = jax.flatten_util.ravel_pytree(state.position)
        state_out = update_cov(adaptor_state.welford_state, position)
        return CovarianceAdaptorState(state_out), params, None

    def finalize(adaptor_state, params):
        covariance, _, _ = final_cov(adaptor_state.welford_state)
        if output == "standard_deviation":
            value = jnp.sqrt(covariance)
        elif output == "cholesky":
            value = jnp.linalg.cholesky(covariance)
        else:
            value = covariance
        return replace_params(params, **{field: value}), None

    return Adaptor(init, update, finalize)


def slice_step_size_adaptor(
    target_num_evaluations: float,
    max_evaluations: int,
    **kwargs,
) -> Adaptor:
    """Local dual-averaging rule for slice-width adaptation."""
    return step_size_adaptor(
        target=target_num_evaluations / max_evaluations,
        target_from_info_fn=lambda info: info.num_evals / max_evaluations,
        **kwargs,
    )


def window_warmup(
    *,
    diagonal: bool = True,
    target_acceptance_rate: float = 0.8,
    initial_buffer_size: int = 75,
    final_buffer_size: int = 50,
    first_window_size: int = 25,
) -> Warmup:
    """Finite Stan-style warmup using BlackJAX's window-adaptation base."""
    init_window, update_window, final_window = window_adaptation_base(
        diagonal, target_acceptance_rate
    )

    def run(key, kernel, state, params, num_steps, args=None):
        if args is not None:
            raise ValueError("window_warmup does not support per-step arguments")
        schedule = build_schedule(
            num_steps,
            initial_buffer_size=initial_buffer_size,
            final_buffer_size=final_buffer_size,
            first_window_size=first_window_size,
        )
        get_param(params, "inverse_mass_matrix")
        warmup_state = init_window(state.position, get_param(params, "step_size"))
        keys = jax.random.split(key, num_steps)

        def one_step(carry, xs):
            state, params, warmup_state = carry
            step_key, stage = xs
            state, info = kernel.step(step_key, state, params)
            warmup_state = update_window(
                warmup_state, stage, state.position, info.acceptance_rate
            )
            params = replace_params(
                params,
                step_size=warmup_state.step_size,
                inverse_mass_matrix=warmup_state.inverse_mass_matrix,
            )
            return (state, params, warmup_state), None

        (state, params, warmup_state), _ = jax.lax.scan(
            one_step, (state, params, warmup_state), (keys, schedule)
        )
        step_size, inverse_mass_matrix = final_window(warmup_state)
        params = replace_params(
            params,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        return WarmupResult(state, params)

    return Warmup(run)
