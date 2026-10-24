from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

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
from probjax.inference.base import Adaptor, Kernel, RunResult, Warmup
from probjax.inference.mcmc.base import ravel_position


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

    def init(_state: Any, params: Any) -> Any:
        return init_da(get_param(params, "step_size"))

    def update(
        _state: Any, info: Any, adaptor_state: Any, params: Any
    ) -> tuple[Any, Any, None]:
        adaptor_state = update_da(adaptor_state, target_from_info_fn(info))
        params = replace_params(params, step_size=jnp.exp(adaptor_state.log_step_size))
        return adaptor_state, params, None

    def finalize(adaptor_state: Any, params: Any) -> tuple[Any, None]:
        return replace_params(params, step_size=final_da(adaptor_state)), None

    return Adaptor(init, update, finalize)


def mass_matrix_adaptor(*, diagonal: bool = True) -> Adaptor:
    """Local mass-matrix estimator backed by BlackJAX Welford adaptation."""
    init_mm, update_mm, final_mm = mass_matrix_adaptation(diagonal)

    def init(state: Any, params: Any) -> Any:
        position = ravel_position(state)
        get_param(params, "inverse_mass_matrix")
        return init_mm(position.size)

    def update(
        state: Any, _info: Any, adaptor_state: Any, params: Any
    ) -> tuple[Any, Any, None]:
        return update_mm(adaptor_state, state.position), params, None

    def finalize(adaptor_state: Any, params: Any) -> tuple[Any, None]:
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

    def init(state: Any, params: Any) -> CovarianceAdaptorState:
        position = ravel_position(state)
        get_param(params, field)
        return CovarianceAdaptorState(init_cov(position.size))

    def update(
        state: Any, _info: Any, adaptor_state: CovarianceAdaptorState, params: Any
    ) -> tuple[CovarianceAdaptorState, Any, None]:
        position = ravel_position(state)
        state_out = update_cov(adaptor_state.welford_state, position)
        return CovarianceAdaptorState(state_out), params, None

    def finalize(
        adaptor_state: CovarianceAdaptorState, params: Any
    ) -> tuple[Any, None]:
        covariance, _, _ = final_cov(adaptor_state.welford_state)
        if output == "standard_deviation":
            value = jnp.sqrt(covariance)
        elif output == "cholesky":
            value = jnp.linalg.cholesky(covariance)
        else:
            value = covariance
        return replace_params(params, **{field: value}), None

    return Adaptor(init, update, finalize)


def mean_adaptor(
    *,
    field: str = "mean",
    diagonal: bool = True,
) -> Adaptor:
    """Local proposal-mean estimator backed by BlackJAX Welford updates."""
    init_mean, update_mean, final_mean = welford_algorithm(diagonal)

    def init(state: Any, params: Any) -> CovarianceAdaptorState:
        position = ravel_position(state)
        get_param(params, field)
        return CovarianceAdaptorState(init_mean(position.size))

    def update(
        state: Any, _info: Any, adaptor_state: CovarianceAdaptorState, params: Any
    ) -> tuple[CovarianceAdaptorState, Any, None]:
        position = ravel_position(state)
        state_out = update_mean(adaptor_state.welford_state, position)
        return CovarianceAdaptorState(state_out), params, None

    def finalize(
        adaptor_state: CovarianceAdaptorState, params: Any
    ) -> tuple[Any, None]:
        _, _, mean = final_mean(adaptor_state.welford_state)
        return replace_params(params, **{field: mean}), None

    return Adaptor(init, update, finalize)


def slice_step_size_adaptor(target: float = 0.5, **kwargs: Any) -> Adaptor:
    """Dual averaging of the slice width on the share of bracket expansions.

    A step size that is too small spends its evaluations stepping out, one
    that is too large spends them shrinking the bracket. The expanding share
    falls monotonically with the step size, so driving it to ``target``
    (expansions as many as contractions by default, as in Karamanis & Beutler,
    2021) is stable, unlike targeting the total number of evaluations.
    """
    return step_size_adaptor(
        target=target,
        target_from_info_fn=lambda info: info.num_expansions / info.num_evals,
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
    import warnings

    # `window_adaptation.base` is deprecated upstream in favor of the bound
    # `window_adaptation`/`staged_adaptation` algorithms, but those don't expose
    # the init/update/final triple this Warmup protocol needs. Keep using the
    # base primitives and silence the deprecation until the protocol migrates
    # to the staged engine.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*window_adaptation.base.*", category=DeprecationWarning
        )
        init_window, update_window, final_window = window_adaptation_base(
            diagonal, target_acceptance_rate
        )

    @partial(jax.jit, static_argnames=("kernel", "num_steps"))
    def run(rng, kernel: Kernel, state: Any, params: Any, num_steps: int) -> RunResult:
        schedule = build_schedule(
            num_steps,
            initial_buffer_size=initial_buffer_size,
            final_buffer_size=final_buffer_size,
            first_window_size=first_window_size,
        )
        get_param(params, "inverse_mass_matrix")
        warmup_state = init_window(state.position, get_param(params, "step_size"))
        keys = jax.random.split(rng, num_steps)

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
        return RunResult(state, params)

    return Warmup(run)
