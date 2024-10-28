from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from blackjax.adaptation.base import AdaptationResults
from blackjax.adaptation.step_size import dual_averaging_adaptation
from blackjax.base import AdaptationAlgorithm
from jaxtyping import Array

from probjax.utils.linalg import cholesky_update

# window_adaptation -> HMC, NUTs single chain
# chees_adaptation -> dynamic HMC, NUTs single chain
# meads_adaptation -> GHMC
# pathfinder_adaptation -> HMC, NUTs single chain


def step_size_adaption(
    algorithm,
    logdensity_fn: Callable,
    params,
    target: float,
    adaptation_info_fn=lambda *args, **kwargs: None,
    target_from_info_fn=lambda info: info.acceptance_rate,
    t0=10,
    gamma=0.05,
    kappa=0.75,
    init_kwargs: Optional[dict] = None,
    algorithm_kwargs: Optional[dict] = None,
):
    if not hasattr(params, "step_size"):
        raise ValueError("The params object must have a step_size attribute")

    mcmc_step = algorithm(logdensity_fn, **algorithm_kwargs if algorithm_kwargs else {})
    adapt_init, adapt_step, adapt_final = dual_averaging_adaptation(
        target,
        t0=t0,
        gamma=gamma,
        kappa=kappa,
    )

    def one_step(carry, key):
        state, adaption_state = carry
        step_size = jnp.exp(adaption_state.log_step_size)
        new_params = params._replace(step_size=step_size)

        state, info = mcmc_step(
            key,
            state,
            new_params,
        )
        target = target_from_info_fn(info)
        adaption_state = adapt_step(adaption_state, target)
        adaption_info = adaptation_info_fn(state, info, adaption_state)
        return (state, adaption_state), adaption_info

    def run(rng_key, position, num_steps):
        initial_state = algorithm.init(
            position, logdensity_fn, **init_kwargs if init_kwargs else {}
        )
        initial_step_size = params.step_size
        init_adaptiation_state = adapt_init(initial_step_size)

        keys = jax.random.split(rng_key, num_steps)
        init_carry = (initial_state, init_adaptiation_state)
        (final_state, final_adaptation_state), info = jax.lax.scan(
            one_step, init_carry, keys
        )
        step_size = adapt_final(final_adaptation_state)
        parameters = params._replace(step_size=step_size)
        result = AdaptationResults(final_state, parameters)
        return result, info

    return AdaptationAlgorithm(run)


def step_size_and_scale_adaption(
    algorithm,
    logdensity_fn,
    params,
    target_acceptance_rate: float = 0.23,
    is_diagonal_matrix: bool = True,
    adaptation_info_fn=lambda *args, **kwargs: None,
    t0=10,
    gamma=0.05,
    kappa=0.75,
):
    if not hasattr(params, "step_size") or not hasattr(params, "scale"):
        raise ValueError("The params object must have a step_size and scale attribute")

    mcmc_step = algorithm(logdensity_fn)
    adapt_init, adapt_step, adapt_final = dual_averaging_adaptation(
        target_acceptance_rate,
        t0=t0,
        gamma=gamma,
        kappa=kappa,
    )
    adapt_init_scale, adapt_step_scale, adapt_final_scale = square_root_algorithm(
        is_diagonal_matrix=is_diagonal_matrix
    )

    def one_step(carry, key):
        state, ss_state, sr_state = carry
        step_size = jnp.exp(ss_state.log_step_size)
        scale, _, _ = adapt_final_scale(sr_state)

        new_params = params._replace(step_size=step_size, scale=scale)

        state, info = mcmc_step(
            key,
            state,
            new_params,
        )

        ss_state = adapt_step(ss_state, info.acceptance_rate)
        sr_state = adapt_step_scale(sr_state, jnp.array(state.position))
        info = adaptation_info_fn(state, info, (ss_state, sr_state))
        return (state, ss_state, sr_state), info

    def run(rng_key, position, num_steps):
        initial_state = algorithm.init(position, logdensity_fn)
        initial_step_size = params.step_size
        ss_state = adapt_init(initial_step_size)
        sr_state = adapt_init_scale(position.shape[0])

        keys = jax.random.split(rng_key, num_steps)
        init_carry = (initial_state, ss_state, sr_state)
        (final_state, final_ss_state, final_sr_state), info = jax.lax.scan(
            one_step, init_carry, keys
        )
        step_size = adapt_final(final_ss_state)
        scale, _, _ = adapt_final_scale(final_sr_state)

        parameters = params._replace(step_size=step_size, scale=scale)
        result = AdaptationResults(final_state, parameters)
        return result, info

    return AdaptationAlgorithm(run)


class SquareRootState(NamedTuple):
    mean: Array
    L: Array
    sample_size: int


def square_root_algorithm(
    is_diagonal_matrix: bool,
) -> tuple[Callable, Callable, Callable]:
    def init(n_dims: int) -> SquareRootState:
        """Initialize the covariance estimation.

        When the matrix is diagonal it is sufficient to work with an array that contains
        the diagonal value. Otherwise we need to work with the matrix in full.

        Parameters
        ----------
        n_dims: int
            The number of dimensions of the problem, which corresponds to the size
            of the corresponding square mass matrix.

        """
        sample_size = 0
        mean = jnp.zeros((n_dims,))
        L = jnp.ones((n_dims,)) if is_diagonal_matrix else jnp.eye(n_dims)
        return SquareRootState(mean, L, sample_size)

    def update(sq_state: SquareRootState, value: Array) -> SquareRootState:
        mean, L, sample_size = sq_state
        sample_size = sample_size + 1

        delta = value - mean
        mean = mean + delta / sample_size

        if is_diagonal_matrix:
            updated_delta = value - mean
            L = L + delta * updated_delta
        else:
            # This might be slightly biased, ...
            L = cholesky_update(L, delta)

        return SquareRootState(mean, L, sample_size)

    def final(
        sq_state: SquareRootState,
    ) -> tuple[Array, int, Array]:
        mean, L, sample_size = sq_state
        if is_diagonal_matrix:
            L = jnp.sqrt(L / (sample_size - 1))
        else:
            L = L / jnp.sqrt(sample_size - 1)

        return L, sample_size, mean

    return init, update, final
