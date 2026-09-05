from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.random_walk import RWInfo, RWState

from probjax.inference.base import Kernel, Warmup, WarmupResult
from probjax.inference.mcmc.base import make_kernel_api
from probjax.utils.typing import Array, PyTree, RngKey


class IMHParams(NamedTuple):
    pass


class NeuralIMHParams(NamedTuple):
    """Parameters for an independent neural proposal."""

    proposal_state: PyTree
    iteration: Array


class NeuralIMHState(NamedTuple):
    """IMH state with a cached proposal density."""

    position: PyTree
    logdensity: Array
    proposal_logdensity: Array
    proposal_iteration: Array


class NeuralIMHWarmupInfo(NamedTuple):
    """Per-round diagnostics from neural proposal adaptation."""

    losses: Array
    acceptance_rate: Array


def wrap_logpdf(logpdf: Callable) -> Callable:
    """Wrap the logpdf function to work with the IMH kernel."""

    def wrapped_logpdf(x: Any, y: Any, *args, **kwargs) -> Array:
        return logpdf(y, *args, **kwargs)

    return wrapped_logpdf


def build_imh_step(
    logdensity_fn: Callable,
    proposal_fn: Callable,
    proposal_logpdf: Callable,
) -> Callable:
    """A function to build the Independent Metropolis-Hastings kernel.

    Args:
        logdensity_fn (Callable): The log density function.
        proposal_fn (Callable): Proposal function.
        proposal_logpdf (Callable): Proposal log pdf.

    Returns:
        Callable: Step function for the IMH kernel.
    """
    kernel = blackjax.irmh.build_kernel()

    def step(
        key: RngKey, state: RWState, params: Optional[IMHParams] = None
    ) -> Tuple[RWState, RWInfo]:
        _proposal_fn = partial(proposal_fn, params=params)
        _proposal_logpdf = partial(wrap_logpdf(proposal_logpdf), params=params)
        return kernel(
            key,
            state,
            logdensity_fn,
            proposal_distribution=_proposal_fn,
            proposal_logdensity_fn=_proposal_logpdf,
        )

    return step


def init_imh_params(state: PyTree, rng_key=None) -> IMHParams:
    """Generally, there are no parameters to initialize for the IMH kernel."""
    return IMHParams()


imh = make_kernel_api(
    name="imh",
    init_fn=blackjax.irmh.init,
    init_params_fn=init_imh_params,
    build_step_fn=build_imh_step,
)


def neural_imh(logdensity_fn: Callable, proposal, *, event_spec=None) -> Kernel:
    """Build IMH with a tractable generative model as its proposal.

    Proposal weights are explicit kernel parameters so updates made during
    warmup remain visible inside compiled MCMC scans.
    """
    if event_spec is None:
        try:
            distribution = proposal.as_dist()
        except TypeError as error:
            raise TypeError(
                "event_spec is required for proposal models without an intrinsic "
                "event shape."
            ) from error
    else:
        distribution = proposal.as_dist(event_spec)
    if not distribution.has_logpdf:
        raise ValueError("The proposal model must expose a tractable logpdf.")
    distribution.compile("sample", "logpdf")

    def init(key, position=None, **kwargs):
        if position is None:
            position, key = key, kwargs.pop("rng_key", None)
        proposal_state = distribution.model_state()
        return NeuralIMHState(
            position,
            logdensity_fn(position),
            distribution.logpdf_with_state(proposal_state, position),
            jnp.array(0, dtype=jnp.uint32),
        )

    def init_params(state):
        del state
        return NeuralIMHParams(
            distribution.model_state(), jnp.array(0, dtype=jnp.uint32)
        )

    def step(key, state, params):
        key_proposal, key_accept = jax.random.split(key)
        proposed_position = distribution.sample_with_state(
            params.proposal_state, key_proposal
        )
        proposed_position = jax.tree.map(
            lambda proposed, current: proposed.astype(current.dtype),
            proposed_position,
            state.position,
        )
        proposed_state = NeuralIMHState(
            proposed_position,
            logdensity_fn(proposed_position),
            distribution.logpdf_with_state(params.proposal_state, proposed_position),
            params.iteration,
        )
        current_proposal_logdensity = jax.lax.cond(
            state.proposal_iteration == params.iteration,
            lambda: state.proposal_logdensity,
            lambda: distribution.logpdf_with_state(
                params.proposal_state, state.position
            ),
        )
        log_acceptance_ratio = (
            proposed_state.logdensity
            - proposed_state.proposal_logdensity
            - state.logdensity
            + current_proposal_logdensity
        )
        acceptance_rate = jnp.exp(jnp.minimum(log_acceptance_ratio, 0.0))
        is_accepted = jax.random.uniform(key_accept) < acceptance_rate
        current_state = NeuralIMHState(
            state.position,
            state.logdensity,
            current_proposal_logdensity,
            params.iteration,
        )
        next_state = jax.lax.cond(
            is_accepted, lambda: proposed_state, lambda: current_state
        )
        return next_state, RWInfo(acceptance_rate, is_accepted, proposed_state)

    return Kernel(init, step, init_params)


def _concatenate_data(left, right):
    if left is None:
        return right
    if jax.tree.structure(left) != jax.tree.structure(right):
        raise ValueError(
            "data and proposal samples must have the same pytree structure."
        )
    return jax.tree.map(lambda x, y: jnp.concatenate((x, y), axis=0), left, right)


def _limit_data(data, max_size):
    if max_size is None:
        return data
    return jax.tree.map(lambda value: value[-max_size:], data)


def _rao_blackwellized_data(
    initial_position, samples, proposed_positions, acceptance_rate
):
    current_positions = jax.tree.map(
        lambda initial, values: jnp.concatenate((initial[None], values[:-1])),
        initial_position,
        samples,
    )
    positions = jax.tree.map(
        lambda current, proposed: jnp.stack((current, proposed), axis=1).reshape(
            (-1,) + current.shape[1:]
        ),
        current_positions,
        proposed_positions,
    )
    weights = jnp.stack((1.0 - acceptance_rate, acceptance_rate), axis=1).reshape(-1)
    return positions, weights


def neural_imh_warmup(
    proposal,
    *,
    data=None,
    num_adaptations: int = 5,
    fit_steps: int = 100,
    batch_size: Optional[int] = 256,
    max_buffer_size: Optional[int] = 10_000,
    rao_blackwellize: bool = True,
    learning_rate: float = 1e-3,
    optimizer=None,
) -> Warmup:
    """Adapt a neural IMH proposal on chain positions and optional seed data.

    The requested warmup transitions are split into ``num_adaptations`` blocks.
    After each block, the proposal is fitted to positions collected so far,
    augmented by ``data`` when supplied. By default each transition contributes
    its current and proposed positions with weights ``1 - alpha`` and ``alpha``.
    The returned parameters are then fixed for regular MCMC sampling.
    """
    if num_adaptations < 1:
        raise ValueError("num_adaptations must be at least one.")
    if fit_steps < 1:
        raise ValueError("fit_steps must be at least one.")
    if max_buffer_size is not None and max_buffer_size < 1:
        raise ValueError("max_buffer_size must be positive or None.")
    fit_optimizer = optimizer
    if fit_optimizer is None:
        import optax

        fit_optimizer = optax.adam(learning_rate)

    def run(key, kernel, state, params, num_steps):
        from flax import nnx

        from probjax.inference.mcmc_runner import MCMC

        if not isinstance(params, NeuralIMHParams):
            raise TypeError("neural_imh_warmup requires a neural_imh kernel.")
        if num_steps < num_adaptations:
            raise ValueError("num_steps must be at least num_adaptations.")

        nnx.update(proposal, params.proposal_state)

        quotient, remainder = divmod(num_steps, num_adaptations)
        block_sizes = [
            quotient + (round_index < remainder)
            for round_index in range(num_adaptations)
        ]
        collected = None
        collected_weights = None
        losses = []
        acceptance_rates = []

        for block_size in block_sizes:
            key, sample_key, fit_key = jax.random.split(key, 3)
            initial_position = state.position
            collect_info = (
                ("acceptance_rate", "proposal")
                if rao_blackwellize
                else ("acceptance_rate",)
            )
            result = MCMC.sample_kernel(
                sample_key,
                kernel,
                state,
                block_size,
                params,
                collect_info=collect_info,
            )
            state = result.state
            if rao_blackwellize:
                round_data, round_weights = _rao_blackwellized_data(
                    initial_position,
                    result.samples,
                    result.info["proposal"].position,
                    result.info["acceptance_rate"],
                )
            else:
                round_data = result.samples
                round_weights = jnp.ones((block_size,))
            collected = _limit_data(
                _concatenate_data(collected, round_data), max_buffer_size
            )
            collected_weights = _limit_data(
                _concatenate_data(collected_weights, round_weights),
                max_buffer_size,
            )
            fit_data = _concatenate_data(data, collected)
            fit_weights = collected_weights
            if data is not None:
                num_seed = jax.tree.leaves(data)[0].shape[0]
                fit_weights = jnp.concatenate((jnp.ones(num_seed), fit_weights))
            round_losses = proposal.fit(
                fit_key,
                fit_data,
                weights=fit_weights,
                num_steps=fit_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer=fit_optimizer,
            )
            _, proposal_state = nnx.split(proposal)
            proposal_state = jax.tree.map(lambda value: value.copy(), proposal_state)
            params = NeuralIMHParams(proposal_state, params.iteration + 1)
            losses.append(round_losses)
            acceptance_rates.append(jnp.mean(result.info["acceptance_rate"]))

        info = NeuralIMHWarmupInfo(
            losses=jnp.stack(losses),
            acceptance_rate=jnp.stack(acceptance_rates),
        )
        return WarmupResult(state, params, info)

    return Warmup(run)


class GaussianIMHParams(NamedTuple):
    """Parameters for the Gaussian IMH kernel."""

    mean: Array
    cov: Array
    unflatten: Callable


def init_gaussian_imh_params(
    state: PyTree, mean: Optional[Array] = None, cov: Optional[Array] = None
) -> GaussianIMHParams:
    """Initialize the parameters for the Gaussian IMH kernel."""
    position = state.position if hasattr(state, "position") else state
    flat_position, unflatten = jax.flatten_util.ravel_pytree(position)
    if mean is None:
        mean = flat_position
    if cov is None:
        cov = jnp.ones_like(flat_position)
    return GaussianIMHParams(mean=mean, cov=cov, unflatten=unflatten)


def proposal_gaussian(key: RngKey, *, params: GaussianIMHParams):
    """Generate a new position from a Gaussian proposal."""
    mean = params.mean
    cov = params.cov
    eps = jax.random.normal(key, mean.shape)
    if cov.ndim == 1:
        new_position = mean + jnp.sqrt(cov) * eps
    else:
        new_position = mean + jnp.dot(jnp.linalg.cholesky(cov), eps)
    return params.unflatten(new_position)


def proposal_gaussian_logpdf(state, *, params: GaussianIMHParams):
    """Log pdf of the Gaussian proposal."""
    x = state.position
    flat_position, _ = jax.flatten_util.ravel_pytree(x)
    mean = params.mean
    cov = params.cov
    if cov.ndim == 1:
        return jax.scipy.stats.norm.logpdf(flat_position, mean, cov).sum()
    else:
        return jax.scipy.stats.multivariate_normal.logpdf(flat_position, mean, cov)


gaussian_imh = make_kernel_api(
    name="gaussian_imh",
    init_fn=blackjax.irmh.init,
    init_params_fn=init_gaussian_imh_params,
    build_step_fn=partial(
        build_imh_step,
        proposal_fn=proposal_gaussian,
        proposal_logpdf=proposal_gaussian_logpdf,
    ),
)
