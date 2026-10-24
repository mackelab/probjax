from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple

import blackjax
import jax
import jax.numpy as jnp
import optax
from blackjax.mcmc.random_walk import RWInfo, RWState
from flax import nnx
from jax.flatten_util import ravel_pytree

from probjax.inference.adaptation import compose_adaptors
from probjax.inference.base import Kernel, Warmup, WarmupResult
from probjax.inference.mcmc.adaptation import covariance_adaptor, mean_adaptor
from probjax.inference.mcmc.base import (
    MarkovKernelAPI,
    ravel_position,
    sample_kernel,
)
from probjax.nn.training import FitConfig
from probjax.utils.jaxutils import leading_dim
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey


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


class NeuralIMHInfo(NamedTuple):
    """Per-step diagnostics of the neural IMH kernel.

    Mirrors BlackJAX's ``RWInfo`` but keeps the proposal state of probjax's
    own ``NeuralIMHState`` and the array-valued acceptance statistics.
    """

    acceptance_rate: Array
    is_accepted: Array
    proposal: NeuralIMHState


def wrap_logpdf(logpdf: Callable) -> Callable:
    """Wrap the logpdf function to work with the IMH kernel."""

    def wrapped_logpdf(x: ArrayLike, y: ArrayLike, *args: Any, **kwargs: Any) -> Array:
        return logpdf(y, *args, **kwargs)

    return wrapped_logpdf


def build_imh_step(
    logdensity_fn: Callable,
    proposal_fn: Callable,
    proposal_logpdf: Callable,
) -> Callable:
    kernel = blackjax.irmh.build_kernel()

    def step(
        rng, state: RWState, params: IMHParams | None = None
    ) -> tuple[RWState, RWInfo]:
        _proposal_fn = partial(proposal_fn, params=params)
        _proposal_logpdf = partial(wrap_logpdf(proposal_logpdf), params=params)
        return kernel(
            rng,
            state,
            logdensity_fn,
            proposal_distribution=_proposal_fn,
            proposal_logdensity_fn=_proposal_logpdf,
        )

    return step


def init_imh_params(state: Any, rng = None) -> IMHParams:
    """Generally, there are no parameters to initialize for the IMH kernel."""
    return IMHParams()


class imh(MarkovKernelAPI):
    r"""Independent Metropolis-Hastings with a user-supplied proposal.

    Proposals ignore the current position. The kernel params are passed to
    both proposal functions as ``params=``; the default ``init_params``
    returns the empty :class:`IMHParams`.

    Args:
        logdensity_fn: Log density of the target.
        proposal_fn: ``(key, params) -> position``.
        proposal_logpdf: ``(state, params) -> log q(state.position)``,
            evaluated on BlackJAX ``RWState``\ s.
    """

    init = staticmethod(blackjax.irmh.init)
    init_params = staticmethod(init_imh_params)
    build_step = staticmethod(build_imh_step)


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

    def init(rng: RngKey, position: Any):
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

    def step(rng, state, params):
        key_proposal, key_accept = jax.random.split(rng)
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
        return next_state, NeuralIMHInfo(acceptance_rate, is_accepted, proposed_state)

    return Kernel(init, step, init_params, lambda: neural_imh_warmup(proposal))


def _concatenate_data(left: Any, right: Any) -> Any:
    if left is None:
        return right
    if jax.tree.structure(left) != jax.tree.structure(right):
        raise ValueError(
            "data and proposal samples must have the same pytree structure."
        )
    return jax.tree.map(lambda x, y: jnp.concatenate((x, y), axis=0), left, right)


def _limit_data(data: Any, max_size: int | None) -> Any:
    if max_size is None:
        return data
    return jax.tree.map(lambda value: value[-max_size:], data)


def _rao_blackwellized_data(
    initial_position: Any,
    samples: Any,
    proposed_positions: Any,
    acceptance_rate: Array,
) -> tuple[Any, Array]:
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
    data: Any = None,
    num_adaptations: int = 5,
    fit_steps: int = 100,
    batch_size: int | None = 256,
    max_buffer_size: int | None = 10_000,
    rao_blackwellize: bool = True,
    learning_rate: float = 1e-3,
    optimizer: Any = None,
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
        fit_optimizer = optax.adam(learning_rate)

    def run(
        rng, kernel: Kernel, state: Any, params: Any, num_steps: int
    ) -> WarmupResult:
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
            rng, sample_key, fit_key = jax.random.split(rng, 3)
            initial_position = state.position
            collect_info = (
                ("acceptance_rate", "proposal")
                if rao_blackwellize
                else ("acceptance_rate",)
            )
            result = sample_kernel(
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
                num_seed = leading_dim(data)
                fit_weights = jnp.concatenate((jnp.ones(num_seed), fit_weights))
            round_result = proposal.fit(
                fit_key,
                fit_data,
                weights=fit_weights,
                config=FitConfig(
                    max_steps=fit_steps,
                    batch_size=batch_size,
                    # The optimizer carries the learning rate; the config's
                    # own field only matters when it builds one.
                    optimizer=fit_optimizer,
                ),
            )
            round_losses = round_result.losses
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
    """Parameters for the Gaussian IMH kernel (arrays only, JIT-safe)."""

    mean: Array
    cov: Array


def init_gaussian_imh_params(
    state: Any, mean: Array | None = None, cov: Array | None = None
) -> GaussianIMHParams:
    """Initialize the parameters for the Gaussian IMH kernel."""
    flat_position = ravel_position(state)
    if mean is None:
        mean = flat_position
    if cov is None:
        cov = jnp.ones_like(flat_position)
    return GaussianIMHParams(mean=mean, cov=cov)


def proposal_gaussian(
    rng, *, params: GaussianIMHParams, unflatten: Callable
) -> Any:
    """Generate a new position from a Gaussian proposal."""
    mean = params.mean
    cov = params.cov
    eps = jax.random.normal(rng, mean.shape)
    if cov.ndim == 1:
        new_position = mean + jnp.sqrt(cov) * eps
    else:
        new_position = mean + jnp.dot(jnp.linalg.cholesky(cov), eps)
    return unflatten(new_position)


def build_gaussian_imh_step(logdensity_fn: Callable) -> Callable:
    """Build the Gaussian IMH step, deriving the unravel fn from state.

    The pytree structure is static, so ``ravel_pytree`` runs fine inside the
    compiled step — and no Python callable ends up in the JIT-traced params.
    """
    kernel = blackjax.irmh.build_kernel()

    def step(
        rng, state: RWState, params: GaussianIMHParams
    ) -> tuple[RWState, RWInfo]:
        _, unflatten = ravel_pytree(state.position)
        _proposal_fn = partial(proposal_gaussian, params=params, unflatten=unflatten)
        _proposal_logpdf = partial(wrap_logpdf(proposal_gaussian_logpdf), params=params)
        return kernel(
            rng,
            state,
            logdensity_fn,
            proposal_distribution=_proposal_fn,
            proposal_logdensity_fn=_proposal_logpdf,
        )

    return step


def proposal_gaussian_logpdf(state: Any, *, params: GaussianIMHParams) -> Array:
    """Log pdf of the Gaussian proposal."""
    x = state.position
    flat_position, _ = ravel_pytree(x)
    mean = params.mean
    cov = params.cov
    if cov.ndim == 1:
        return jax.scipy.stats.norm.logpdf(flat_position, mean, jnp.sqrt(cov)).sum()
    else:
        return jnp.asarray(
            jax.scipy.stats.multivariate_normal.logpdf(flat_position, mean, cov)
        )


def _gaussian_imh_adaptation():
    return compose_adaptors(
        mean=mean_adaptor(field="mean"),
        cov=covariance_adaptor(field="cov", output="covariance"),
    )


class gaussian_imh(MarkovKernelAPI):
    """Independent Metropolis-Hastings with a Gaussian proposal.

    Tunable parameters come from ``kernel.init_params(state, mean=None,
    cov=None)`` (see :class:`GaussianIMHParams`); ``mean`` defaults to the
    initial position and ``cov`` (a diagonal vector or a dense matrix) to the
    identity. Default adaptation: running estimates of the mean and
    covariance.

    Args:
        logdensity_fn: Log density of the target.
    """

    init = staticmethod(blackjax.irmh.init)
    init_params = staticmethod(init_gaussian_imh_params)
    build_step = staticmethod(build_gaussian_imh_step)
    default_adaptation_fn = staticmethod(_gaussian_imh_adaptation)
