"""Rejection sampling from a :class:`~probjax.inference.Density` proposal."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from probjax.inference.adaptation import get_param, replace_params
from probjax.inference.base import Adaptor, Density
from probjax.inference.mcmc.base import MarkovKernelAPI
from probjax.utils.typing import Array


class RejectionState(NamedTuple):
    """The last exact draw."""

    position: Any
    logdensity: Array


class RejectionParams(NamedTuple):
    """The proposal's params and ``log M``, a bound on ``log p - log q``."""

    proposal: Any
    log_bound: Array


class RejectionInfo(NamedTuple):
    """Per-draw diagnostics.

    ``max_log_ratio`` is the largest ``log p - log q`` seen among this step's
    proposals; if it exceeds ``log_bound`` the bound is wrong there and the
    draw over-represents that region. ``is_accepted`` is false only if
    ``max_proposals`` ran out, in which case the state is unchanged.
    """

    num_proposals: Array
    max_log_ratio: Array
    is_accepted: Array


def init(position: Any, logdensity_fn: Callable, rng: Any = None) -> RejectionState:
    del rng
    return RejectionState(position, logdensity_fn(position))


def init_params(
    state: RejectionState, proposal: Any = None, log_bound: Any = -jnp.inf
) -> RejectionParams:
    return RejectionParams(proposal, jnp.asarray(log_bound, dtype=float))


def build_step(
    logdensity_fn: Callable, proposal: Density, max_proposals: int = 10_000
) -> Callable:
    def step(
        rng, state: RejectionState, params: RejectionParams
    ) -> tuple[RejectionState, RejectionInfo]:
        def propose(carry):
            key, state, accepted, num_proposals, max_log_ratio = carry
            key, sample_key, accept_key = jax.random.split(key, 3)
            candidate = proposal.sample(sample_key, params.proposal)
            candidate = jax.tree.map(
                lambda new, old: new.astype(old.dtype), candidate, state.position
            )
            logdensity = logdensity_fn(candidate)
            log_ratio = logdensity - proposal.logpdf(candidate, params.proposal)
            log_ratio = jnp.where(jnp.isnan(log_ratio), -jnp.inf, log_ratio)
            accept = jnp.log(jax.random.uniform(accept_key)) < (
                log_ratio - params.log_bound
            )
            state = jax.tree.map(
                lambda new, old: jnp.where(accept, new, old),
                RejectionState(candidate, logdensity),
                state,
            )
            max_log_ratio = jnp.maximum(max_log_ratio, log_ratio)
            return key, state, accept, num_proposals + 1, max_log_ratio

        def searching(carry):
            _, _, accepted, num_proposals, _ = carry
            return ~accepted & (num_proposals < max_proposals)

        _, state, accepted, num_proposals, max_log_ratio = jax.lax.while_loop(
            searching,
            propose,
            (rng, state, jnp.array(False), jnp.zeros((), jnp.int32), -jnp.inf),
        )
        return state, RejectionInfo(num_proposals, max_log_ratio, accepted)

    return step


def bound_adaptor(
    field: str = "log_bound", margin: float = 0.0, axis_name: str | None = None
) -> Adaptor:
    """Raise ``params.<field>`` to the largest ``log p - log q`` seen.

    The bound only grows, so draws taken while it is still growing are
    slightly biased towards where it was too low; run this as warmup and keep
    the bound fixed for the draws you report. With ``axis_name`` the chains
    under ``jax.vmap`` share the maximum over all of them.
    """

    def init(state: Any, params: Any) -> tuple[()]:
        return ()

    def update(state: Any, info: RejectionInfo, adaptor_state: Any, params: Any):
        seen = info.max_log_ratio + margin
        if axis_name is not None:
            seen = jax.lax.pmax(seen, axis_name)
        bound = jnp.maximum(get_param(params, field), seen)
        return adaptor_state, replace_params(params, **{field: bound}), None

    def finalize(adaptor_state: Any, params: Any) -> tuple[Any, None]:
        return params, None

    return Adaptor(init, update, finalize)


def _rejection_adaptation():
    return bound_adaptor()


class rejection(MarkovKernelAPI):
    """Rejection sampling with a :class:`~probjax.inference.Density` proposal.

    Every step returns one *exact, independent* draw from the target, provided
    ``params.log_bound`` bounds ``log p - log q`` from above: proposals
    ``x ~ q`` are accepted with probability ``exp(log p(x) - log q(x) - log M)``
    until one is. ``log p`` need not be normalised.

    Params come from ``kernel.init_params(state, proposal_params,
    log_bound=-inf)``. The default adaptation, :func:`bound_adaptor`, learns
    the bound as the largest ratio seen, so warm up before collecting draws. A
    learned proposal trains with :func:`~probjax.inference.fit_adaptor` on
    ``field="proposal"``; the accepted draws are exact target samples, so the
    default ``data_fn`` is the right training data.

    Args:
        logdensity_fn: Log density of the target, up to a constant.
        proposal: The proposal. It needs heavier tails than the target, or no
            finite bound exists.
        max_proposals: Proposals allowed per step before giving up.
    """

    init = staticmethod(init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_step)
    default_adaptation_fn = staticmethod(_rejection_adaptation)
