"""Independent Metropolis-Hastings.

The proposal ignores the current position, so the chain mixes exactly as well
as the proposal covers the target. That makes IMH the natural home for a
learned proposal: :func:`~probjax.inference.neural.density` turns a flow into
a :class:`~probjax.inference.Density`, the kernel carries its weights as
``params``, and the default adaptation trains them on the chain's own output.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple

import blackjax
import jax
import jax.numpy as jnp
import optax
from blackjax.mcmc.random_walk import RWState
from jax.flatten_util import ravel_pytree

from probjax.inference.adaptation import (
    compose_adaptors,
    fit_adaptor,
    maximum_likelihood,
)
from probjax.inference.base import Density
from probjax.inference.mcmc.adaptation import covariance_adaptor, mean_adaptor
from probjax.inference.mcmc.base import MarkovKernelAPI, ravel_position
from probjax.utils.typing import Array


class IMHInfo(NamedTuple):
    """Per-step diagnostics.

    ``previous_position`` is the position the step started from; with
    ``proposal`` and ``acceptance_rate`` it gives the Rao-Blackwellised
    estimate of the next position's distribution (see :func:`imh_data`).
    """

    acceptance_rate: Array
    is_accepted: Array
    proposal: Any
    previous_position: Any


def imh_step(
    rng, state: RWState, params: Any, logdensity_fn: Callable, proposal: Density
) -> tuple[RWState, IMHInfo]:
    """One independent Metropolis-Hastings transition."""
    key_proposal, key_accept = jax.random.split(rng)
    proposed = proposal.sample(key_proposal, params)
    proposed = jax.tree.map(
        lambda new, old: new.astype(old.dtype), proposed, state.position
    )
    proposed_logdensity = logdensity_fn(proposed)
    # q is re-evaluated at the current position on every step: one extra
    # density call, but exact while an adaptor is changing the proposal.
    log_ratio = (
        proposed_logdensity
        - proposal.logpdf(proposed, params)
        - state.logdensity
        + proposal.logpdf(state.position, params)
    )
    log_ratio = jnp.where(jnp.isnan(log_ratio), -jnp.inf, log_ratio)
    acceptance_rate = jnp.exp(jnp.minimum(log_ratio, 0.0))
    is_accepted = jax.random.uniform(key_accept) < acceptance_rate
    new_state = jax.tree.map(
        lambda new, old: jnp.where(is_accepted, new, old),
        RWState(proposed, proposed_logdensity),
        state,
    )
    return new_state, IMHInfo(acceptance_rate, is_accepted, proposed, state.position)


def imh_data(state: RWState, info: IMHInfo) -> tuple[Any, Array]:
    """Rao-Blackwellised training data for :func:`fit_adaptor`.

    The next position is the proposal with probability ``acceptance_rate`` and
    the previous position otherwise, so both are stored with those weights.
    Proposals the chain rejects still teach the proposal where the target is
    not.
    """
    del state
    points = jax.tree.map(
        lambda previous, proposed: jnp.stack((previous, proposed)),
        info.previous_position,
        info.proposal,
    )
    weights = jnp.stack((1.0 - info.acceptance_rate, info.acceptance_rate))
    return points, weights


def _init_proposal_params(state: Any, proposal_params: Any = None) -> Any:
    return proposal_params


def _build_imh_step(logdensity_fn: Callable, proposal: Density) -> Callable:
    return partial(imh_step, logdensity_fn=logdensity_fn, proposal=proposal)


def _imh_adaptation(
    proposal: Density,
    learning_rate: float = 3e-3,
    **options: Any,
):
    options = {"data_fn": imh_data, **options}
    return fit_adaptor(
        maximum_likelihood(proposal), optax.adam(learning_rate), **options
    )


class imh(MarkovKernelAPI):
    """Independent Metropolis-Hastings with a :class:`~probjax.inference.Density` proposal.

    The kernel's ``params`` are the proposal's params, so a learned proposal's
    weights live in the compiled scan and can be trained there. Default
    adaptation: :func:`~probjax.inference.fit_adaptor` maximising the
    proposal's likelihood on Rao-Blackwellised chain output, one Adam step
    per transition. Pass ``fit_adaptor(..., every=k, num_fit_steps=m)`` as the
    warmup policy to alternate sampling and fitting blocks instead.

    Args:
        logdensity_fn: Log density of the target.
        proposal: The independent proposal. Build one from a flow with
            :func:`probjax.inference.neural.density`.
    """

    init = staticmethod(blackjax.irmh.init)
    init_params = staticmethod(_init_proposal_params)
    build_step = staticmethod(_build_imh_step)
    default_adaptation_fn = staticmethod(_imh_adaptation)


class GaussianIMHParams(NamedTuple):
    """Gaussian proposal over the flattened position.

    ``cov`` is a diagonal vector or a dense matrix.
    """

    mean: Array
    cov: Array


def init_gaussian_imh_params(
    state: Any, mean: Array | None = None, cov: Array | None = None
) -> GaussianIMHParams:
    """``mean`` defaults to the initial position, ``cov`` to the identity."""
    flat_position = ravel_position(state)
    if mean is None:
        mean = flat_position
    if cov is None:
        cov = jnp.ones_like(flat_position)
    return GaussianIMHParams(mean=mean, cov=cov)


def gaussian_proposal(unflatten: Callable) -> Density:
    """A Gaussian :class:`~probjax.inference.Density` with :class:`GaussianIMHParams`."""

    def sample(rng, params: GaussianIMHParams) -> Any:
        noise = jax.random.normal(rng, params.mean.shape)
        if params.cov.ndim == 1:
            return unflatten(params.mean + jnp.sqrt(params.cov) * noise)
        return unflatten(params.mean + jnp.linalg.cholesky(params.cov) @ noise)

    def logpdf(position: Any, params: GaussianIMHParams) -> Array:
        flat_position, _ = ravel_pytree(position)
        if params.cov.ndim == 1:
            scale = jnp.sqrt(params.cov)
            return jax.scipy.stats.norm.logpdf(flat_position, params.mean, scale).sum()
        return jax.scipy.stats.multivariate_normal.logpdf(
            flat_position, params.mean, params.cov
        )

    return Density(sample, logpdf)


def _build_gaussian_imh_step(logdensity_fn: Callable) -> Callable:
    def step(rng, state: RWState, params: GaussianIMHParams):
        # The position's pytree structure is static, so unravelling it here
        # costs nothing at run time.
        _, unflatten = ravel_pytree(state.position)
        return imh_step(rng, state, params, logdensity_fn, gaussian_proposal(unflatten))

    return step


def _gaussian_imh_adaptation():
    return compose_adaptors(
        mean=mean_adaptor(field="mean"),
        cov=covariance_adaptor(field="cov", output="covariance"),
    )


class gaussian_imh(MarkovKernelAPI):
    """Independent Metropolis-Hastings with a Gaussian proposal.

    Tunable parameters come from ``kernel.init_params(state, mean=None,
    cov=None)`` (see :class:`GaussianIMHParams`). Default adaptation: running
    estimates of the mean and covariance.

    Args:
        logdensity_fn: Log density of the target.
    """

    init = staticmethod(blackjax.irmh.init)
    init_params = staticmethod(init_gaussian_imh_params)
    build_step = staticmethod(_build_gaussian_imh_step)
    default_adaptation_fn = staticmethod(_gaussian_imh_adaptation)
