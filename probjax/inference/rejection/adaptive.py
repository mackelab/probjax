"""Adaptive rejection sampling (Gilks & Wild, 1992) as a kernel."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from probjax.inference.mcmc.base import MarkovKernelAPI
from probjax.inference.rejection import univariate
from probjax.utils.typing import Array, ArrayLike


class ARSState(NamedTuple):
    """The last exact draw and the envelope refined so far."""

    position: Array
    envelope: univariate.Envelope


class ARSInfo(NamedTuple):
    """Per-draw diagnostics.

    ``num_proposals`` envelope draws were needed, of which ``num_evals``
    missed the squeeze and cost a log-density evaluation. ``is_accepted`` is
    false only if ``max_proposals`` ran out, in which case the state is
    unchanged.
    """

    num_proposals: Array
    num_evals: Array
    is_accepted: Array


def init(
    position: ArrayLike,
    logdensity_fn: Callable,
    rng: Any = None,
    init_points: ArrayLike | None = None,
    lower: ArrayLike = -jnp.inf,
    upper: ArrayLike = jnp.inf,
    max_points: int = 50,
) -> ARSState:
    del rng
    position = jnp.asarray(position, dtype=float).reshape(())
    if init_points is None:
        init_points = position + jnp.array([-1.0, 0.0, 1.0])
    env = univariate.envelope(logdensity_fn, init_points, lower, upper, max_points)
    return ARSState(position, env)


def init_params(state: ARSState) -> None:
    return None


def build_step(logdensity_fn: Callable, max_proposals: int = 1000) -> Callable:
    value_and_grad = jax.value_and_grad(logdensity_fn)

    def step(rng, state: ARSState, params: Any = None) -> tuple[ARSState, ARSInfo]:
        del params

        def propose(carry):
            key, state, accepted, num_proposals, num_evals = carry
            key, sample_key, accept_key = jax.random.split(key, 3)
            env = state.envelope
            candidate = univariate.sample(env, sample_key)
            log_u = jnp.log(jax.random.uniform(accept_key))
            upper = univariate.upper_hull(env, candidate)
            # Below the squeeze the target need not be evaluated at all.
            squeezed = log_u <= univariate.squeeze(env, candidate) - upper

            def evaluate(_):
                h, grad = value_and_grad(candidate)
                # Exactness does not depend on the envelope, so refining it
                # with every evaluated point is free.
                refined = univariate.refine(env, candidate, h, grad)
                return log_u <= h - upper, refined

            accept, env = jax.lax.cond(
                squeezed, lambda _: (jnp.array(True), env), evaluate, None
            )
            state = ARSState(jnp.where(accept, candidate, state.position), env)
            return (key, state, accept, num_proposals + 1, num_evals + ~squeezed)

        def searching(carry):
            _, _, accepted, num_proposals, _ = carry
            return ~accepted & (num_proposals < max_proposals)

        zero = jnp.zeros((), jnp.int32)
        _, state, accepted, num_proposals, num_evals = jax.lax.while_loop(
            searching, propose, (rng, state, jnp.array(False), zero, zero)
        )
        return state, ARSInfo(num_proposals, num_evals, accepted)

    return step


class ars(MarkovKernelAPI):
    """Adaptive rejection sampling for univariate log-concave targets.

    Every step returns one *exact, independent* draw: ``run`` collects iid
    samples, not a Markov chain. Draws come from a piecewise-exponential
    envelope of the log density; a draw that misses the squeeze costs one
    log-density evaluation and adds a tangent there, so the envelope tightens
    as sampling proceeds and later draws are cheaper. The envelope lives in the
    state; there are no parameters to adapt.

    The envelope is set up by ``kernel.init(key, position, init_points=None,
    lower=-inf, upper=inf, max_points=50)``. ``init_points`` default to
    ``position + [-1, 0, 1]``; on an unbounded side they must straddle the
    mode. The target must be log-concave.

    Args:
        logdensity_fn: Log density of the scalar target.
        max_proposals: Envelope draws allowed per step before giving up.
    """

    init = staticmethod(init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_step)
