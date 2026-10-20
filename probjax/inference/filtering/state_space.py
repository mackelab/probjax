"""State-space model types and temporal filter protocol.

This is a leaf module: it imports only typing/jax, never concrete filters.
Concrete filters (``kalman_filter``, ``particle_filter``) and orchestration
(``temporal``, ``trajectory``, ``joint``) all import from here, never the
reverse.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional


class StateSpaceModel(NamedTuple):
    """A state-space model written for a single particle.

    All functions act on one latent state ``x``, which may be any pytree; the
    filter vectorizes them over particles.

    Attributes:
        init: ``(key, theta) -> x``, a draw from the initial law at ``t0``.
        step: ``(key, theta, x, t_previous, t_next) -> x``, a draw from the
            transition over one interval.
        log_likelihood: ``(theta, x, y, t) -> scalar``, the normalized
            observation log density.
        transition_logdensity: ``(theta, x_next, x_previous, t_previous, t_next)
            -> scalar``. Optional; needed only for backward path sampling and
            particle Gibbs.
    """

    init: Callable
    step: Callable
    log_likelihood: Callable
    transition_logdensity: Optional[Callable] = None


class LinearGaussianModel(NamedTuple):
    """A linear-Gaussian state-space model, filtered exactly by a Kalman filter.

    Attributes:
        initial: ``theta -> (mean, covariance)`` of the state at ``t0``.
        transition: ``(theta, t_previous, t_next) -> (Phi, Q)``.
        observation: ``(theta, t) -> (C, R)``.
    """

    initial: Callable
    transition: Callable
    observation: Callable


class TemporalFilter(NamedTuple):
    """Pure ``init(key, theta, t0)`` and ``step(key, state, theta, t, y, observed)``.

    ``step`` returns ``(state, info)`` with ``info.log_likelihood`` the
    incremental log likelihood. ``likelihood_kind`` describes the likelihood,
    not its logarithm: ``'exact'`` or ``'unbiased'``.
    """

    init: Callable
    step: Callable
    likelihood_kind: str

    def __call__(self, key, state, *args, **kwargs):
        """One step: the same as ``filter.step(key, state, ...)``."""
        return self.step(key, state, *args, **kwargs)


class TemporalTrace(NamedTuple):
    """Stored states at ``ts``, with their immediately preceding initial state.

    Windowed traces condition on the filtering distribution at the window start;
    they do not provide smoothed estimates for states discarded from the window.
    """

    initial_state: Any
    ts: Any
    states: Any
    infos: Any


class TemporalResult(NamedTuple):
    state: Any
    log_likelihood: Any
    key: Any
    trace: Any
