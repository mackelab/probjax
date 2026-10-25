"""State-space models: what a filter needs to know about the dynamics.

Each model is a set of pure functions of the parameters ``theta``, so a filter
can take ``theta`` as its kernel params: gradients, ``vmap`` and an outer
sampler over ``theta`` all see it as an ordinary argument.

This is a leaf module: concrete filters and trajectory samplers import it,
never the reverse.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional


class LinearGaussianModel(NamedTuple):
    """A linear-Gaussian state-space model, filtered exactly by a Kalman filter.

    Attributes:
        initial: ``theta -> (mean, covariance)`` of the state at ``t0``.
        transition: ``(theta, t_previous, t_next) -> (Phi, Q)``: the state moves
            to ``Phi x`` plus noise of covariance ``Q``. Either may be a
            :mod:`~probjax.utils.linear_operator` operator.
        observation: ``(theta, t) -> (C, R)``: ``y`` is ``C x`` plus noise of
            covariance ``R``.
    """

    initial: Callable
    transition: Callable
    observation: Callable


class GaussianModel(NamedTuple):
    """A nonlinear model with additive Gaussian noise, for the EKF and the UKF.

    Attributes:
        initial: ``theta -> (mean, covariance)`` of the state at ``t0``.
        transition: ``(theta, x, t_previous, t_next) -> (f(x), Q)``.
        observation: ``(theta, x, t) -> (h(x), R)``.
    """

    initial: Callable
    transition: Callable
    observation: Callable


class StateSpaceModel(NamedTuple):
    """A state-space model written for a single particle.

    All functions act on one latent state ``x``, which may be any pytree; the
    particle filter vectorizes them over particles.

    Attributes:
        init: ``(key, theta) -> x``, a draw from the initial law at ``t0``.
        step: ``(key, theta, x, t_previous, t_next) -> x``, a draw from the
            transition over one interval.
        log_likelihood: ``(theta, x, y, t) -> scalar``, the normalized
            observation log density.
        transition_logdensity: ``(theta, x_next, x_previous, t_previous, t_next)
            -> scalar``. Optional; needed by proposals, backward path sampling
            and particle Gibbs.
    """

    init: Callable
    step: Callable
    log_likelihood: Callable
    transition_logdensity: Optional[Callable] = None
