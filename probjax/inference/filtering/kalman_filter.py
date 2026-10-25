"""The Kalman filter: exact filtering of a linear-Gaussian model."""

from __future__ import annotations

from typing import Callable, Optional

from probjax.inference.base import Kernel
from probjax.inference.filtering.gaussian import gaussian_filter
from probjax.inference.filtering.state_space import LinearGaussianModel


def kalman_filter(
    model: LinearGaussianModel,
    *,
    linear_solve: Optional[Callable] = None,
    logdet_fn: Optional[Callable] = None,
) -> Kernel:
    r"""Kalman filter for a linear-Gaussian state-space model.

    $$x_t = \Phi x_{t-1} + w_t,\ w_t \sim \mathcal N(0, Q) \qquad
    y_t = C x_t + v_t,\ v_t \sim \mathcal N(0, R)$$

    The transition and observation matrices may be
    :mod:`~probjax.utils.linear_operator` operators: large innovation systems
    are then solved with CG and their log-determinant estimated by Lanczos.

    Args:
        model: The model, as functions of the parameters ``theta``.
        linear_solve: ``(S, rhs) -> S^{-1} rhs`` for the innovation covariance;
            defaults to dense Cholesky or CG by size.
        logdet_fn: ``S -> log det S``; defaults to ``slogdet`` or Lanczos.

    Returns:
        A :class:`~probjax.inference.Kernel` with ``init(key, theta, t0=0.0)``
        and ``step(key, state, theta, t, y, observed=True)``; ``theta`` is its
        params. ``state.log_likelihood`` is the exact log-likelihood of the
        observations so far.
    """

    def predict(theta, mean, cov, t_previous, t):
        Phi, Q = model.transition(theta, t_previous, t)
        return Phi @ mean, Phi @ cov @ Phi.T + Q

    def observe(theta, mean_pred, cov_pred, t):
        C, R = model.observation(theta, t)
        return C @ mean_pred, C, R

    return gaussian_filter(model.initial, predict, observe, linear_solve, logdet_fn)
