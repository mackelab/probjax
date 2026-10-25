"""The extended Kalman filter: a Kalman filter on a linearised model."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import jax

from probjax.inference.base import Kernel
from probjax.inference.filtering.gaussian import gaussian_filter
from probjax.inference.filtering.state_space import GaussianModel
from probjax.utils.linear_operator import Jacobian


def _linearize(fn, x, args, matrix_free):
    """``fn(x, *args)`` and its Jacobian in ``x``, dense or as an operator."""
    if matrix_free:
        # The operator keeps fn static and args as leaves, so passing theta
        # through args keeps traced values out of the static field.
        return fn(x, *args), Jacobian(fn, x, args)
    return fn(x, *args), jax.jacfwd(fn)(x, *args)


def _transition_mean(model, x, theta, t_previous, t):
    return model.transition(theta, x, t_previous, t)[0]


def _observation_mean(model, x, theta, t):
    return model.observation(theta, x, t)[0]


def extended_kalman_filter(
    model: GaussianModel,
    *,
    matrix_free: bool = False,
    linear_solve: Optional[Callable] = None,
    logdet_fn: Optional[Callable] = None,
) -> Kernel:
    r"""Extended Kalman filter for a nonlinear model with Gaussian noise.

    $$x_t = f(x_{t-1}) + w_t \qquad y_t = h(x_t) + v_t$$

    Each step linearises ``f`` at the filtered mean and ``h`` at the predicted
    mean, then applies the Kalman update. Its log-likelihood is that of the
    linearised model, an approximation.

    Args:
        model: The model, as functions of the parameters ``theta``.
        matrix_free: Use the Jacobians as matrix-free operators instead of
            materialising them; pays off for large states.
        linear_solve, logdet_fn: As for :func:`kalman_filter`.

    Returns:
        A :class:`~probjax.inference.Kernel` like :func:`kalman_filter`'s.
    """

    transition_mean = partial(_transition_mean, model)
    observation_mean = partial(_observation_mean, model)

    def predict(theta, mean, cov, t_previous, t):
        mean_pred, Phi = _linearize(
            transition_mean, mean, (theta, t_previous, t), matrix_free
        )
        Q = model.transition(theta, mean, t_previous, t)[1]
        return mean_pred, Phi @ cov @ Phi.T + Q

    def observe(theta, mean_pred, cov_pred, t):
        y_pred, C = _linearize(
            observation_mean, mean_pred, (theta, t), matrix_free
        )
        R = model.observation(theta, mean_pred, t)[1]
        return y_pred, C, R

    return gaussian_filter(
        model.initial, predict, observe, linear_solve, logdet_fn, symmetrize=True
    )
