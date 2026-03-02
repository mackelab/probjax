from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterAPI
from probjax.inference.filtering.kalman_filter import (
    KalmanFilterInfo,
    KalmanFilterState,
    init,
)


# Extended Kalman filter for a nonlinear state space model of the form:
# x_{t+1} = f(x_t, t) + w_t, y_t ~ N(h(x_t, t), R_t)
def build_kernel(
    transition_fn: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
    observation_fn: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
    transition_matrix_and_covariance_fn: Callable[[float | ArrayLike], ArrayLike]
    | ArrayLike,
    observation_matrix_and_covariance_fn: Callable[[float | ArrayLike], ArrayLike]
    | ArrayLike,
) -> Callable:
    def kernel(
        state: KalmanFilterState,
        t: Optional[ArrayLike] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[jnp.ndarray] = None,
    ) -> Tuple[KalmanFilterState, KalmanFilterInfo]:
        mu0 = state.mean
        cov0 = state.cov
        t_old = state.t
        is_observed = observed is not None

        # Predict
        _f = lambda x: transition_fn(x, t_old, t)
        mu1_ = _f(mu0)

        # In this case it does not make to much sense to only use jvp
        Phi, Q = transition_matrix_and_covariance_fn(mu0, cov0, t)

        cov1_ = jnp.dot(Phi, jnp.dot(cov0, Phi.T)) + Q

        if is_observed:
            C, R = observation_matrix_and_covariance_fn(mu0, cov0, t)

            # Kalman gain
            y = observed
            y_ = observation_fn(mu1_, t)
            r = y - y_
            S = C @ cov1_ @ C.T
            S = S + R
            K = jnp.linalg.solve(S.T, (cov1_ @ C.T).T).T

            # Update mean and covariance
            mu1 = mu1_ + K @ r
            cov1 = cov1_ - K @ C @ cov1_
            cov1 = 0.5 * (cov1 + cov1.T)  # Ensure symmetry

            # Compute log likelihood
            log_likelihood = -0.5 * (
                jnp.linalg.slogdet(S)[1] + r.T @ jnp.linalg.solve(S, r)
            )

            return KalmanFilterState(mu1, cov1, t), KalmanFilterInfo(
                mu1_, cov1_, log_likelihood
            )
        else:
            return KalmanFilterState(mu1_, cov1_, t), KalmanFilterInfo(
                mu1_, cov1_, jnp.array(0.0)
            )

    return kernel


# API


class extended_kalman_filter(FilterAPI):
    r"""
    Extended Kalman filter for a nonlinear state space model.

    $$x_{t+1} = f(x_t, t) + w_t \qquad y_t = \mathcal{N}(y_t; h(x_t, t), R_t)$$

    The EKF linearizes the transition and observation functions around the current
    state estimate to apply the standard Kalman filter update equations.

    Args:
        transition_fn (Callable): Nonlinear transition function f(x, t_old, t) -> x_new
        observation_fn (Callable): Nonlinear observation function h(x, t) -> y
        transition_matrix_and_covariance_fn (Callable): Returns (Phi, Q) — the
            Jacobian of the transition function and the process noise covariance.
        observation_matrix_and_covariance_fn (Callable): Returns (C, R) — the
            Jacobian of the observation function and the observation noise covariance.
    """

    init = init
    build_kernel = build_kernel

    @staticmethod
    def default_unpack(state, info):
        return (state.mean, state.cov)
