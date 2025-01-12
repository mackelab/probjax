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


# This is the discrete time Kalman filter for a linear Gaussian model of the form:
# x_t = A_t x_{t-1} + C**1/2 @ w_t
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
        rng: Optional[jnp.ndarray] = None,
    ) -> Tuple[KalmanFilterState, KalmanFilterInfo]:
        mu0 = state.mean
        cov0 = state.cov
        t_old = state.t
        is_observed = observed is not None

        # Predict
        _f = lambda x: transition_fn(x, t_old, t)
        mu1_, _f_jvp = jax.linearize(_f, mu0)

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
    Kalman filter for a general state space model.

    $$dx_t = f(x_t, t) + B_t dw_t \qquad y_t = \mathcal{N}(y_t; C x_t, R_t)$$

    To build a Kalman filter kernel, we require the following components:

    Args:
        transition_matrix (Callable[[float | ArrayLike], ArrayLike] | ArrayLike):
            Transition matrix A_t
        transition_covariance_matrix (Callable[[float | ArrayLike], ArrayLike] |
            ArrayLike): Transition covariance matrix Q_t
        observation_matrix (Callable[[float | ArrayLike], ArrayLike] | ArrayLike):
            Observation matrix C_t
        observation_covariance (Callable[[float | ArrayLike], ArrayLike] | ArrayLike):
            Observation covariance matrix R_t
    """

    init = init
    build_kernel = build_kernel
