from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterAPI
from probjax.utils.linear_operator import LinearOperator


class KalmanFilterState(NamedTuple):
    mean: ArrayLike
    cov: ArrayLike
    t: Optional[ArrayLike]


class KalmanFilterInfo(NamedTuple):
    mean_pred: ArrayLike
    cov_pred: ArrayLike
    log_likelihood: Optional[ArrayLike] = None


def init(
    mean: ArrayLike, cov: ArrayLike, t: Optional[ArrayLike] = None
) -> KalmanFilterState:
    return KalmanFilterState(mean, cov, t)


def default_solve(S, res):
    # If given as a matrix -> dense solve
    S = jnp.asarray(S)
    res = jnp.asarray(res)
    return jax.scipy.linalg.solve(S, res.T, assume_a="pos").T


# This is the discrete time Kalman filter for a linear Gaussian model of the form:
# x_t = A_t x_{t-1} + C**1/2 @ w_t
def build_kernel(
    transition_model_fns: Callable,
    observation_model_fns: Callable,
    linear_solve: Optional[Callable] = None,
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

        # Phi - promised to be a linear mapping
        # Q a positive definite matrix
        Phi, Q = transition_model_fns(t_old, t)

        assert isinstance(Q, (ArrayLike, LinearOperator)), "Q must be and Array"
        assert isinstance(Phi, (ArrayLike, LinearOperator)), (
            "Phi must be an Array or LinearOperator"
        )

        # Predict
        mu1_ = Phi @ mu0
        cov1_ = Phi @ cov0 @ Phi.T + Q

        if is_observed:
            C, R = observation_model_fns(t)

            assert isinstance(R, (ArrayLike, LinearOperator)) or R is None, (
                "R must be an Array or None"
            )
            assert isinstance(C, (ArrayLike, LinearOperator)), (
                "C must be an Array or LinearOperator"
            )

            # Kalman gain
            y = observed
            y_ = C @ mu1_
            r = y - y_
            S = C @ cov1_ @ C.T
            S = S + R if R else S
            res = cov1_ @ C.T

            solve = default_solve if linear_solve is None else linear_solve

            K = solve(S, res)

            # Update mean and covariance
            mu1 = mu1_ + K @ r
            cov1 = cov1_ - K @ C @ cov1_
            # Update log likelihood
            logdet = jnp.linalg.slogdet(jnp.asarray(S))[1]
            log_likelihood = -0.5 * (logdet + r.T @ solve(S, r))

            mu1 = jnp.asarray(mu1)
            cov1 = jnp.asarray(cov1)
            mu1_ = jnp.asarray(mu1_)
            cov1_ = jnp.asarray(cov1_)
            return KalmanFilterState(mu1, cov1, t), KalmanFilterInfo(
                mu1_, cov1_, log_likelihood
            )
        else:
            mu1_ = jnp.asarray(mu1_)
            cov1_ = jnp.asarray(cov1_)
            return KalmanFilterState(mu1_, cov1_, t), KalmanFilterInfo(
                mu1_, cov1_, jnp.array(0.0)
            )

    return kernel


# API


class kalman_filter(FilterAPI):
    r"""
    Kalman filter for a linear Gaussian state space model.

    $$dx_t = A_t x_t + B_t dw_t \qquad y_t = \mathcal{N}(y_t; C x_t, R_t)$$

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
