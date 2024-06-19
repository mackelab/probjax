import blackjax

from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.flatten_util import ravel_pytree


from probjax.inference.filtering.base import FilterState, FilterInfo, FilterKernel
from probjax.inference.smc.resampling import (
    resample_systematic,
    resample_multinomial,
    resample_residual,
    resample_ot,
)
from blackjax.smc.ess import ess


class KalmanFilterState(NamedTuple):
    mean: ArrayLike
    cov: ArrayLike


class KalmanFilterInfo(NamedTuple):
    mean_pred: ArrayLike
    cov_pred: ArrayLike


def init(mean: ArrayLike, cov: ArrayLike) -> KalmanFilterState:
    return KalmanFilterState(mean, cov)


def unpack_matrix(A, t):
    if isinstance(A, Callable):
        return A(t)
    else:
        return A


# This is the discrete time Kalman filter for a linear Gaussian model of the form:
# x_t = A_t x_{t-1} + C**1/2 @ w_t
def build_discrete_kernel(
    transition_matrix: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
    transition_covariance_matrix: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
    observation_matrix: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
    observation_covariance: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
) -> Callable:

    def kernel(
        state: KalmanFilterState,
        t: Optional[float | int] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[jnp.ndarray] = None,
    ) -> Tuple[KalmanFilterState, KalmanFilterInfo]:

        mu0 = state.mean
        cov0 = state.cov
        is_observed = observed is not None

        Phi = unpack_matrix(transition_matrix, t)
        Q = unpack_matrix(transition_covariance_matrix, t)

        # Predict
        mu1_ = jnp.dot(Phi, mu0)
        cov1_ = jnp.dot(Phi, jnp.dot(cov0, Phi.T)) + Q

        if is_observed:
            C = unpack_matrix(observation_matrix, t)
            R = unpack_matrix(observation_covariance, t)

            # Kalman gain
            y = observed
            y_ = C @ mu1_
            r = y - y_
            S = C @ cov1_ @ C.T
            S = S + R
            K = cov1_ @ jnp.linalg.solve(S, C).T

            # Update mean and covariance
            mu1 = mu0 + K @ r
            cov1 = cov0 - K @ S @ K.T

            return KalmanFilterState(mu1, cov1), KalmanFilterInfo(mu1_, cov1_)
        else:
            return KalmanFilterState(mu1_, cov1_), KalmanFilterInfo(mu1_, cov1_)

    return kernel
