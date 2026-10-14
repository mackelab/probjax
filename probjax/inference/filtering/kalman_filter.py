from typing import Callable, Optional, Tuple

from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterAPI
from probjax.inference.filtering.gaussian import (
    GaussianFilterInfo,
    GaussianFilterState,
    gaussian_filter_template,
    gaussian_init,
    gaussian_unpack,
)
from probjax.utils.linear_operator import (
    as_array,
    is_array_or_operator,
)

# Public state/info names for this filter. They share the Gaussian layout
# defined in gaussian.py; the aliases keep existing imports working
# (tests construct KalmanFilterState directly).
KalmanFilterState = GaussianFilterState
KalmanFilterInfo = GaussianFilterInfo
init = gaussian_init


# This is the discrete time Kalman filter for a linear Gaussian model of the form:
# x_t = A_t x_{t-1} + C**1/2 @ w_t
def build_kernel(
    transition_model_fns: Callable,
    observation_model_fns: Callable,
    linear_solve: Optional[Callable] = None,
    logdet_fn: Optional[Callable] = None,
) -> Callable:
    def kernel(
        state: KalmanFilterState,
        t: Optional[ArrayLike] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[ArrayLike] = None,
    ) -> Tuple[KalmanFilterState, KalmanFilterInfo]:
        def predict(mu0, cov0, t_old, t):
            # Phi - promised to be a linear mapping
            # Q a positive definite matrix
            Phi, Q = transition_model_fns(t_old, t)

            assert is_array_or_operator(Q), "Q must be an Array or LinearOperator"
            assert is_array_or_operator(Phi), "Phi must be an Array or LinearOperator"

            # Predict
            mu1_ = Phi @ mu0
            cov1_ = Phi @ cov0 @ Phi.T + Q
            # Materialize predicted covariance — needed for update and info
            return as_array(mu1_), as_array(cov1_)

        def observe(mu1_, cov1_, t):
            C, R = observation_model_fns(t)

            assert R is None or is_array_or_operator(R), "R must be an Array or None"
            assert is_array_or_operator(C), "C must be an Array or LinearOperator"

            return C @ mu1_, C, R

        return gaussian_filter_template(
            state, t, observed, predict, observe, linear_solve, logdet_fn
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

    default_unpack = staticmethod(gaussian_unpack)
