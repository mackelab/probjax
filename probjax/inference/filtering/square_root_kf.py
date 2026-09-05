from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterAPI


class SqKalmanFilterState(NamedTuple):
    mean: ArrayLike
    std: ArrayLike
    t: Optional[ArrayLike]


class SqKalmanFilterInfo(NamedTuple):
    mean_pred: ArrayLike
    std_pred: ArrayLike
    log_likelihood: Optional[ArrayLike] = None


def init(
    mean: ArrayLike, std: ArrayLike, t: Optional[ArrayLike] = None
) -> SqKalmanFilterState:
    return SqKalmanFilterState(mean, std, t)


def unpack_matrix(A: ArrayLike, t_old: ArrayLike, t: Optional[ArrayLike] = None):
    if callable(A):
        if t is not None:
            return A(t_old, t)
        else:
            return A(t_old)
    else:
        return A


def sqrt_kf_predict(
    m: ArrayLike,
    CL: ArrayLike,
    A: ArrayLike,
    QL: ArrayLike,
) -> Tuple[jax.Array, jax.Array]:
    """
    Predict step of the square root Kalman filter.
    """
    m_new = A @ m
    block = jnp.vstack((A @ CL, QL))
    _, R = jnp.linalg.qr(block, mode="reduced")
    CL_new = R.T
    return m_new, CL_new


def sqrt_kf_correct(
    m: ArrayLike,
    CL: ArrayLike,
    H: ArrayLike,
    RL: ArrayLike,
    y: ArrayLike,
    compute_likelihood: bool = False,
) -> Tuple[ArrayLike, ArrayLike, float]:
    """
    Correction step of the square root Kalman filter.
    """
    d, D = H.shape

    y_hat = H @ m

    # QR decomposition
    RL_padded = jnp.hstack((RL, jnp.zeros((d, D - d))))

    # Construct QR decomposition block
    block = jnp.vstack((RL_padded, H @ CL))
    _, R = jnp.linalg.qr(block, mode="reduced")

    # Extract submatrices
    SL = jnp.tril(R[:d, :d].T)  # Observation covariance sqrt
    CL_new_factor = jnp.tril(R[d:, d:].T)  # Updated state covariance sqrt

    # Make SL a valid Cholesky factor
    signs = jnp.sign(jnp.diag(SL))
    SL = SL * signs[:, None]
    SL_chol = jax.scipy.linalg.cholesky(SL @ SL.T, lower=True)

    residual = y - y_hat
    Sinv_residual = jnp.linalg.solve(SL, residual)

    m_new = m + jnp.dot(R[:d, d:].T, Sinv_residual)
    CL_new = CL_new_factor

    if compute_likelihood:
        log_likelihood = -0.5 * (
            jnp.dot(residual.T, jnp.linalg.solve(SL_chol, residual))
            + jnp.log(jnp.linalg.det(SL_chol))
        )
    else:
        log_likelihood = jnp.array(0.0)

    return m_new, CL_new, log_likelihood


def build_kernel(
    transition_matrix: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
    transition_covariance_matrix_sqrt: Callable[[float | ArrayLike], ArrayLike]
    | ArrayLike,
    observation_matrix: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
    observation_covariance: Callable[[float | ArrayLike], ArrayLike] | ArrayLike,
) -> Callable:
    def kernel(
        state: SqKalmanFilterState,
        t: Optional[ArrayLike] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[jnp.ndarray] = None,
    ) -> Tuple[SqKalmanFilterState, SqKalmanFilterInfo]:
        mu0 = state.mean
        std = state.std
        t_old = state.t
        is_observed = observed is not None

        # Predict step
        Phi = unpack_matrix(transition_matrix, t_old, t)
        Q_sqrt = unpack_matrix(transition_covariance_matrix_sqrt, t_old, t)

        # Predicted mean
        mu1_, std1_ = sqrt_kf_predict(mu0, std, Phi, Q_sqrt)

        # If observation is available, perform update step
        if is_observed:
            C = unpack_matrix(observation_matrix, t)
            R = unpack_matrix(observation_covariance, t)

            # Correct step
            H = C @ std1_
            R_sqrt = jax.scipy.linalg.cholesky(R, lower=True)
            mu1, std1, log_likelihood = sqrt_kf_correct(
                mu1_, std1_, H, R_sqrt, observed, compute_likelihood=True
            )

        else:
            # If no observation, no update step
            mu1 = mu1_
            std1 = std1_
            log_likelihood = jnp.array(0.0)

        # Construct the new state and info
        new_state = SqKalmanFilterState(mu1, std1, t)
        info = SqKalmanFilterInfo(
            mean_pred=mu1_, std_pred=std1_, log_likelihood=log_likelihood
        )

        return new_state, info

    return kernel


# API
class sq_kalman_filter(FilterAPI):
    r"""
    Square root Kalman filter for a linear Gaussian state space model.

    $$dx_t = A_t x_t + B_t dw_t \qquad y_t = \mathcal{N}(y_t; C x_t, R_t)$$

    To build a Kalman filter kernel, we require the following components:

    Args:
        transition_matrix (Callable[[float | ArrayLike], ArrayLike] | ArrayLike):
            Transition matrix A_t
        transition_covariance_matrix_sqrt
            (Callable[[float | ArrayLike], ArrayLike] | ArrayLike): Transition
                covariance matrix Q_t
        observation_matrix
            (Callable[[float | ArrayLike], ArrayLike] | ArrayLike): Observation matrix
            C_t
        observation_covariance
            (Callable[[float | ArrayLike], ArrayLike] | ArrayLike): Observation
            covariance matrix R_t
    """

    init = init
    build_kernel = build_kernel

    @staticmethod
    def default_unpack(state, info):
        return (state.mean, state.std)
