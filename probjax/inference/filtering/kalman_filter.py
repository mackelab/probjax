from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterAPI
from probjax.utils.linalg import batched_pcg_solve, lanczos_logdet
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


def default_solve(S, res, dense_mem_limit=200):
    """Solve S @ x = res for the Kalman gain and residual.

    Chooses between dense factorization and batched PCG based on total memory
    required. Dense needs to materialize S (obs_dim^2) on top of the already-
    materialized res (nrhs * obs_dim). PCG avoids materializing S entirely.

    Decision rule: use dense when obs_dim^2 * 8 bytes < dense_mem_limit MB,
    i.e. when the materialized S matrix fits comfortably in memory. This
    naturally accounts for the RHS shape: for small obs_dim, dense factorization
    is O(obs_dim^3) and amortizes over all nrhs columns cheaply. For large
    obs_dim, PCG avoids the O(obs_dim^3) factorization and O(obs_dim^2) memory.

    Args:
        S: SPD system matrix (obs_dim x obs_dim), array or LinearOperator.
        res: RHS matrix, shape (nrhs, obs_dim).
        dense_mem_limit: Max MB for the materialized S matrix. Default 200 MB,
            corresponding to obs_dim ~ 5000 in f64 or ~7000 in f32.
    """
    if isinstance(S, LinearOperator):
        obs_dim = S.out_dim
    else:
        S = jnp.asarray(S)
        obs_dim = S.shape[0]

    res = jnp.asarray(res)

    # Estimate S memory in MB (use f64 = 8 bytes as upper bound)
    s_mem_mb = obs_dim * obs_dim * 8 / (1024 * 1024)

    if isinstance(S, LinearOperator) and s_mem_mb > dense_mem_limit:
        # Batched PCG — no materialization
        matvec = S.operator
        rhs = res.T  # (obs_dim, nrhs)
        if rhs.ndim == 1:
            return jax.scipy.sparse.linalg.cg(matvec, rhs, tol=1e-4)[0]
        else:
            X, _info = batched_pcg_solve(matvec, rhs, tol=1e-4, block_size=128)
            return X.T
    else:
        # Dense solve — materialize S if needed, then factorize once
        if isinstance(S, LinearOperator):
            S = S.as_array()
        return jax.scipy.linalg.solve(S, res.T, assume_a="pos").T


def default_logdet(S, dense_mem_limit=200):
    """Compute logdet of S. Uses Lanczos for large LinearOperator, dense slogdet otherwise.

    Uses the same memory-based decision as default_solve: if materializing S
    would exceed dense_mem_limit MB, use matrix-free Lanczos SLQ instead.

    Benchmarks (CPU, f64): dense slogdet (including materialization) is faster
    until dim ~2000. Above that, Lanczos avoids O(n^2) materialization and
    O(n^3) factorization.
    """
    if isinstance(S, LinearOperator):
        dim = S.out_dim
        s_mem_mb = dim * dim * 8 / (1024 * 1024)
        if s_mem_mb > dense_mem_limit:
            return lanczos_logdet(S, num_steps=min(500, dim))
        S = S.as_array()
    return jnp.linalg.slogdet(jnp.asarray(S)).logabsdet


def _kalman_update(
    mu1_: ArrayLike,
    cov1_: ArrayLike,
    y_: ArrayLike,
    observed: ArrayLike,
    C,
    R,
    solve_fn: Callable,
    logdet_fn_: Callable,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Kalman observation update: innovation, gain, covariance update, log-likelihood.

    Shared by the standard KF and the EKF. Everything before this (predict step,
    linearization) is filter-specific; everything here is the same.

    Args:
        mu1_: Predicted mean.
        cov1_: Predicted covariance.
        y_: Predicted observation (C @ mu1_ for KF, h(mu1_) for EKF).
        observed: Actual observation.
        C: Observation matrix (or LinearOperator).
        R: Observation noise covariance (or None).
        solve_fn: Solves S @ x = rhs.
        logdet_fn_: Computes log determinant of S.

    Returns:
        (updated_mean, updated_cov, log_likelihood)
    """
    r = observed - y_

    # Materialize S and res for solve. C can stay as a LinearOperator —
    # we use its matvec to build the dense matrices S and res without
    # materializing C itself.
    if isinstance(C, LinearOperator):
        # S = C @ cov1_ @ C.T  (obs_dim x obs_dim)
        # cov1_ @ C.T = (C @ cov1_.T).T
        # C @ cov1_.T: apply C to each column of cov1_.T (= each row of cov1_)
        C_covT = jax.vmap(C.operator, in_axes=1, out_axes=1)(
            cov1_.T
        )  # (obs_dim, state_dim)
        S = C_covT @ C_covT.T
        # res = cov1_ @ C.T = (C @ cov1_.T).T = C_covT.T
        res = C_covT.T  # (state_dim, obs_dim)
    else:
        C = jnp.asarray(C)
        S = C @ cov1_ @ C.T
        res = cov1_ @ C.T

    S = S + R if R is not None else S

    K = solve_fn(S, res)

    mu1 = mu1_ + K @ r

    # cov1 = cov1_ - K @ C @ cov1_  =  cov1_ - K @ (C @ cov1_)
    if isinstance(C, LinearOperator):
        C_cov1 = jax.vmap(C.operator, in_axes=1, out_axes=1)(
            cov1_
        )  # (obs_dim, state_dim)
        cov1 = cov1_ - K @ C_cov1
    else:
        cov1 = cov1_ - K @ C @ cov1_

    logdet = logdet_fn_(S)
    log_likelihood = -0.5 * (logdet + r.T @ solve_fn(S, r))

    return jnp.asarray(mu1), jnp.asarray(cov1), log_likelihood


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
        rng_key: Optional[jnp.ndarray] = None,
    ) -> Tuple[KalmanFilterState, KalmanFilterInfo]:
        mu0 = state.mean
        cov0 = state.cov
        t_old = state.t
        is_observed = observed is not None

        # Phi - promised to be a linear mapping
        # Q a positive definite matrix
        Phi, Q = transition_model_fns(t_old, t)

        assert isinstance(Q, (jnp.ndarray, LinearOperator)), (
            "Q must be an Array or LinearOperator"
        )
        assert isinstance(Phi, (jnp.ndarray, LinearOperator)), (
            "Phi must be an Array or LinearOperator"
        )

        # Predict
        mu1_ = Phi @ mu0
        cov1_ = Phi @ cov0 @ Phi.T + Q
        # Materialize predicted covariance — needed for update and info
        mu1_ = mu1_.as_array() if isinstance(mu1_, LinearOperator) else mu1_
        cov1_ = cov1_.as_array() if isinstance(cov1_, LinearOperator) else cov1_

        if is_observed:
            C, R = observation_model_fns(t)

            assert isinstance(R, (ArrayLike, LinearOperator)) or R is None, (
                "R must be an Array or None"
            )
            assert isinstance(C, (ArrayLike, LinearOperator)), (
                "C must be an Array or LinearOperator"
            )

            y_ = C @ mu1_
            solve = default_solve if linear_solve is None else linear_solve
            _logdet_fn = logdet_fn if logdet_fn is not None else default_logdet

            mu1, cov1, log_likelihood = _kalman_update(
                mu1_, cov1_, y_, observed, C, R, solve, _logdet_fn
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

    @staticmethod
    def default_unpack(state, info):
        return (state.mean, state.cov)
