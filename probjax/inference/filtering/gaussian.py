"""Shared Gaussian-filter math, state types and step.

Every Gaussian filter (KF, EKF, UKF, square-root and rank-reduced KF) is a
:class:`~probjax.inference.Kernel` with ``init(key, theta, t0=0.0)`` and
``step(key, state, theta, t, y, observed=True)``. This module holds what they
share: the innovation likelihood, the Kalman update, and the predict/update
skeleton of the KF and the EKF.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from probjax.inference.base import Kernel
from probjax.utils.linalg import symmetrize_matrix
from probjax.utils.linear_operator import as_array, densify_if_small, is_operator
from probjax.utils.typing import Array, ArrayLike


class GaussianFilterState(NamedTuple):
    """The filtering distribution ``N(mean, cov)`` at time ``t``.

    ``log_likelihood`` accumulates ``log p(y_1:t | theta)`` over the steps.
    """

    mean: Array
    cov: Array
    t: Array
    log_likelihood: Array


class GaussianFilterInfo(NamedTuple):
    """The one-step prediction and this step's log-likelihood term."""

    mean_pred: Array
    cov_pred: Array
    log_likelihood_increment: Array


def as_time(t: ArrayLike) -> Array:
    return jnp.asarray(t, dtype=jnp.result_type(t, float))


def unobserved(observed: Any, y: Any) -> Any:
    """Zero the observation of a step that has none.

    A missing observation may be NaN. Under ``vmap`` both branches of the
    observed/unobserved ``cond`` run, and a NaN in the discarded one would
    still poison gradients.
    """
    return jnp.where(observed, y, jnp.zeros_like(y))


def default_solve(S: Any, res: Any, dense_mem_limit: int = 200) -> Array:
    """Solve S @ x = res for the Kalman gain and residual.

    Chooses between dense Cholesky and matrix-free CG based on the memory
    needed to materialize S. Dense needs S (obs_dim^2) on top of the already-
    materialized res (nrhs * obs_dim). CG avoids materializing S entirely.

    Decision rule: use dense when the materialized S fits in dense_mem_limit
    MB. For small obs_dim, the O(obs_dim^3) factorization amortizes over all
    nrhs columns cheaply. For large obs_dim, CG avoids the factorization and
    the O(obs_dim^2) memory.

    Args:
        S: SPD system matrix (obs_dim x obs_dim), array or LinearOperator.
        res: RHS matrix, shape (nrhs, obs_dim).
        dense_mem_limit: Max MB for the materialized S matrix. Default 200 MB,
            corresponding to obs_dim ~ 5000 in f64 or ~7000 in f32.
    """
    if is_operator(S):
        # LinearOperator.solve takes rhs as (obs_dim, nrhs); the Kalman
        # convention here is (nrhs, obs_dim), hence the transposes.
        S = densify_if_small(S, dense_mem_limit)
        return S.solve(jnp.asarray(res).T, psd=True).T
    S = jnp.asarray(S)
    res = jnp.asarray(res)
    return jax.scipy.linalg.solve(S, res.T, assume_a="pos").T


def default_logdet(S: Any, dense_mem_limit: int = 200) -> Array:
    """Compute the logdet with Lanczos for large operators, otherwise slogdet.

    Uses the same memory-based decision as default_solve: if materializing S
    would exceed dense_mem_limit MB, use matrix-free Lanczos SLQ instead.

    Benchmarks (CPU, f64): dense slogdet (including materialization) is faster
    until dim ~2000. Above that, Lanczos avoids O(n^2) materialization and
    O(n^3) factorization.
    """
    if is_operator(S):
        S = densify_if_small(S, dense_mem_limit)
        return S.logdet(psd=True, num_steps=500)
    return jnp.linalg.slogdet(jnp.asarray(S)).logabsdet


def innovation_loglik(
    residual: Array,
    S: Any,
    solve_fn: Callable = default_solve,
    logdet_fn: Callable = default_logdet,
) -> Array:
    """Normalized Gaussian innovation log-likelihood, including ``n*log(2*pi)``.

    Shared by the KF/EKF update, the UKF and the rank-reduced KF so the
    formula cannot drift between filters. ``solve_fn``/``logdet_fn`` default
    to the dense/iterative-dispatched versions; callers with exact legacy
    ops pass them explicitly.
    """
    return -0.5 * (
        residual.size * jnp.log(2 * jnp.pi)
        + logdet_fn(S)
        + residual.T @ solve_fn(S, residual)
    )


def kalman_update(
    mu1_: Array,
    cov1_: Array,
    y_: Array,
    observed: Array,
    C: Any,
    R: Any,
    solve_fn: Callable,
    logdet_fn_: Callable,
) -> Tuple[Array, Array, Array]:
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
    # we use operator composition to build the dense matrices S and res
    # without materializing C itself.
    S = as_array(C @ cov1_ @ C.T)  # (obs_dim x obs_dim)
    res = as_array(cov1_ @ C.T)  # (state_dim x obs_dim)
    if not is_operator(C):
        C = jnp.asarray(C)

    S = S + R if R is not None else S

    K = solve_fn(S, res)

    mu1 = mu1_ + K @ r

    # cov1 = cov1_ - K @ C @ cov1_  =  cov1_ - K @ (C @ cov1_)
    C_cov1 = as_array(C @ cov1_)  # (obs_dim x state_dim)
    cov1 = cov1_ - K @ C_cov1

    log_likelihood = innovation_loglik(r, S, solve_fn=solve_fn, logdet_fn=logdet_fn_)

    return jnp.asarray(mu1), jnp.asarray(cov1), log_likelihood


def gaussian_filter(
    initial: Callable,
    predict: Callable,
    observe: Callable,
    linear_solve: Callable | None = None,
    logdet_fn: Callable | None = None,
    symmetrize: bool = False,
) -> Kernel:
    """The predict/update kernel shared by the KF and the EKF.

    Args:
        initial: ``theta -> (mean, cov)`` at ``t0``.
        predict: ``(theta, mean, cov, t_previous, t) -> (mean_pred, cov_pred)``.
        observe: ``(theta, mean_pred, cov_pred, t) -> (y_pred, C, R)``.
        linear_solve, logdet_fn: Override how the innovation covariance is
            solved and its log-determinant computed.
        symmetrize: Symmetrize the updated covariance.
    """
    solve = default_solve if linear_solve is None else linear_solve
    logdet = default_logdet if logdet_fn is None else logdet_fn

    def init(key: Any, theta: Any, t0: ArrayLike = 0.0) -> GaussianFilterState:
        del key
        mean, cov = initial(theta)
        mean = jnp.asarray(mean)
        return GaussianFilterState(
            mean, as_array(cov), as_time(t0), jnp.zeros((), mean.dtype)
        )

    def step(
        key: Any,
        state: GaussianFilterState,
        theta: Any,
        t: ArrayLike,
        y: ArrayLike,
        observed: Any = True,
    ) -> Tuple[GaussianFilterState, GaussianFilterInfo]:
        del key
        t = jnp.asarray(t, state.t.dtype)
        mean_pred, cov_pred = predict(theta, state.mean, state.cov, state.t, t)
        mean_pred, cov_pred = jnp.asarray(mean_pred), as_array(cov_pred)
        y = unobserved(observed, jnp.asarray(y))

        def update(_):
            y_pred, C, R = observe(theta, mean_pred, cov_pred, t)
            mean, cov, increment = kalman_update(
                mean_pred, cov_pred, jnp.asarray(y_pred), y, C, R, solve, logdet
            )
            if symmetrize:
                cov = symmetrize_matrix(cov)
            return mean, cov, increment.astype(state.log_likelihood.dtype)

        def predict_only(_):
            return mean_pred, cov_pred, jnp.zeros_like(state.log_likelihood)

        mean, cov, increment = jax.lax.cond(observed, update, predict_only, None)
        new_state = GaussianFilterState(mean, cov, t, state.log_likelihood + increment)
        return new_state, GaussianFilterInfo(mean_pred, cov_pred, increment)

    return Kernel(init, step, lambda state, theta=None: theta)
