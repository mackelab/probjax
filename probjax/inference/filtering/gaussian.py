"""Shared Gaussian-filter math and state types.

Leaf module: imports only ``base`` (for ``FilterAPI`` unpack types) and
``utils``. Every Gaussian filter (KF, EKF, UKF likelihood, rank-reduced KF)
imports from here instead of reaching into ``kalman_filter``, so the concrete
Kalman filter is no longer a hub dependency.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from probjax.inference.filtering.base import FilterInfo
from probjax.utils.linalg import symmetrize_matrix
from probjax.utils.linear_operator import as_array, densify_if_small, is_operator
from probjax.utils.typing import Array, ArrayLike


class GaussianFilterState(NamedTuple):
    mean: Array
    cov: Array
    t: Optional[Array]


class GaussianFilterInfo(NamedTuple):
    mean_pred: Array
    cov_pred: Array
    log_likelihood: Optional[Array] = None


def _as_time(t: Optional[ArrayLike]) -> Optional[Array]:
    """Normalize a filter time stamp to an array (or ``None``)."""
    return None if t is None else jnp.asarray(t)


def gaussian_init(
    mean: ArrayLike, cov: ArrayLike, t: Optional[ArrayLike] = None
) -> GaussianFilterState:
    return GaussianFilterState(
        jnp.asarray(mean), jnp.asarray(cov), None if t is None else jnp.asarray(t)
    )


def gaussian_unpack(
    state: GaussianFilterState, info: FilterInfo
) -> Tuple[Array, Array]:
    """Return the posterior mean and covariance from a Gaussian filter state."""
    return state.mean, state.cov


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


def gaussian_filter_template(
    state: GaussianFilterState,
    t: Optional[ArrayLike],
    observed: Optional[ArrayLike],
    predict_fn: Callable,
    observe_fn: Callable,
    linear_solve: Callable | None = None,
    logdet_fn: Callable | None = None,
    symmetrize: bool = False,
) -> tuple[GaussianFilterState, GaussianFilterInfo]:
    """Shared predict/update skeleton for Gaussian (EKF-family) filters.

    ``predict_fn(mu0, cov0, t_old, t)`` returns ``(mu1_, cov1_)``;
    ``observe_fn(mu1_, cov1_, t)`` returns ``(y_, C, R)``. Everything else
    (observed fork, solver defaulting, state/info construction) is shared.
    """
    mu0 = state.mean
    cov0 = state.cov
    t_old = state.t
    is_observed = observed is not None

    mu1_, cov1_ = predict_fn(mu0, cov0, t_old, t)

    if is_observed:
        y_, C, R = observe_fn(mu1_, cov1_, t)

        solve = default_solve if linear_solve is None else linear_solve
        _logdet_fn = logdet_fn if logdet_fn is not None else default_logdet

        mu1, cov1, log_likelihood = kalman_update(
            jnp.asarray(mu1_),
            jnp.asarray(cov1_),
            jnp.asarray(y_),
            jnp.asarray(observed),
            C,
            R,
            solve,
            _logdet_fn,
        )
        if symmetrize:
            cov1 = symmetrize_matrix(cov1)

        return GaussianFilterState(mu1, cov1, _as_time(t)), GaussianFilterInfo(
            mu1_, cov1_, log_likelihood
        )
    else:
        return GaussianFilterState(mu1_, cov1_, _as_time(t)), GaussianFilterInfo(
            mu1_, cov1_, jnp.array(0.0)
        )
