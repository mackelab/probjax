"""The rank-reduced Kalman filter, for states too large for a dense covariance."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from probjax.inference.base import Kernel
from probjax.inference.filtering.gaussian import (
    as_time,
    default_solve,
    innovation_loglik,
    unobserved,
)
from probjax.inference.filtering.state_space import LinearGaussianModel
from probjax.utils.linalg import symmetrize_matrix
from probjax.utils.typing import Array, ArrayLike


class RankReducedKalmanFilterState(NamedTuple):
    """The filtering distribution with ``cov = cov_factor cov_core cov_factor^T``."""

    mean: Array
    cov_factor: Array
    cov_core: Array
    t: Array
    log_likelihood: Array


class RankReducedKalmanFilterInfo(NamedTuple):
    mean_pred: Array
    cov_factor_pred: Array
    cov_core_pred: Array
    log_likelihood_increment: Array


def _eigh_clip(cov: Array, min_eig: float) -> Tuple[Array, Array]:
    vals, vecs = jnp.linalg.eigh(symmetrize_matrix(jnp.asarray(cov)))
    vals = jnp.clip(vals, min=min_eig)
    return vals, vecs


def _matrix_to_lowrank(
    cov: Array,
    rank: Optional[int],
    min_eig: float,
) -> Tuple[Array, Array]:
    vals, vecs = _eigh_clip(cov, min_eig=min_eig)
    k = vals.shape[0] if rank is None else min(rank, vals.shape[0])
    vals = vals[-k:]
    vecs = vecs[:, -k:]
    return vecs, jnp.diag(vals)


def _core_sqrt(core: Array, min_eig: float) -> Array:
    vals, vecs = _eigh_clip(core, min_eig=min_eig)
    return vecs @ jnp.diag(jnp.sqrt(vals))


def _compress_from_chol_like(
    L: Array,
    rank: int,
    min_eig: float,
) -> Tuple[Array, Array]:
    # P = L L^T; use SVD of L to get low-rank eigendecomposition.
    U, sing, _ = jnp.linalg.svd(L, full_matrices=False)
    k = min(rank, sing.shape[0])
    U = U[:, :k]
    vals = jnp.clip(sing[:k] ** 2, min=min_eig)
    return U, jnp.diag(vals)


def _effective_rank_from_energy(
    vals_desc: Array,
    energy_threshold: float,
    min_rank: int,
) -> Array:
    total = jnp.sum(vals_desc)
    total = jnp.where(total <= 0.0, 1.0, total)
    cumsum = jnp.cumsum(vals_desc)
    ratios = cumsum / total
    k = jnp.argmax(ratios >= energy_threshold) + 1
    k = jnp.maximum(k, min_rank)
    k = jnp.minimum(k, vals_desc.shape[0])
    return k


def _apply_energy_threshold(
    vals_desc: Array,
    min_eig: float,
    energy_threshold: Optional[float],
    min_rank: int,
) -> Array:
    vals_desc = jnp.clip(vals_desc, min=min_eig)
    if energy_threshold is None:
        return vals_desc
    k = _effective_rank_from_energy(vals_desc, energy_threshold, min_rank)
    idx = jnp.arange(vals_desc.shape[0])
    keep = idx < k
    return jnp.where(keep, vals_desc, min_eig)


def _sum_lowrank_terms(
    factor_a: Array,
    core_a: Array,
    factor_b: Array,
    core_b: Array,
    rank: int,
    min_eig: float,
) -> Tuple[Array, Array]:
    # A + B = (La La^T) + (Lb Lb^T) with
    # La = factor_a @ sqrt(core_a), Lb = factor_b @ sqrt(core_b)
    La = factor_a @ _core_sqrt(core_a, min_eig=min_eig)
    Lb = factor_b @ _core_sqrt(core_b, min_eig=min_eig)
    L = jnp.concatenate([La, Lb], axis=1)
    return _compress_from_chol_like(L, rank=rank, min_eig=min_eig)


def _parse_process_noise(
    Q: Any,
    process_noise_rank: int,
    min_eig: float,
) -> Tuple[Array, Array]:
    """Parse process noise specification into low-rank (Uq, Sq).

    Supported forms:
    - Dense covariance matrix Q (d, d)
    - Cholesky-like factor L (d, r), interpreted as Q = L L^T
    - Tuple (Uq, Sq) where Q = Uq Sq Uq^T
      (Sq can be vector diag or dense core)
    """
    if isinstance(Q, tuple):
        Uq, Sq = Q
        Uq = jnp.asarray(Uq)
        Sq = jnp.asarray(Sq)
        if Sq.ndim == 1:
            vals = jnp.clip(Sq, min=min_eig)
            core_sqrt = jnp.diag(jnp.sqrt(vals))
        else:
            vals, vecs = _eigh_clip(Sq, min_eig=min_eig)
            core_sqrt = vecs @ jnp.diag(jnp.sqrt(vals))

        Lq = Uq @ core_sqrt
        return _compress_from_chol_like(Lq, rank=process_noise_rank, min_eig=min_eig)

    Q = jnp.asarray(Q)
    if Q.ndim == 2 and Q.shape[0] != Q.shape[1]:
        return _compress_from_chol_like(Q, rank=process_noise_rank, min_eig=min_eig)

    return _matrix_to_lowrank(Q, rank=process_noise_rank, min_eig=min_eig)


def rank_reduced_kalman_filter(
    model: LinearGaussianModel,
    rank: int,
    *,
    linear_solve: Optional[Callable] = None,
    min_eig: float = 1e-9,
    process_noise_rank: Optional[int] = None,
    energy_threshold: Optional[float] = None,
    min_rank: int = 1,
) -> Kernel:
    """Kalman filter with the covariance kept at low rank, ``P ~= U S U^T``.

    The predict step combines the propagated covariance and the process noise
    in factored form and truncates to ``rank``; the update works in the
    current subspace. Costs scale with ``rank`` instead of the state size.

    Args:
        model: As for :func:`kalman_filter`. The initial covariance and ``Q``
            may also be given as a factor ``L`` (``L L^T``) or a pair
            ``(U, S)`` (``U S U^T``, ``S`` a matrix or its diagonal).
        rank: Rank ``r`` of the covariance factor.
        linear_solve: Solver for the innovation covariance.
        min_eig: Floor for the kept eigenvalues.
        process_noise_rank: Rank kept of ``Q``; defaults to ``rank``.
        energy_threshold: Keep only the leading directions holding this
            fraction of the variance (at least ``min_rank``).
        min_rank: Smallest effective rank under ``energy_threshold``.

    Returns:
        A :class:`~probjax.inference.Kernel` like :func:`kalman_filter`'s;
        its state carries ``cov_factor`` and ``cov_core``.
    """
    if process_noise_rank is None:
        process_noise_rank = rank
    if energy_threshold is not None and not (0.0 < energy_threshold <= 1.0):
        raise ValueError("energy_threshold must be in (0, 1].")
    min_rank = max(1, min(min_rank, rank))
    solve = default_solve if linear_solve is None else linear_solve

    def truncate(vals_desc):
        return _apply_energy_threshold(
            vals_desc,
            min_eig=min_eig,
            energy_threshold=energy_threshold,
            min_rank=min_rank,
        )

    def init(
        key: Any, theta: Any, t0: ArrayLike = 0.0
    ) -> RankReducedKalmanFilterState:
        del key
        mean, cov = model.initial(theta)
        mean = jnp.asarray(mean)
        factor, core = _parse_process_noise(
            cov, process_noise_rank=rank, min_eig=min_eig
        )
        return RankReducedKalmanFilterState(
            mean, factor, core, as_time(t0), jnp.zeros((), mean.dtype)
        )

    def step(
        key: Any,
        state: RankReducedKalmanFilterState,
        theta: Any,
        t: ArrayLike,
        y: ArrayLike,
        observed: Any = True,
    ) -> Tuple[RankReducedKalmanFilterState, RankReducedKalmanFilterInfo]:
        del key
        t = jnp.asarray(t, state.t.dtype)
        Phi, Q = model.transition(theta, state.t, t)
        Uq, Sq = _parse_process_noise(
            Q, process_noise_rank=process_noise_rank, min_eig=min_eig
        )
        # P_pred = Phi U S U^T Phi^T + Uq Sq Uq^T, compressed to the rank.
        U_pred, S_pred = _sum_lowrank_terms(
            Phi @ state.cov_factor, state.cov_core, Uq, Sq, rank=rank, min_eig=min_eig
        )
        S_pred = jnp.diag(truncate(jnp.diag(S_pred)))
        mean_pred = Phi @ state.mean
        y = unobserved(observed, jnp.asarray(y))

        def update(_):
            C, R = model.observation(theta, t)
            residual = y - C @ mean_pred
            # Innovation S_y = C U S U^T C^T + R and gain K = U S M^T S_y^-1.
            M = C @ U_pred
            innovation = M @ S_pred @ M.T + R
            B_Sinv = solve(innovation, S_pred @ M.T)
            mean = mean_pred + U_pred @ B_Sinv @ residual
            S_post = symmetrize_matrix(S_pred - B_Sinv @ (M @ S_pred))
            # Re-diagonalise in the subspace and truncate again.
            vals, vecs = _eigh_clip(S_post, min_eig=min_eig)
            k = min(rank, vals.shape[0])
            vals, vecs = vals[-k:], vecs[:, -k:]
            vals = truncate(vals[::-1])[::-1]
            increment = innovation_loglik(
                residual,
                innovation,
                solve_fn=solve,
                logdet_fn=lambda S: jnp.linalg.slogdet(jnp.asarray(S))[1],
            )
            return (
                mean,
                U_pred @ vecs,
                jnp.diag(vals),
                increment.astype(state.log_likelihood.dtype),
            )

        def predict_only(_):
            return mean_pred, U_pred, S_pred, jnp.zeros_like(state.log_likelihood)

        mean, U, S, increment = jax.lax.cond(observed, update, predict_only, None)
        new_state = RankReducedKalmanFilterState(
            mean, U, S, t, state.log_likelihood + increment
        )
        info = RankReducedKalmanFilterInfo(mean_pred, U_pred, S_pred, increment)
        return new_state, info

    return Kernel(init, step, lambda state, theta=None: theta)
