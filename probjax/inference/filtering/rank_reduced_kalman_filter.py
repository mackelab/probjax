from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterAPI
from probjax.utils.linalg import symmetrize_matrix


class RankReducedKalmanFilterState(NamedTuple):
    mean: ArrayLike
    cov_factor: ArrayLike
    cov_core: ArrayLike
    t: Optional[ArrayLike]


class RankReducedKalmanFilterInfo(NamedTuple):
    mean_pred: ArrayLike
    cov_factor_pred: ArrayLike
    cov_core_pred: ArrayLike
    log_likelihood: Optional[ArrayLike] = None


def _default_solve(S: ArrayLike, res: ArrayLike) -> ArrayLike:
    S = jnp.asarray(S)
    res = jnp.asarray(res)
    return jax.scipy.linalg.solve(S, res.T, assume_a="pos").T


def _factor_to_cov(cov_factor: ArrayLike, cov_core: ArrayLike) -> ArrayLike:
    return cov_factor @ cov_core @ cov_factor.T


def _eigh_clip(cov: ArrayLike, min_eig: float) -> Tuple[ArrayLike, ArrayLike]:
    vals, vecs = jnp.linalg.eigh(symmetrize_matrix(jnp.asarray(cov)))
    vals = jnp.clip(vals, min=min_eig)
    return vals, vecs


def _matrix_to_lowrank(
    cov: ArrayLike,
    rank: Optional[int],
    min_eig: float,
) -> Tuple[ArrayLike, ArrayLike]:
    vals, vecs = _eigh_clip(cov, min_eig=min_eig)
    k = vals.shape[0] if rank is None else min(rank, vals.shape[0])
    vals = vals[-k:]
    vecs = vecs[:, -k:]
    return vecs, jnp.diag(vals)


def _core_sqrt(core: ArrayLike, min_eig: float) -> ArrayLike:
    vals, vecs = _eigh_clip(core, min_eig=min_eig)
    return vecs @ jnp.diag(jnp.sqrt(vals))


def _compress_from_chol_like(
    L: ArrayLike,
    rank: int,
    min_eig: float,
) -> Tuple[ArrayLike, ArrayLike]:
    # P = L L^T; use SVD of L to get low-rank eigendecomposition.
    U, sing, _ = jnp.linalg.svd(L, full_matrices=False)
    k = min(rank, sing.shape[0])
    U = U[:, :k]
    vals = jnp.clip(sing[:k] ** 2, min=min_eig)
    return U, jnp.diag(vals)


def _effective_rank_from_energy(
    vals_desc: ArrayLike,
    energy_threshold: float,
    min_rank: int,
) -> int:
    total = jnp.sum(vals_desc)
    total = jnp.where(total <= 0.0, 1.0, total)
    cumsum = jnp.cumsum(vals_desc)
    ratios = cumsum / total
    k = jnp.argmax(ratios >= energy_threshold) + 1
    k = jnp.maximum(k, min_rank)
    k = jnp.minimum(k, vals_desc.shape[0])
    return k


def _apply_energy_threshold(
    vals_desc: ArrayLike,
    min_eig: float,
    energy_threshold: Optional[float],
    min_rank: int,
) -> ArrayLike:
    vals_desc = jnp.clip(vals_desc, min=min_eig)
    if energy_threshold is None:
        return vals_desc
    k = _effective_rank_from_energy(vals_desc, energy_threshold, min_rank)
    idx = jnp.arange(vals_desc.shape[0])
    keep = idx < k
    return jnp.where(keep, vals_desc, min_eig)


def _sum_lowrank_terms(
    factor_a: ArrayLike,
    core_a: ArrayLike,
    factor_b: ArrayLike,
    core_b: ArrayLike,
    rank: int,
    min_eig: float,
) -> Tuple[ArrayLike, ArrayLike]:
    # A + B = (La La^T) + (Lb Lb^T) with
    # La = factor_a @ sqrt(core_a), Lb = factor_b @ sqrt(core_b)
    La = factor_a @ _core_sqrt(core_a, min_eig=min_eig)
    Lb = factor_b @ _core_sqrt(core_b, min_eig=min_eig)
    L = jnp.concatenate([La, Lb], axis=1)
    return _compress_from_chol_like(L, rank=rank, min_eig=min_eig)


def _parse_process_noise(
    Q: ArrayLike | Tuple[ArrayLike, ArrayLike],
    process_noise_rank: int,
    min_eig: float,
) -> Tuple[ArrayLike, ArrayLike]:
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


def init(
    mean: ArrayLike,
    cov: ArrayLike | Tuple[ArrayLike, ArrayLike],
    t: Optional[ArrayLike] = None,
    rank: Optional[int] = None,
    min_eig: float = 1e-9,
) -> RankReducedKalmanFilterState:
    if rank is None:
        rank = jnp.asarray(mean).shape[0]

    cov_factor, cov_core = _parse_process_noise(
        cov,
        process_noise_rank=rank,
        min_eig=min_eig,
    )
    return RankReducedKalmanFilterState(mean, cov_factor, cov_core, t)


def build_kernel(
    transition_model_fns: Callable,
    observation_model_fns: Callable,
    rank: int,
    linear_solve: Optional[Callable] = None,
    min_eig: float = 1e-9,
    process_noise_rank: Optional[int] = None,
    energy_threshold: Optional[float] = None,
    min_rank: int = 1,
) -> Callable:
    """Build a rank-reduced Kalman filter kernel.

    Covariance is represented as P ~= U S U^T with U in R^{d x r}, S in R^{r x r}.
    Predict step is performed in low-rank form by combining propagated state
    covariance and process covariance through a compressed factorization.
    Update step uses low-rank algebra in the current subspace.
    """

    if process_noise_rank is None:
        process_noise_rank = rank

    if energy_threshold is not None and not (0.0 < energy_threshold <= 1.0):
        raise ValueError("energy_threshold must be in (0, 1].")
    min_rank = max(1, min(min_rank, rank))

    def kernel(
        state: RankReducedKalmanFilterState,
        t: Optional[ArrayLike] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[jnp.ndarray] = None,
    ) -> Tuple[RankReducedKalmanFilterState, RankReducedKalmanFilterInfo]:
        del rng_key

        mu0 = state.mean
        U0 = state.cov_factor
        S0 = state.cov_core
        t_old = state.t
        is_observed = observed is not None

        transition = transition_model_fns(t_old, t)
        if isinstance(transition, tuple) and len(transition) == 3:
            Phi, Uq, Sq = transition
            Uq, Sq = _parse_process_noise(
                (Uq, Sq),
                process_noise_rank=process_noise_rank,
                min_eig=min_eig,
            )
        else:
            Phi, Q = transition
            Uq, Sq = _parse_process_noise(
                Q,
                process_noise_rank=process_noise_rank,
                min_eig=min_eig,
            )

        # Predict in low-rank form: P_pred = Phi U0 S0 U0^T Phi^T + Uq Sq Uq^T
        Up = Phi @ U0
        U_pred, S_pred = _sum_lowrank_terms(
            Up,
            S0,
            Uq,
            Sq,
            rank=rank,
            min_eig=min_eig,
        )
        pred_vals = jnp.diag(S_pred)
        pred_vals = _apply_energy_threshold(
            pred_vals,
            min_eig=min_eig,
            energy_threshold=energy_threshold,
            min_rank=min_rank,
        )
        S_pred = jnp.diag(pred_vals)
        mu_pred = Phi @ mu0

        if is_observed:
            C, R = observation_model_fns(t)
            y = observed
            y_pred = C @ mu_pred
            residual = y - y_pred

            # Innovation: S_y = C U S U^T C^T + R
            M = C @ U_pred
            innovation = M @ S_pred @ M.T + R

            # K = U S M^T S_y^{-1}
            B = S_pred @ M.T
            solve = _default_solve if linear_solve is None else linear_solve
            B_Sinv = solve(innovation, B)
            K = U_pred @ B_Sinv

            mu = mu_pred + K @ residual

            # Covariance update in reduced coordinates:
            # S_post = S - S M^T S_y^{-1} M S
            S_post = symmetrize_matrix(S_pred - B_Sinv @ (M @ S_pred))

            # Re-orthogonalize/truncate in-subspace if needed
            vals, vecs = _eigh_clip(S_post, min_eig=min_eig)
            k = min(rank, vals.shape[0])
            vals = vals[-k:]
            vecs = vecs[:, -k:]
            vals_desc = vals[::-1]
            vals_desc = _apply_energy_threshold(
                vals_desc,
                min_eig=min_eig,
                energy_threshold=energy_threshold,
                min_rank=min_rank,
            )
            vals = vals_desc[::-1]
            U = U_pred @ vecs
            S = jnp.diag(vals)

            logdet = jnp.linalg.slogdet(jnp.asarray(innovation))[1]
            log_likelihood = -0.5 * (logdet + residual.T @ solve(innovation, residual))

            return RankReducedKalmanFilterState(
                mu, U, S, t
            ), RankReducedKalmanFilterInfo(
                mu_pred,
                U_pred,
                S_pred,
                log_likelihood,
            )

        return RankReducedKalmanFilterState(
            mu_pred, U_pred, S_pred, t
        ), RankReducedKalmanFilterInfo(mu_pred, U_pred, S_pred, jnp.array(0.0))

    return kernel


class rank_reduced_kalman_filter(FilterAPI):
    """Rank-reduced Kalman filter.

    Stores covariance in low-rank form P ~= U S U^T and truncates to a fixed rank.
    """

    init = init
    build_kernel = build_kernel

    @staticmethod
    def default_unpack(state, info):
        return (state.mean, _factor_to_cov(state.cov_factor, state.cov_core))
