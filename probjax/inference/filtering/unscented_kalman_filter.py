"""The unscented Kalman filter and its sigma-point rules."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from probjax.inference.base import Kernel
from probjax.inference.filtering.gaussian import (
    GaussianFilterInfo,
    GaussianFilterState,
    as_time,
    innovation_loglik,
    unobserved,
)
from probjax.inference.filtering.state_space import GaussianModel
from probjax.utils.typing import Array, ArrayLike


def merwe_sigma_point(
    mu0: Array,
    cov0: Array,
    alpha: ArrayLike = 1.0,
    beta: ArrayLike = 2.0,
    kappa: ArrayLike = 0.0,
) -> Tuple[Array, Array, Array]:
    """Generate sigma points for unscented Kalman filter, see [1].

    Args:
        mu0 (ArrayLike): Mean of the state
        cov0 (ArrayLike): Covariance of the state
        alpha (ArrayLike, optional): Determines the spread around the mean
            (small values lead to large weight which require high precission (64 bit)).
            Large values will spread the sigma points further, typically letting
            to an overestimation of the covariance, small values will lead to an
            underestimation. Literature suggests 1e-3 but this requires 64 bit
            precision to work at all... . Defaults to 1..
        beta (ArrayLike, optional): Prior on covariance. Defaults to 2..
        kappa (ArrayLike, optional): Additional parameter. Defaults to 0..

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Sigma points, weights_mean, weights_cov

    References:
        [1] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
            Inference in Dynamic State-Space Models" (Doctoral dissertation)
    """

    with jax.ensure_compile_time_eval():
        D = mu0.shape[0]
        lambda_ = alpha**2 * (D + kappa) - D
        weights_mean_0 = jnp.atleast_1d(lambda_ / (D + lambda_))
        weights_cov_0 = jnp.atleast_1d(weights_mean_0 + (1 - alpha**2 + beta))
        weights_mean_cov_1_L = 1 / (2 * (D + lambda_)) * jnp.ones(2 * D)

        weights_mean = jnp.concatenate([weights_mean_0, weights_mean_cov_1_L], axis=0)
        weights_cov = jnp.concatenate([weights_cov_0, weights_mean_cov_1_L], axis=0)

    sqrt_cov = jnp.linalg.cholesky((D + lambda_) * cov0)
    sigma_points_0 = mu0[None, :]
    sigma_points_1_L = mu0 + sqrt_cov.T
    sigma_points_L_2L = mu0 - sqrt_cov.T
    sigma_points = jnp.concatenate(
        [sigma_points_0, sigma_points_1_L, sigma_points_L_2L], axis=0
    )

    return sigma_points, weights_mean, weights_cov


def julier_uhlmann_sigma_points(
    mu0: Array,
    cov0: Array,
    kappa: float = 0.0,
) -> Tuple[Array, Array, Array]:
    """Generate Julier-Uhlmann sigma points.

    Args:
        mu0 (jnp.ndarray): Mean of the state (D, )
        cov0 (jnp.ndarray): Covariance of the state (D, D)
        kappa (float): Spread parameter.
            Often chosen as kappa = 3 - D for state dimension D.

    Returns:
        Tuple[sigma_points, weights_mean, weights_cov]:
            sigma_points: (2D+1, D)
            weights_mean: (2D+1,)
            weights_cov: (2D+1,)
    """
    D = mu0.shape[0]
    lambda_ = D + kappa
    # Compute sqrt of scaled covariance
    sqrt_cov = jnp.linalg.cholesky(lambda_ * cov0)

    # Sigma points
    sigma_points_0 = mu0[None, :]
    sigma_points_pos = mu0 + sqrt_cov.T
    sigma_points_neg = mu0 - sqrt_cov.T
    sigma_points = jnp.concatenate(
        [sigma_points_0, sigma_points_pos, sigma_points_neg], axis=0
    )

    # Weights
    w0 = kappa / (D + kappa)
    wi = 1.0 / (2.0 * (D + kappa)) * jnp.ones(2 * D)
    weights_mean = jnp.concatenate([jnp.array([w0]), wi], axis=0)
    weights_cov = jnp.concatenate([jnp.array([w0]), wi], axis=0)

    return sigma_points, weights_mean, weights_cov


def spherical_simplex_sigma_points(
    mu0: Array, cov0: Array
) -> Tuple[Array, Array, Array]:
    """
    Generate spherical simplex sigma points without explicit Python loops.

    This creates a (D+1) x D array of points that form a regular simplex.
    The steps are:
    1. Construct an initial (D+1, D) array corresponding to a simplex.
    2. Center the points so they have zero mean.
    3. Scale the points to achieve unit covariance.
    4. Apply the square root of the covariance (via Cholesky) and then add mu0.

    Args:
        mu0 (jnp.ndarray): Mean of the state, shape (D,)
        cov0 (jnp.ndarray): Covariance of the state, shape (D,D)

    Returns:
        sigma_points (jnp.ndarray): (D+1, D)
        weights_mean (jnp.ndarray): (D+1,)
        weights_cov (jnp.ndarray): (D+1,)
    """
    D = mu0.shape[0]
    # A Helmert basis spans the subspace orthogonal to the all-ones
    # vector. Its columns are orthonormal, so equal weights recover I.
    rows = jnp.arange(D + 1)[:, None]
    cols = jnp.arange(D)[None, :]
    denominator = jnp.sqrt((cols + 1) * (cols + 2))
    basis = jnp.where(rows <= cols, 1.0, jnp.where(rows == cols + 1, -(cols + 1), 0.0))
    points = jnp.sqrt(D + 1) * basis / denominator
    sigma_points = mu0[None, :] + points @ jnp.linalg.cholesky(cov0).T

    # Equal weights for mean and covariance
    weights_mean = jnp.ones(D + 1) / (D + 1)
    weights_cov = jnp.ones(D + 1) / (D + 1)

    return sigma_points, weights_mean, weights_cov


def unscented_transform(
    sigma_points: Array,
    weights_mean: Array,
    weights_cov: Array,
    noise_cov: Optional[ArrayLike] = None,
    mean_fn: Optional[Callable] = None,
    cov_fn: Optional[Callable] = None,
) -> Tuple[Array, Array]:
    """Unscented transform

    Args:
        sigma_points (ArrayLike): (Transformed) sigma points (2D+1, D)
        weights_mean (ArrayLike): Weights for the mean (2D+1,)
        weights_cov (ArrayLike): Weights for the covariance (2D+1,)
        noise_cov (Optional[ArrayLike], optional): Additive noise covariance (D,D).
            Defaults to None.
        mean_fn (Optional[Callable], optional): Custom mean_fn predictor
            mean_fn(simga_point, weight_mean). Defaults to None.
        cov_fn (Optional[Callable], optional): Custom cov_fn predict
            cov_fn(sigma_points, mean, weights_cov). Defaults to None.

    Returns:
        Tuple[NDArray, NDArray]: _description_
    """
    if mean_fn is not None:
        mean = mean_fn(sigma_points, weights_mean)
    else:
        mean = jnp.dot(weights_mean, sigma_points)

    if cov_fn is not None:
        cov = cov_fn(sigma_points, mean, weights_cov)
    else:
        diff = sigma_points - mean[None, :]
        cov = jnp.dot(weights_cov * diff.T, diff)

    if noise_cov is not None:
        cov += noise_cov

    return mean, cov


def unscented_kalman_filter(
    model: GaussianModel, *, sigma_points: Callable = merwe_sigma_point
) -> Kernel:
    """Unscented Kalman filter for a nonlinear model with Gaussian noise.

    Propagates sigma points through the transition and the observation
    instead of linearising them, so it needs no Jacobians.

    Args:
        model: The model, as functions of the parameters ``theta``.
        sigma_points: ``(mean, cov) -> (points, weights_mean, weights_cov)``,
            e.g. :func:`merwe_sigma_point`, :func:`julier_uhlmann_sigma_points`
            or :func:`spherical_simplex_sigma_points`.

    Returns:
        A :class:`~probjax.inference.Kernel` like :func:`kalman_filter`'s.
    """

    def init(key: Any, theta: Any, t0: ArrayLike = 0.0) -> GaussianFilterState:
        del key
        mean, cov = model.initial(theta)
        mean = jnp.asarray(mean)
        return GaussianFilterState(
            mean, jnp.asarray(cov), as_time(t0), jnp.zeros((), mean.dtype)
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
        points, weights_mean, weights_cov = sigma_points(state.mean, state.cov)
        moved = jax.vmap(lambda x: model.transition(theta, x, state.t, t)[0])(points)
        Q = model.transition(theta, state.mean, state.t, t)[1]
        mean_pred, cov_pred = unscented_transform(
            moved, weights_mean, weights_cov, noise_cov=Q
        )
        y = unobserved(observed, jnp.asarray(y))

        def update(_):
            # Fresh sigma points carry the process noise into the observation.
            points, weights_mean, weights_cov = sigma_points(mean_pred, cov_pred)
            observed_points = jax.vmap(lambda x: model.observation(theta, x, t)[0])(
                points
            )
            R = model.observation(theta, mean_pred, t)[1]
            mean_y, cov_y = unscented_transform(
                observed_points, weights_mean, weights_cov, noise_cov=R
            )
            cross_cov = jnp.dot(
                weights_cov * (points - mean_pred).T, observed_points - mean_y
            )
            gain = jnp.linalg.solve(cov_y.T, cross_cov.T).T
            residual = y - mean_y
            increment = innovation_loglik(
                residual,
                cov_y,
                solve_fn=lambda S, res: jnp.linalg.solve(S, res),
                logdet_fn=lambda S: jnp.linalg.slogdet(S)[1],
            )
            return (
                mean_pred + gain @ residual,
                cov_pred - gain @ cov_y @ gain.T,
                increment.astype(state.log_likelihood.dtype),
            )

        def predict_only(_):
            return mean_pred, cov_pred, jnp.zeros_like(state.log_likelihood)

        mean, cov, increment = jax.lax.cond(observed, update, predict_only, None)
        new_state = GaussianFilterState(mean, cov, t, state.log_likelihood + increment)
        return new_state, GaussianFilterInfo(mean_pred, cov_pred, increment)

    return Kernel(init, step, lambda state, theta=None: theta)
