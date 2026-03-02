from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterAPI


class UnscentedKalmanFilterState(NamedTuple):
    mean: ArrayLike
    cov: ArrayLike
    t: Optional[ArrayLike]


class UnscentedKalmanFilterInfo(NamedTuple):
    mean_pred: ArrayLike
    cov_pred: ArrayLike
    log_likelihood: Optional[ArrayLike]


def merwe_sigma_point(
    mu0: ArrayLike,
    cov0: ArrayLike,
    alpha: ArrayLike = 1.0,
    beta: ArrayLike = 2.0,
    kappa: ArrayLike = 0.0,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
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
    sigma_points_1_L = mu0 + sqrt_cov
    sigma_points_L_2L = mu0 - sqrt_cov
    sigma_points = jnp.concatenate(
        [sigma_points_0, sigma_points_1_L, sigma_points_L_2L], axis=0
    )

    return sigma_points, weights_mean, weights_cov


def julier_uhlmann_sigma_points(
    mu0: jnp.ndarray,
    cov0: jnp.ndarray,
    kappa: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    sigma_points_pos = mu0 + sqrt_cov
    sigma_points_neg = mu0 - sqrt_cov
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
    mu0: jnp.ndarray, cov0: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    sqrt_val = jnp.sqrt(D + 1)

    # Create a (D+1, D) array filled with -√(D+1)
    base_points = jnp.full((D + 1, D), -sqrt_val)

    # Create row and column indices
    rows = jnp.arange(D + 1)[:, None]  # Shape (D+1,1)
    cols = jnp.arange(D)[None, :]  # Shape (1,D)

    # Set a diagonal pattern: for (i+1, i), set to +√(D+1)
    mask = (rows - 1) == cols
    base_points = jnp.where(mask, sqrt_val, base_points)

    # Ensure the points have zero mean
    base_points = base_points - jnp.mean(base_points, axis=0, keepdims=True)

    # Scale to achieve unit covariance before transforming by cov0
    scale = jnp.sqrt(D / (2 * (D + 1)))
    base_points = base_points * scale

    # Apply the covariance transformation
    A = jnp.linalg.cholesky(cov0)
    sigma_points = mu0[None, :] + base_points @ A

    # Equal weights for mean and covariance
    weights_mean = jnp.ones(D + 1) / (D + 1)
    weights_cov = jnp.ones(D + 1) / (D + 1)

    return sigma_points, weights_mean, weights_cov


def unscented_transform(
    sigma_points: ArrayLike,
    weights_mean: ArrayLike,
    weights_cov: ArrayLike,
    noise_cov: Optional[ArrayLike] = None,
    mean_fn: Optional[Callable] = None,
    cov_fn: Optional[Callable] = None,
) -> Tuple[ArrayLike, ArrayLike]:
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


def init(
    mu0: ArrayLike, cov0: ArrayLike, t: Optional[float | int] = None
) -> UnscentedKalmanFilterState:
    """Initialize the unscented Kalman filter.

    Args:
        mu0 (ArrayLike): Initial mean of the state
        cov0 (ArrayLike): Initial covariance of the state
        t (Optional[float | int], optional): Time. Defaults to None.

    Returns:
        UnscentedKalmanFilterState: Initial state of the unscented Kalman filter
    """
    return UnscentedKalmanFilterState(mu0, cov0, t)


def build_kernel(
    transition_fn: Callable,
    transition_covariance_matrix: Callable | ArrayLike,
    observation_fn: Callable,
    observation_covariance: Callable | ArrayLike,
    sigma_point_fn: Callable = merwe_sigma_point,
) -> Callable:
    """Build an unscented Kalman filter kernel.

    Args:
        transition_fn (Callable): General transition function f(x_t, t, t+1) -> x_{t+1}
        transition_covariance_matrix (Callable | ArrayLike): Transition covariance
        matrix Q(t, t+1) or Q observation_fn (Callable): General observation function
        h(x_t, t) -> y_t  observation_covariance (Callable | ArrayLike): Observation
        covariance matrix R(t) or R
        sigma_point_fn (Callable, optional): How to generate sigma points. Defaults to
        merwe_sigma_point.

    Returns:
        Callable: Unscented Kalman filter step
    """

    def kernel(
        state: UnscentedKalmanFilterState,
        t: Optional[float | int] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[jnp.ndarray] = None,
    ) -> Tuple[UnscentedKalmanFilterState, UnscentedKalmanFilterInfo]:
        """One step of the unscented Kalman filter.

        Args:
            state (UnscentedKalmanFilterState): Mean and covariance of the state
            t (Optional[float  |  int], optional): Time. Defaults to None.
            observed (Optional[ArrayLike], optional): Observation. Defaults to None.
            rng_key (Optional[jnp.ndarray], optional): Random generator key.
                Defaults to None.

        Returns:
            Tuple[UnscentedKalmanFilterState, UnscentedKalmanFilterInfo]: _description_
        """

        mu0 = state.mean
        cov0 = state.cov
        t_old = state.t
        is_observed = observed is not None

        sigma_points, weights_mean, weights_cov = sigma_point_fn(mu0, cov0)
        predicted_sigma_points = jax.vmap(transition_fn, in_axes=(0, None, None))(
            sigma_points, t_old, t
        )
        if isinstance(transition_covariance_matrix, Callable):
            Q = transition_covariance_matrix(t_old, t)
        else:
            Q = transition_covariance_matrix

        # Prediction step
        mu1_, cov1_ = unscented_transform(
            predicted_sigma_points, weights_mean, weights_cov, noise_cov=Q
        )

        if is_observed:
            # Observed steps
            y_sigma_points = jax.vmap(observation_fn, in_axes=(0, None))(
                predicted_sigma_points, t
            )
            mu_y, cov_y = unscented_transform(
                y_sigma_points,
                weights_mean,
                weights_cov,
                noise_cov=observation_covariance,
            )

            # Compute the cross-covariance
            r_state = predicted_sigma_points - mu1_[None, :]
            r_obs = y_sigma_points - mu_y[None, :]
            cross_cov = jnp.dot(weights_cov * r_state.T, r_obs)

            # Compute the Kalman gain
            K = jnp.linalg.solve(cov_y.T, cross_cov.T).T

            # Compute the updated mean and covariance
            r = observed - mu_y
            mu1 = mu1_ + jnp.dot(K, r)
            cov1 = cov1_ - jnp.dot(K, jnp.dot(cov_y, K.T))

            # Compute the log-likelihood
            log_likelihood = -0.5 * (
                jnp.linalg.slogdet(cov_y)[1] + r.T @ jnp.linalg.solve(cov_y, r)
            )

            return UnscentedKalmanFilterState(mu1, cov1, t), UnscentedKalmanFilterInfo(
                mu1_, cov1_, log_likelihood
            )
        else:
            log_likelihood = jnp.array(0.0)
            return UnscentedKalmanFilterState(
                mu1_, cov1_, t
            ), UnscentedKalmanFilterInfo(mu1_, cov1_, log_likelihood)

    return kernel


class ukf(FilterAPI):
    """Unscented Kalman filter inference algorithm.

    This class implements the unscented Kalman filter algorithm. The unscented Kalman
    filter is a generalization of the Kalman filter to non-linear and non-Gaussian
    models.

    To build an unscented Kalman filter, you need to provide the following functions:
    Args:
        transition_fn (Callable): Transition function f(x_t, t) -> x_{t+1}
        transition_covariance_matrix (Callable | ArrayLike): Transition covariance
            matrix Q(t) or Q
        observation_fn (Callable): Observation function h(x_t, t) -> y_t
        observation_covariance (Callable | ArrayLike): Observation covariance matrix
            R(t) or R
    """

    init = init
    build_kernel = build_kernel

    @staticmethod
    def default_unpack(state, info):
        return (state.mean, state.cov)
