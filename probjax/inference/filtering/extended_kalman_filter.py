from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterAPI
from probjax.inference.filtering.kalman_filter import (
    KalmanFilterInfo,
    KalmanFilterState,
    _kalman_update,
    default_logdet,
    default_solve,
    init,
)
from probjax.utils.linalg import matrix_fraction_decomposition
from probjax.utils.linear_operator import LinearOperator

# ---------------------------------------------------------------------------
# EKF kernel
# ---------------------------------------------------------------------------


def build_kernel(
    transition_model_fn: Callable,
    observation_model_fn: Callable,
    linear_solve: Optional[Callable] = None,
    logdet_fn: Optional[Callable] = None,
) -> Callable:
    """Build an Extended Kalman filter kernel.

    Args:
        transition_model_fn: (x, cov, t_old, t) -> (x_pred, Phi, Q)
            Returns the nonlinear predicted state, the Jacobian of the
            transition, and the process noise covariance.
        observation_model_fn: (x, cov, t) -> (y_pred, C, R)
            Returns the nonlinear predicted observation, the Jacobian of the
            observation function, and the observation noise covariance.
        linear_solve: Optional custom solve function (see kalman_filter).
        logdet_fn: Optional custom logdet function (see kalman_filter).
    """

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

        # Predict: nonlinear mean + linearized covariance
        mu1_, Phi, Q = transition_model_fn(mu0, cov0, t_old, t)
        cov1_ = Phi @ cov0 @ Phi.T + Q
        # Materialize predicted covariance — needed for update step
        if isinstance(cov1_, LinearOperator):
            cov1_ = cov1_.as_array()

        if is_observed:
            y_, C, R = observation_model_fn(mu1_, cov1_, t)

            solve = default_solve if linear_solve is None else linear_solve
            _logdet_fn = logdet_fn if logdet_fn is not None else default_logdet

            mu1, cov1, log_likelihood = _kalman_update(
                mu1_, cov1_, y_, observed, C, R, solve, _logdet_fn
            )

            # Ensure symmetry (EKF linearization can introduce asymmetry)
            cov1 = 0.5 * (cov1 + cov1.T)

            return KalmanFilterState(mu1, cov1, t), KalmanFilterInfo(
                mu1_, cov1_, log_likelihood
            )
        else:
            return KalmanFilterState(mu1_, cov1_, t), KalmanFilterInfo(
                mu1_, cov1_, jnp.array(0.0)
            )

    return kernel


# ---------------------------------------------------------------------------
# Helpers for building transition / observation model functions
# ---------------------------------------------------------------------------


def _jacobian_as_linear_operator(fn, *fixed_args, in_dim, out_dim):
    """Wrap fn as a LinearOperator Jacobian without materializing it.

    Forward:  J @ v   via jax.jvp.
    Adjoint:  J.T @ w via jax.vjp.
    """

    def operator(v):
        _primals, tangents = jax.jvp(fn, fixed_args, (v,))
        return tangents

    return LinearOperator(operator, in_dim, out_dim)


def make_linearized_transition(
    transition_fn, Q_fn, in_dim, out_dim=None, materialize=False
):
    """Build an EKF transition model.

    By default the Jacobian Phi = df/dx is returned as a matrix-free
    LinearOperator. Set materialize=True to return a dense array instead
    (faster for small state dimensions, but O(n^2) memory).

    Args:
        transition_fn: f(x, t_old, t) -> x_new.  Nonlinear transition.
        Q_fn: Either a callable (t_old, t) -> Q returning the process noise
            covariance, or a fixed array / LinearOperator.
        in_dim: State dimension.
        out_dim: Output dimension. Defaults to in_dim.
        materialize: If True, return Phi as a dense array via jax.jacfwd.
            If False (default), return Phi as a matrix-free LinearOperator.

    Returns:
        transition_model_fn: (x, cov, t_old, t) -> (x_pred, Phi, Q)
    """
    if out_dim is None:
        out_dim = in_dim

    def transition_model_fn(x, cov, t_old, t):
        x_pred = transition_fn(x, t_old, t)
        if materialize:
            Phi = jax.jacfwd(transition_fn)(x, t_old, t)
        else:
            Phi = _jacobian_as_linear_operator(
                lambda x: transition_fn(x, t_old, t),
                x,
                in_dim=in_dim,
                out_dim=out_dim,
            )
        Q = Q_fn(t_old, t) if callable(Q_fn) else Q_fn
        return x_pred, Phi, Q

    return transition_model_fn


def make_linearized_observation(
    observation_fn, R_fn, in_dim, out_dim, materialize=False
):
    """Build an EKF observation model.

    By default the Jacobian C = dh/dx is returned as a matrix-free
    LinearOperator. Set materialize=True to return a dense array instead.

    Args:
        observation_fn: h(x, t) -> y.  Nonlinear observation.
        R_fn: Either a callable (t,) -> R returning the observation noise
            covariance, or a fixed array / LinearOperator.
        in_dim: State (input) dimension.
        out_dim: Observation (output) dimension.
        materialize: If True, return C as a dense array via jax.jacfwd.
            If False (default), return C as a matrix-free LinearOperator.

    Returns:
        observation_model_fn: (x, cov, t) -> (y_pred, C, R)
    """

    def observation_model_fn(x, cov, t):
        y_pred = observation_fn(x, t)
        if materialize:
            C = jax.jacfwd(observation_fn)(x, t)
        else:
            C = _jacobian_as_linear_operator(
                lambda x: observation_fn(x, t),
                x,
                in_dim=in_dim,
                out_dim=out_dim,
            )
        R = R_fn(t) if callable(R_fn) else R_fn
        return y_pred, C, R

    return observation_model_fn


def make_continuous_transition(drift_fn, diffusion_matrix, in_dim, materialize=False):
    """Build an EKF transition model from continuous SDE.

    For an SDE  dx = f(x,t) dt + B dw,  this computes:
      - x_pred = x + f(x, t_old) * (t - t_old)   (Euler step)
      - Phi, Q via matrix_fraction_decomposition

    Since matrix_fraction_decomposition must materialize the 2d x 2d block
    matrix for expm, Phi is always dense regardless of the materialize flag.
    The flag only controls whether the separate Jacobian A is kept dense
    or discarded (it's computed regardless for the decomposition).

    Args:
        drift_fn: f(x, t) -> dx/dt.  The drift function of the SDE.
        diffusion_matrix: B, the diffusion matrix (constant). Shape (d, m).
        in_dim: State dimension.
        materialize: Kept for API consistency — Phi is always dense here
            because expm requires dense matrices.

    Returns:
        transition_model_fn: (x, cov, t_old, t) -> (x_pred, Phi, Q)
    """
    B = diffusion_matrix

    def transition_model_fn(x, cov, t_old, t):
        dt = t - t_old
        x_pred = x + drift_fn(x, t_old) * dt

        # Dense Jacobian required for matrix_fraction_decomposition
        A = jax.jacfwd(lambda x: drift_fn(x, t_old))(x)
        Phi, Q = matrix_fraction_decomposition(t_old, t, A, B)
        return x_pred, Phi, Q

    return transition_model_fn


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class extended_kalman_filter(FilterAPI):
    r"""
    Extended Kalman filter for a nonlinear state space model.

    $$x_{t+1} = f(x_t, t) + w_t \qquad y_t = \mathcal{N}(y_t; h(x_t, t), R_t)$$

    The EKF linearizes the transition and observation functions around the current
    state estimate to apply the standard Kalman filter update equations.

    Args:
        transition_model_fn: (x, cov, t_old, t) -> (x_pred, Phi, Q)
            Returns the nonlinear predicted state, the Jacobian of the
            transition, and the process noise covariance.
        observation_model_fn: (x, cov, t) -> (y_pred, C, R)
            Returns the nonlinear predicted observation, the Jacobian of the
            observation function, and the observation noise covariance.

    Helpers for building these callables:
        - `make_linearized_transition(f, Q)`: wraps f(x,t_old,t)->x with auto-Jacobian
        - `make_linearized_observation(h, R)`: wraps h(x,t)->y with auto-Jacobian
        - `make_continuous_transition(drift, B)`: continuous SDE via matrix fraction decomposition
    """

    init = init
    build_kernel = build_kernel

    @staticmethod
    def default_unpack(state, info):
        return (state.mean, state.cov)
