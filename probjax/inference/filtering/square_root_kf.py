"""The square-root Kalman filter: a Kalman filter on covariance factors.

The covariance is carried as a factor ``S`` with ``P = S S^T`` and updated by QR
decompositions, which keeps it positive semi-definite in finite precision where
the plain Kalman update can lose symmetry or definiteness.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from probjax.inference.base import Kernel
from probjax.inference.filtering.gaussian import as_time, unobserved
from probjax.inference.filtering.state_space import LinearGaussianModel
from probjax.utils.linear_operator import as_array
from probjax.utils.typing import Array, ArrayLike


class SquareRootFilterState(NamedTuple):
    """The filtering distribution ``N(mean, std std^T)`` at time ``t``."""

    mean: Array
    std: Array
    t: Array
    log_likelihood: Array


class SquareRootFilterInfo(NamedTuple):
    mean_pred: Array
    std_pred: Array
    log_likelihood_increment: Array


def _lower_factor(blocks: Array) -> Array:
    """A lower-triangular ``L`` with ``L L^T = blocks blocks^T``."""
    return jnp.linalg.qr(blocks.T, mode="r").T


def sqrt_kf_predict(mean: Array, std: Array, Phi: Array, Q_std: Array):
    """Predict: ``P = Phi S S^T Phi^T + Lq Lq^T`` without forming ``P``."""
    return Phi @ mean, _lower_factor(jnp.concatenate([Phi @ std, Q_std], axis=1))


def sqrt_kf_correct(
    mean: Array, std: Array, C: Array, R_std: Array, y: Array
) -> Tuple[Array, Array, Array]:
    """Update by one QR of the array ``[[Lr, C S], [0, S]]``.

    The factor's blocks are the innovation factor ``Ls``, the scaled gain
    ``K Ls`` and the updated state factor.
    """
    num_obs, num_states = C.shape
    pre = jnp.block([
        [R_std, C @ std],
        [jnp.zeros((num_states, num_obs)), std],
    ])
    post = _lower_factor(pre)
    innovation_std = post[:num_obs, :num_obs]
    scaled_gain = post[num_obs:, :num_obs]
    std_new = post[num_obs:, num_obs:]
    residual = y - C @ mean
    whitened = jax.scipy.linalg.solve_triangular(innovation_std, residual, lower=True)
    log_likelihood = -0.5 * (
        whitened @ whitened
        + 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(innovation_std))))
        + num_obs * jnp.log(2 * jnp.pi)
    )
    return mean + scaled_gain @ whitened, std_new, log_likelihood


def square_root_kalman_filter(model: LinearGaussianModel) -> Kernel:
    """Square-root Kalman filter for a linear-Gaussian model.

    Takes the same model as :func:`kalman_filter`, with covariances; it factors
    them itself. The state carries ``std`` with ``cov = std @ std.T``.

    Returns:
        A :class:`~probjax.inference.Kernel` like :func:`kalman_filter`'s.
    """

    def init(key: Any, theta: Any, t0: ArrayLike = 0.0) -> SquareRootFilterState:
        del key
        mean, cov = model.initial(theta)
        mean = jnp.asarray(mean)
        return SquareRootFilterState(
            mean,
            jnp.linalg.cholesky(as_array(cov)),
            as_time(t0),
            jnp.zeros((), mean.dtype),
        )

    def step(
        key: Any,
        state: SquareRootFilterState,
        theta: Any,
        t: ArrayLike,
        y: ArrayLike,
        observed: Any = True,
    ) -> Tuple[SquareRootFilterState, SquareRootFilterInfo]:
        del key
        t = jnp.asarray(t, state.t.dtype)
        Phi, Q = model.transition(theta, state.t, t)
        mean_pred, std_pred = sqrt_kf_predict(
            state.mean,
            state.std,
            as_array(Phi),
            jnp.linalg.cholesky(as_array(Q)),
        )
        y = unobserved(observed, jnp.asarray(y))

        def update(_):
            C, R = model.observation(theta, t)
            mean, std, increment = sqrt_kf_correct(
                mean_pred,
                std_pred,
                as_array(C),
                jnp.linalg.cholesky(as_array(R)),
                y,
            )
            return mean, std, increment.astype(state.log_likelihood.dtype)

        def predict_only(_):
            return mean_pred, std_pred, jnp.zeros_like(state.log_likelihood)

        mean, std, increment = jax.lax.cond(observed, update, predict_only, None)
        new_state = SquareRootFilterState(
            mean, std, t, state.log_likelihood + increment
        )
        return new_state, SquareRootFilterInfo(mean_pred, std_pred, increment)

    return Kernel(init, step, lambda state, theta=None: theta)
