import jax
import jax.numpy as jnp
from jax import lax
from jax import core
import jax.random as jrandom

from jaxtyping import Array, Float, PyTree
from jax.random import PRNGKeyArray


@jax.jit
def brownian_bridge(
    key: PRNGKeyArray, t: Float, t0: Float, t1: Float, w0: Array, w1: Array
) -> Array:
    """Brownian bridge between two points.

    Args:
        key (PRNGKeyArray): Random generator key.
        t (Float): Time at which to sample.
        t0 (Float): Time at which the bridge starts.
        t1 (Float): Time at which the bridge ends.
        w0 (Array): Value of the bridge at t0.
        w1 (Array): Value of the bridge at t1.

    Returns:
        Array: Value of the bridge at t.
    """
    length = t1 - t0
    dist_end = t1 - t
    dist_start = t - t0
    mean = (dist_end * w0 + dist_start * w1) / length
    std = jnp.sqrt(dist_end * dist_start / length)
    shape = mean.shape

    return mean + std * jrandom.normal(key, shape)


@jax.jit
def brownian_path(key, ts):
    diffs = ts[1:] - ts[:-1]

    ws = jrandom.normal(key, (len(ts) - 1,)) * jnp.sqrt(diffs)
    return jnp.cumsum(jnp.concatenate([jnp.zeros(1), ws], axis=0), axis=0)

@jax.jit
def brownian_tree(
    key: PRNGKeyArray, t: Float, t0: Float, t1: Float, w0: Array, tol: Float
) -> Array:
    """Brownian motion between two points using a tree. This allows to evaluate it at any time, without having to save the whole trajectory.

    Args:
        key (PRNGKeyArray): Random generator key.
        t (Float): Time at which to sample.
        t0 (Float): Start time.
        t1 (Float): End time.
        w0 (Array): Start value.
        tol (Float): Tolerance for the tree.

    Returns:
        Array: Value of the bridge at t.
    """
    key, init_key = jrandom.split(key, 2)
    shape = w0.shape

    t_half = t0 + 0.5 * (t1 - t0)
    w1 = jrandom.normal(init_key, shape) * jnp.sqrt(t1 - t0)
    w_half = brownian_bridge(key, t_half, t0, t1, w0, w1)

    init_state = (t0, t_half, t1, w0, w_half, w1, key)

    def cond_fun(state):
        start_time, _, end_time, _, _, _, _ = state
        return (end_time - start_time) > tol

    def body_fun(state):
        t0, t_half, t1, w0, w_half, w1, key = state

        _key1, _key2 = jrandom.split(key, 2)
        _cond = t > t_half
        _s = jnp.where(_cond, t_half, t0)
        _u = jnp.where(_cond, t1, t_half)
        _w_s = jnp.where(_cond, w_half, w0)
        _w_u = jnp.where(_cond, w1, w_half)
        _key = jnp.where(_cond, _key1, _key2)

        _t = _s + 0.5 * (_u - _s)
        _w_t = brownian_bridge(_key, _t, _s, _u, _w_s, _w_u)

        return (_s, _t, _u, _w_s, _w_t, _w_u, key)

    t0, t_half, t1, w0, w_half, w1, key = lax.while_loop(cond_fun, body_fun, init_state)

    rescale_t = (t - t0) / (t1 - t0)
    A = jnp.array([[2, -4, 2], [-3, 4, -1], [1, 0, 0]])
    coeffs = jnp.tensordot(A, jnp.stack([w0, w_half, w1]), axes=1)
    return jnp.polyval(coeffs, rescale_t)
