import math
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import lax
from jax.random import PRNGKey
from jaxtyping import Array, Float

# Iterated integrals


@jax.jit
def iterated_ito_integral_general(key: PRNGKey, dW: Array, dt: Array, n: int = 5):
    """Matrix I approximating repeated Ito integrals based on the method of Kloeden,
    Platen and Wright (1992).

    Args:
        key (PRNGKey): PRNGKey
        dW (Array): Wiener increments
        dt (Array): Time step size
        n (int, optional): Truncation of Fourier series. Defaults to 5.

    Returns:
        (Array, Array): Matrix of Ito integrals and Levy areas.


    NOTE: Based on https://github.com/mattja/sdeint/blob/master/sdeint/wiener.py#L102
    """

    dW = jnp.atleast_1d(dW)
    m = dW.shape[0]

    sqrt2h = jnp.sqrt(2.0 / dt)

    def body_fun(i, val):
        key, A0 = val
        next_key, key1, key2 = jax.random.split(key, 3)
        Xk = jax.random.normal(key1, shape=(m,))
        Yk = jax.random.normal(key2, shape=(m,))
        term1 = jnp.outer(Xk, (Yk + sqrt2h * dW))
        term2 = jnp.outer(Yk + sqrt2h * dW, Xk)
        A1 = A0 + (term1 - term2) / i

        return (next_key, A1)

    A0 = jnp.zeros((m, m))
    n = jax.lax.cond(m == 1, lambda _: 0, lambda _: n, None)  # No iteration for 1D
    init_val = (key, A0)
    _, A1 = jax.lax.fori_loop(1, n + 1, body_fun, init_val)

    A1 = (dt / (2.0 * jnp.pi)) * A1
    I = 0.5 * (jnp.outer(dW, dW) - dt * jnp.eye(m)) + A1  # noqa: E741

    return I, A1


def iterated_stratowich_integral_general(
    key: PRNGKey, dW: Array, dt: Array, n: int = 5
):
    """Matrix I approximating repeated Stratonovich integrals based on the method of
    Kloeden, Platen and Wright (1992)."""
    I, A = iterated_ito_integral_general(key, dW, dt, n)  # noqa: E741
    J = I + 0.5 * dt * jnp.eye(dW.shape[0])
    return J, A


def iterated_stochastic_integral_diagonal(key: PRNGKey, dW: Array, dt: Array, **kwargs):
    I_diag = 0.5 * (jnp.square(dW) - dt)
    return I_diag


def iterated_stochastic_integral_commutative_noise(
    key: PRNGKey, dW: Array, dt: Array, **kwargs
):
    I = jnp.outer(dW, dW) - dt * jnp.eye(dW.shape[0])  # noqa: E741
    return I


def get_iterated_integrals_fn(noise_type: str, sde_type: str):
    """Returns the iterated integrals function for a given noise type and sde type."""
    if noise_type == "diagonal":
        return iterated_stochastic_integral_diagonal
    elif noise_type == "commutative":
        return iterated_stochastic_integral_commutative_noise
    elif noise_type == "general":
        if sde_type == "ito":
            return lambda *args, **kwargs: iterated_ito_integral_general(
                *args, **kwargs
            )[0]
        elif sde_type == "stratonovich":
            return lambda *args, **kwargs: iterated_stratowich_integral_general(
                *args, **kwargs
            )[0]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


# Brownian bridge and tree


def brownian_path(key, x0, ts):
    shape = (ts.shape[0] - 1,) + x0.shape
    xs = x0[None, :] + jnp.cumsum(
        jnp.sqrt(ts[1] - ts[0]) * jax.random.normal(key, shape), axis=0
    )
    return jnp.concatenate([x0[None, :], xs], axis=0)


@jax.jit
def brownian_bridge(
    key: PRNGKey, t: Float, t0: Float, t1: Float, w0: Array, w1: Array
) -> Array:
    """Brownian bridge between two points.

    Args:
        key (PRNGKey): Random generator key.
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


def _depth_from_tol(tol: float) -> int:
    """Pick a static refinement depth so leaf width on a unit interval ≤ tol."""
    return max(1, int(math.ceil(-math.log2(float(tol)))))


@partial(jax.jit, static_argnames=("tol", "depth"))
def brownian_tree(
    key: PRNGKey,
    t: Float,
    t0: Float,
    t1: Float,
    w0: Array,
    tol: Optional[float] = None,
    *,
    depth: Optional[int] = None,
) -> Array:
    """Brownian motion at arbitrary ``t`` via a virtual tree.

    Refines a Brownian-bridge tree to a fixed (compile-time) ``depth`` so the
    inner loop is :func:`lax.fori_loop`, which vectorizes cleanly under
    :func:`jax.vmap` and supports reverse-mode differentiation. The
    historical ``tol`` parameter is accepted for backward compatibility and
    converted to ``depth = ceil(-log2(tol))`` at compile time.

    Args:
        key: Master key — defines the underlying Brownian path.
        t: Query time, ``t0 ≤ t ≤ t1``.
        t0, t1, w0: Tree-root interval and starting value.
        tol: Static target leaf width; converted to ``depth``. Mutually
            exclusive with ``depth``.
        depth: Static refinement depth. Per-query cost is ``O(depth)``.
            Default ``16`` (≈ 1.5e-5 leaf width on a unit interval) — fine
            enough for any controller-driven step above ``T·2^-16``. Pass a
            tighter value (e.g. ``20``) for stiff or very-tight-tolerance
            runs.

    Returns:
        ``W(t)`` for the Brownian path keyed by ``key``.
    """
    if depth is None:
        depth = _depth_from_tol(tol) if tol is not None else 16

    key, init_key = jrandom.split(key, 2)
    shape = w0.shape

    t_half = t0 + 0.5 * (t1 - t0)
    w1 = jrandom.normal(init_key, shape) * jnp.sqrt(t1 - t0)
    w_half = brownian_bridge(key, t_half, t0, t1, w0, w1)

    def body(_, state):
        s0, sh, s1, ws0, wsh, ws1 = state
        k1, k2 = jrandom.split(key, 2)
        cond = t > sh
        s = jnp.where(cond, sh, s0)
        u = jnp.where(cond, s1, sh)
        w_s = jnp.where(cond, wsh, ws0)
        w_u = jnp.where(cond, ws1, wsh)
        kk = jnp.where(cond, k1, k2)
        new_t = s + 0.5 * (u - s)
        new_w = brownian_bridge(kk, new_t, s, u, w_s, w_u)
        return (s, new_t, u, w_s, new_w, w_u)

    init_state = (t0, t_half, t1, w0, w_half, w1)
    t0, t_half, t1, w0, w_half, w1 = lax.fori_loop(0, depth, body, init_state)

    rescale_t = (t - t0) / (t1 - t0)
    A = jnp.array([[2, -4, 2], [-3, 4, -1], [1, 0, 0]])
    coeffs = jnp.tensordot(A, jnp.stack([w0, w_half, w1]), axes=1)
    return jnp.polyval(coeffs, rescale_t)


# Estimate weak and strong error
