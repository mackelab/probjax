from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterInfo, FilterKernel, FilterState
from probjax.inference.filtering.kalman_filter import kalman_filter
from probjax.inference.filtering.particle_filter import ParticleFilter
from probjax.utils.jaxutils import nested_checkpoint_scan


def filter(
    key: PRNGKey,
    ts: ArrayLike,
    t_o: Optional[ArrayLike],
    x_o: Optional[ArrayLike],
    kernel: FilterKernel,
    *args,
    unpack_fn: Optional[Callable] = None,
    checkpoint_lengths: Optional[Sequence[int]] = None,
    unroll: int = 1,
    **kwargs,
):
    inital_state = kernel.init(*args, t=ts[0], **kwargs)

    if unpack_fn is None:
        unpack_fn = get_default_unpack_fn(kernel)

    def scan_fn(carry, t):
        state, key, i = carry
        key, subkey = jax.random.split(key)
        is_observed = t == t_o[i]

        def update_fn(subkey, state, i):
            state, info = kernel(state, t=t_o[i], observed=x_o[i], rng=subkey)
            return state, info, i + 1

        def predict_fn(subkey, state, i):
            state, info = kernel(state, t=t_o[i], rng=subkey)
            return state, info, i

        state, info, i = jax.lax.cond(
            is_observed, update_fn, predict_fn, subkey, state, i
        )
        out = unpack_fn(state, info)
        return (state, key, i), out

    carry = (inital_state, key, 0)

    if checkpoint_lengths is None:
        _, output = jax.lax.scan(scan_fn, carry, ts[1:], unroll=unroll)

    else:
        _, output = nested_checkpoint_scan(
            scan_fn, carry, ts[1:], nested_lengths=checkpoint_lengths, unroll=unroll
        )
        # output = jax.tree_util.tree_map(lambda x: jnp.concatenate([inital_output, x]), output)

    return output


def smooth(
    ts: Array, mus: Array, covs: Array, mus_: Array, covs_: Array, smooth: Callable
) -> Tuple[Array, Array]:
    """Smooths the state given a Kalman filter output.

    Args:
        ts (Array): Time grid
        mus (Array): Means
        covs (Array): Covs
        mus_ (Array): Predicted means
        covs_ (Array): Predicted covs
        smooth (Callable): Smoothing function

    Returns:
        Tuple[Array, Array]: _description_
    """

    idx_last = jnp.where((mus != mus_).all(-1))[-1][-1]
    mus_needed_ = jnp.flip(mus_[1 : idx_last + 1])
    covs_needed_ = jnp.flip(covs_[1 : idx_last + 1])
    mus_needed = jnp.flip(mus[:idx_last])
    covs_needed = jnp.flip(covs[:idx_last])
    ts_needed = jnp.flip(ts[:idx_last])

    def scan_fun(carry, data):
        (mu0_s, cov0_s, t1) = carry
        t0, mu0, cov0, mu0_, cov0_ = data
        mu1, cov1 = smooth(t0, t1, mu0_s, cov0_s, mu0, cov0, mu0_, cov0_)
        return (mu1, cov1, t0), (mu1, cov1)

    init_carry = (mus[idx_last], covs[idx_last], ts[idx_last])
    _, (mus_s, covs_s) = jax.lax.scan(
        scan_fun,
        init_carry,
        (ts_needed, mus_needed, covs_needed, mus_needed_, covs_needed_),
    )

    mus = jnp.concatenate([mus_s[::-1], mus[idx_last:]])
    covs = jnp.concatenate([covs_s[::-1], covs[idx_last:]])

    return mus, covs


def filter_log_likelihood(
    key, ts, t_o, x_o, kernel, *args, checkpoint_lengths=None, unroll=1, **kwargs
):
    output = filter(
        key,
        ts,
        t_o,
        x_o,
        kernel,
        *args,
        unpack_fn=unpack_loglikeliood,
        checkpoint_lengths=checkpoint_lengths,
        unroll=unroll,
        **kwargs,
    )
    ll = output.sum()
    return ll


def get_default_unpack_fn(kernel: FilterKernel):
    if isinstance(kernel, ParticleFilter):
        return lambda state, info: state.particles
    elif type(kernel) is kalman_filter:
        return lambda state, info: (state.mean, state.cov)
    else:
        return lambda state, info: (state, info)


def unpack_loglikeliood(state: FilterState, info: FilterInfo):
    if info is not None and hasattr(info, "log_likelihood"):
        return info.log_likelihood
    else:
        return jnp.array(0.0)
