from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterInfo, FilterKernel, FilterState
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
            state, info = kernel(state, t=t_o[i], observed=x_o[i], rng_key=subkey)
            return state, info, i + 1

        def predict_fn(subkey, state, i):
            state, info = kernel(state, t=t_o[i], rng_key=subkey)
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
        output = jax.tree_map(lambda x: jnp.concatenate([inital_output, x]), output)

    inital_output = unpack_fn(inital_state, None)
    output = jax.tree_map(
        lambda init_x, x: jnp.concatenate([init_x[None, ...], x]), inital_output, output
    )
    return output


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
    else:
        return lambda state, info: (state, info)


def unpack_loglikeliood(state: FilterState, info: FilterInfo):
    if info is not None and hasattr(info, "log_likelihood"):
        return info.log_likelihood
    else:
        return jnp.array(0.0)
