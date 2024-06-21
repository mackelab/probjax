from typing import Optional, Sequence, Callable
import jax 
import jax.numpy as jnp

from jax.typing import ArrayLike
from jax.random import PRNGKey
from probjax.utils.jaxutils import nested_checkpoint_scan
from probjax.inference.filtering.base import FilterState, FilterInfo, FilterKernel
from probjax.inference.filtering.particle_filter import ParticleFilter


def filter(key: PRNGKey, ts: ArrayLike, t_o: Optional[ArrayLike], x_o: Optional[ArrayLike], kernel: FilterKernel, *args, unpack_fn : Optional[Callable]=None, checkpoint_lengths: Optional[Sequence[int]]=None, unroll:int=1, **kwargs):

    state = kernel.init(*args, **kwargs)
    if unpack_fn is None:
        unpack_fn = get_default_unpack_fn(kernel)

    def scan_fn(carry, t):
        state, key, i = carry
        key, subkey = jax.random.split(key)
        is_observed = t == t_o[i]
        def update_fn(subkey, state, i):
            state, info = kernel(state, t_o[i], observed=x_o[i], rng_key=subkey)
            return state,info, i+1

        def predict_fn(subkey, state, i):
            state, info = kernel(state, t_o[i], rng_key=subkey)
            return state, info, i

        state, info, i = jax.lax.cond(is_observed, update_fn, predict_fn,subkey, state, i)
        out = unpack_fn(state, info)
        return (state, key, i), out

    carry = (state, key, 0)

    if checkpoint_lengths is None:
        _, output = jax.lax.scan(scan_fn, carry, ts, unroll=unroll)
    else:
        _, output = nested_checkpoint_scan(scan_fn, carry, ts, nested_lengths=checkpoint_lengths, unroll=unroll)

    return output


def filter_log_likelihood(key, ts, t_o, x_o, kernel, *args, checkpoint_lengths=None, unroll=1, **kwargs):
    output = filter(key, ts, t_o, x_o, kernel, *args, unpack_fn=unpack_loglikeliood, checkpoint_lengths=checkpoint_lengths, unroll=unroll, **kwargs)
    ll = output.sum()
    return ll


def get_default_unpack_fn(kernel: FilterKernel):

    if isinstance(kernel, ParticleFilter):
        return lambda state, info: state.particles
    else:
        return lambda state, info: (state, info)

def unpack_loglikeliood(state: FilterState, info: FilterInfo):
    return info.logZ
