from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterInfo, FilterKernel, FilterState
from probjax.inference.filtering.smoothing import (
    particle_smoother,
    smooth as smooth_gaussian,
)
from probjax.utils.jaxutils import nested_checkpoint_scan


class FilteringTrace(NamedTuple):
    ts: ArrayLike
    initial_state: FilterState
    states: Any
    infos: Any
    outputs: Any


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
    return_trace: bool = False,
    **kwargs,
):
    initial_state = kernel.init(*args, t=ts[0], **kwargs)

    if unpack_fn is None:
        unpack_fn = kernel.default_unpack

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
        return (state, key, i), (state, info, out)

    carry = (initial_state, key, 0)

    if checkpoint_lengths is None:
        _, (states, infos, output) = jax.lax.scan(scan_fn, carry, ts[1:], unroll=unroll)
    else:
        _, (states, infos, output) = nested_checkpoint_scan(
            scan_fn, carry, ts[1:], nested_lengths=checkpoint_lengths, unroll=unroll
        )

    if return_trace:
        return FilteringTrace(
            ts=ts,
            initial_state=initial_state,
            states=states,
            infos=infos,
            outputs=output,
        )

    return output


def _trace_ts(trace: FilteringTrace):
    num_states = jax.tree_util.tree_leaves(trace.states)[0].shape[0]
    if trace.ts.shape[0] == num_states + 1:
        return trace.ts[1:]
    return trace.ts


def smooth_from_states(
    trace: FilteringTrace,
    smoother: Callable,
):
    """Run Gaussian smoothing from filtering states + infos.

    This API aligns smoothing inputs with filtering outputs by consuming
    `FilteringTrace` directly.
    """
    ts = _trace_ts(trace)

    states = trace.states
    infos = trace.infos

    if hasattr(states, "cov"):
        covs = states.cov
    elif hasattr(states, "std"):
        covs = jax.vmap(lambda s: s @ s.T)(states.std)
    elif hasattr(states, "cov_factor") and hasattr(states, "cov_core"):
        covs = jax.vmap(lambda u, s: u @ s @ u.T)(states.cov_factor, states.cov_core)
    else:
        raise ValueError("Filtering states must have either `cov` or `std`.")

    mus = states.mean

    if hasattr(infos, "cov_pred"):
        covs_pred = infos.cov_pred
    elif hasattr(infos, "std_pred"):
        covs_pred = jax.vmap(lambda s: s @ s.T)(infos.std_pred)
    elif hasattr(infos, "cov_factor_pred") and hasattr(infos, "cov_core_pred"):
        covs_pred = jax.vmap(lambda u, s: u @ s @ u.T)(
            infos.cov_factor_pred, infos.cov_core_pred
        )
    else:
        raise ValueError("Filtering infos must have either `cov_pred` or `std_pred`.")

    mus_pred = infos.mean_pred
    return smooth_gaussian(ts, mus, covs, mus_pred, covs_pred, smoother)


def smooth_particle_from_states(
    key: PRNGKey,
    trace: FilteringTrace,
    transition_logdensity_fn: Callable,
):
    """Run particle smoothing from filtering states + infos."""
    ts = _trace_ts(trace)
    states = trace.states
    ancestors = trace.infos.ancestors if hasattr(trace.infos, "ancestors") else None

    return particle_smoother(
        key,
        ts,
        states.particles,
        states.log_weights,
        transition_logdensity_fn,
        ancestors=ancestors,
    )


def smooth(
    *args,
    smoother: Optional[Callable] = None,
    key: Optional[PRNGKey] = None,
    transition_logdensity_fn: Optional[Callable] = None,
    **kwargs,
):
    """Unified smoothing API consuming filtering states.

    - For Gaussian filters (KF/EKF/UKF/SqKF): pass `smoother`.
    - For particle filters: pass `key` and `transition_logdensity_fn`.
    """
    # Backward compatibility: old Gaussian smoother API
    # smooth(ts, mus, covs, mus_pred, covs_pred, smoother)
    if not args or not isinstance(args[0], FilteringTrace):
        return smooth_gaussian(*args, **kwargs)

    trace = args[0]
    states = trace.states
    if hasattr(states, "particles") and hasattr(states, "log_weights"):
        if key is None or transition_logdensity_fn is None:
            raise ValueError(
                "Particle smoothing requires `key` and `transition_logdensity_fn`."
            )
        return smooth_particle_from_states(key, trace, transition_logdensity_fn)

    if smoother is None:
        raise ValueError("Gaussian smoothing requires `smoother` callback.")
    return smooth_from_states(trace, smoother)


def filter_and_smooth(
    key: PRNGKey,
    ts: ArrayLike,
    t_o: Optional[ArrayLike],
    x_o: Optional[ArrayLike],
    kernel: FilterKernel,
    *args,
    smoother: Optional[Callable] = None,
    smooth_key: Optional[PRNGKey] = None,
    transition_logdensity_fn: Optional[Callable] = None,
    checkpoint_lengths: Optional[Sequence[int]] = None,
    unroll: int = 1,
    **kwargs,
):
    """Run filtering and smoothing in one call.

    This helper returns both the full filtering trace and the smoothing output.
    Smoothing mode is selected from the state type:

    - Gaussian states: provide `smoother`.
    - Particle states: provide `transition_logdensity_fn`; `smooth_key` defaults
      to the filtering key if omitted.
    """
    trace = filter(
        key,
        ts,
        t_o,
        x_o,
        kernel,
        *args,
        checkpoint_lengths=checkpoint_lengths,
        unroll=unroll,
        return_trace=True,
        **kwargs,
    )

    if hasattr(trace.states, "particles") and hasattr(trace.states, "log_weights"):
        if transition_logdensity_fn is None:
            raise ValueError("Particle smoothing requires `transition_logdensity_fn`.")
        if smooth_key is None:
            smooth_key = key
        smoothed = smooth(
            trace,
            key=smooth_key,
            transition_logdensity_fn=transition_logdensity_fn,
        )
    else:
        if smoother is None:
            raise ValueError("Gaussian smoothing requires `smoother` callback.")
        smoothed = smooth(trace, smoother=smoother)

    return trace, smoothed


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
        unpack_fn=unpack_log_likelihood,
        checkpoint_lengths=checkpoint_lengths,
        unroll=unroll,
        **kwargs,
    )
    ll = output.sum()
    return ll


def unpack_log_likelihood(state: FilterState, info: FilterInfo):
    if info is not None and hasattr(info, "log_likelihood"):
        return info.log_likelihood
    else:
        return jnp.array(0.0)
