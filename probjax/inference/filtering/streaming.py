"""Persistent fixed-capacity histories shared by temporal filters and parameter SMC."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from probjax.inference.filtering.temporal import TemporalTrace


class StreamingWindow(NamedTuple):
    initial_state: Any
    states: Any
    infos: Any
    ts: Any
    count: Any


def init_streaming_window(state, info, capacity):
    """Allocate a ring buffer using one step's info (or an abstract info prototype).

    Can be carried alongside the state through arbitrary streaming chunks. The
    initial state must precede the first appended state. All time points must be
    chronological; failed outer SMC steps must not be appended.
    """
    if not isinstance(capacity, int) or capacity < 1:
        raise ValueError('capacity must be a positive integer.')
    allocate = lambda x: jnp.zeros((capacity,) + x.shape, x.dtype)
    return StreamingWindow(
        state,
        jax.tree.map(allocate, state),
        jax.tree.map(allocate, info),
        jnp.zeros(capacity, dtype=state.t.dtype),
        jnp.array(0),
    )


def append_streaming_window(window, state, info):
    """Append one state/diagnostic pair in O(capacity-independent) indexed writes."""
    size = window.ts.shape[0]
    slot = window.count % size
    old = jax.tree.map(lambda x: x[slot], window.states)
    boundary = jax.lax.cond(
        window.count >= size, lambda: old, lambda: window.initial_state
    )
    return StreamingWindow(
        boundary,
        jax.tree.map(lambda x, v: x.at[slot].set(v), window.states, state),
        jax.tree.map(lambda x, v: x.at[slot].set(v), window.infos, info),
        window.ts.at[slot].set(state.t),
        window.count + 1,
    )


def streaming_window_trace(window):
    """Return (fixed-shape trace, validity mask) in chronological order.

    Before capacity is filled, invalid entries are trailing zero padding; use the
    mask for plotting/reductions. Smoothers require an unpadded trace: on the host,
    slice by mask.sum(), or wait until the window is full before compiled smoothing.
    """
    size = window.ts.shape[0]
    count = jnp.minimum(window.count, size)
    start = jnp.where(window.count >= size, window.count % size, 0)
    order = (start + jnp.arange(size)) % size
    return TemporalTrace(
        window.initial_state,
        window.ts[order],
        jax.tree.map(lambda x: x[order], window.states),
        jax.tree.map(lambda x: x[order], window.infos),
    ), jnp.arange(size) < count
