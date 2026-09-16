"""Shared scaffolding for ODE/SDE solver implementations.

Both ``odeutil`` and ``sdeutil`` solvers repeat the same small patterns
(state init, Wiener sampling, grid integration, method registries). They
live here so the two families cannot drift apart. Import paths of the
existing ``odeutil.solvers.base`` / ``sdeutil.base`` modules are unchanged.
"""

from functools import partial

import jax
import jax.numpy as jnp

from probjax.utils.jaxutils import nested_checkpoint_scan


def make_trivial_init(state_cls):
    """State init for solvers whose state is just ``(t0, y0)``."""

    def init(t0, y0, *args, **kwargs):
        del args, kwargs
        return state_cls(t0=jnp.asarray(t0), y0=jnp.asarray(y0))

    return init


def sample_wiener_increment(rng, g0, default_dim, dt, configured_dim=None):
    """Sample a Wiener increment, inferring the noise dimension from ``g0``."""
    from probjax.utils.sdeutil.base import infer_noise_dim

    dim = infer_noise_dim(g0, default_dim, configured_dim)
    return jax.random.normal(rng, (dim,)) * jnp.sqrt(jnp.abs(dt))


def scan_on_grid(
    scan_fun,
    body_fn,
    init_state,
    xs,
    length,
    check_points=None,
    unroll=False,
    _split_transpose=False,
    collect_trace=True,
):
    """Run a grid integration: ``fori_loop`` or (checkpointed) ``scan``.

    ``scan_fun(state, x)`` and ``body_fn(i, carry)`` are solver-specific;
    this helper only owns the ``collect_trace``/``check_points`` branching.
    """
    if not collect_trace:
        state = jax.lax.fori_loop(0, length, body_fn, init_state)
        return state, None
    if check_points is None:
        return jax.lax.scan(
            scan_fun, init_state, xs, unroll=unroll, _split_transpose=_split_transpose
        )
    return nested_checkpoint_scan(
        scan_fun,
        init_state,
        xs,
        nested_lengths=check_points,
        scan_fn=partial(jax.lax.scan, unroll=unroll, _split_transpose=_split_transpose),
    )


def ensure_dtype(ts, y0, dtype):
    """Cast ``ts``/``y0`` to ``dtype`` and normalize ``ts`` to 1D.

    Shared preamble of ``odeutil.core._odeint`` and ``sdeutil.core._sdeint``.
    """
    if dtype is not None:
        ts = ts.astype(dtype)
        y0 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=dtype), y0)
    return jnp.atleast_1d(ts), y0


def make_filter_wrapper(filter_state):
    """Wrap an optional state filter into ``apply_filter``."""

    def apply_filter(tree):
        if filter_state is None:
            return tree
        return filter_state(tree)

    return apply_filter


def stack_trace(init_filtered, ys):
    """Prepend the filtered initial state to a per-step trace."""

    def _stack(init_leaf, trace_leaf):
        return jnp.concatenate(
            [jnp.asarray(init_leaf)[None], jnp.asarray(trace_leaf)], axis=0
        )

    return jax.tree_util.tree_map(_stack, init_filtered, ys)
