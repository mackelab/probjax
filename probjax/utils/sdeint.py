from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Key, PyTree

from probjax.utils.functions import generic_drift
from probjax.utils.sdeutil.adaptive import SDEStepSizeAdaptor
from probjax.utils.sdeutil.core import _sdeint
from probjax.utils.sdeutil.integrate_adaptive import warn_boundary_hits


def _wrap_if_plain_callable(
    fn: Callable[..., PyTree[Array]],
) -> Callable[..., PyTree[Array]]:
    """Wrap plain Python callables as :class:`generic_drift` so they flow as
    pytrees through ``jax.jit``. Marker drifts/diffusions and other
    pytree-registered callables flow through unchanged.
    """
    leaves, _ = jax.tree_util.tree_flatten(fn)
    if len(leaves) == 1 and leaves[0] is fn:
        return generic_drift(fn=fn)
    return fn


def sdeint(
    rng: Key,
    drift: Callable[..., PyTree[Array]],
    diffusion: Callable[..., PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    *args: Any,
    method: str = "euler_maruyama",
    dtype: Optional[jnp.dtype] = jnp.float32,
    sde_type: str = "ito",
    return_brownian: bool = False,
    return_state: bool = False,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    step_size_adaptor: Optional[SDEStepSizeAdaptor] = None,
) -> Union[
    Optional[PyTree[Array]],
    Tuple[Any, Optional[PyTree[Array]]],
    Tuple[
        Optional[PyTree[Array]],
        Optional[PyTree[Array]],
    ],
    Tuple[
        Any,
        Tuple[Optional[PyTree[Array]], Optional[PyTree[Array]]],
    ],
]:
    """Solve a stochastic differential equation
    ``dy = drift(t, y, *args) dt + diffusion(t, y, *args) dW``.

    ``drift`` and ``diffusion`` may be plain Python callables (automatically
    wrapped as :class:`~probjax.utils.functions.generic_drift`) or any
    pytree-registered callable (marker
    :class:`~probjax.utils.functions.Drift` subclass, ``eqx.Module``, etc.).

    Keyword arguments to drift/diffusion are no longer supported. Pass
    parameters positionally via ``*args``, or bind them with
    ``functools.partial`` / ``drift.bind_args(...)``.

    Args:
        rng: Random number generator key.
        drift: Drift function ``f(t, y, *args)`` — deterministic part of
            ``dy = f(t, y, *args) dt + g(t, y, *args) dW_t``.
        diffusion: Diffusion function ``g(t, y, *args)`` — stochastic part.
        y0: Initial state. Single array or pytree of arrays.
        ts: Strictly increasing 1D array of time points.
        *args: Positional arguments forwarded to both drift and diffusion.
        method: Integration method name. Available:

            - ``"euler_maruyama"``: Euler-Maruyama (order 0.5)
            - ``"exp_euler_maruyama"``: Exponential Euler-Maruyama
              (requires :class:`~probjax.utils.functions.split_drift`)
            - ``"milstein"``: Milstein (order 1.0)
            - ``"srk"``: Stochastic Runge-Kutta methods
        dtype: Computation dtype (default ``float32``).
        sde_type: ``"ito"`` (default) or ``"stratonovich"``. Noise layout
            is inferred from the diffusion output shape — scalar/vector
            outputs are diagonal, matrix outputs are full (including
            rectangular).
        return_brownian: Whether to return Brownian paths (requires
            ``collect_trace=True``).
        return_state: Whether to return solver state.
        filter_state: Optional state filter; returning ``None`` disables
            tracing entirely.
        collect_trace: Record the filtered quantity at every time step
            (``True``, default) or return only the filtered terminal state
            (``False``). Must be ``True`` when returning Brownian paths.
        check_points: Optional index sequence for checkpointed grid
            integration.
        step_size_adaptor: When provided, switches integration to an
            adaptive step-doubling Euler-Maruyama scheme driven by the
            given controller. Pass
            :class:`~probjax.utils.sdeutil.adaptive.WeakStepSizeAdaptor`
            for distributional / weak quantities (resamples noise on
            rejection — cheaper) or
            :class:`~probjax.utils.sdeutil.adaptive.StrongStepSizeAdaptor`
            for per-path consistency (refines a single Brownian path via
            a virtual tree). Currently restricted to **diagonal noise**
            and ignores ``method`` (always uses Euler-Maruyama with
            step-doubling error estimation). Defaults to ``None``
            (fixed-step integration via ``method``).

    Returns:
        When ``return_brownian=False``: the filtered trajectory (if
        ``collect_trace=True``) or the filtered terminal state. When
        ``return_brownian=True``: ``(state_trace, brownian_trace)``. If
        ``return_state=True``, the solver state is prepended.

    Example:
        >>> import jax.numpy as jnp
        >>> from probjax.utils.sdeint import sdeint
        >>> from jax import random
        >>>
        >>> def drift(t, state, mu, sigma):
        ...     return {"price": mu * state["price"]}
        >>> def diffusion(t, state, mu, sigma):
        ...     return {"price": sigma * state["price"]}
        >>>
        >>> rng = random.PRNGKey(0)
        >>> y0 = {"price": jnp.array([1.0])}
        >>> ts = jnp.linspace(0, 1, 100)
        >>> ys = sdeint(rng, drift, diffusion, y0, ts, 0.1, 0.2)
    """
    drift = _wrap_if_plain_callable(drift)
    diffusion = _wrap_if_plain_callable(diffusion)
    result = _sdeint(
        rng,
        drift,
        diffusion,
        y0,
        ts,
        tuple(args),
        method=method,
        dtype=dtype,
        sde_type=sde_type,
        return_brownian=return_brownian,
        return_state=return_state,
        filter_state=filter_state,
        collect_trace=collect_trace,
        check_points=check_points,
        step_size_adaptor=step_size_adaptor,
    )
    # ``_sdeint`` appends an adaptive-diagnostic hit count as the last
    # element of its return so we can warn host-side (no per-vmap-element
    # callback overhead). Strip it before returning to the user; warn iff
    # the controller ran out of budget on a meaningful fraction of the
    # output segments.
    if return_state:
        state_obj, payload, diag_hits = result
        out = (state_obj, payload)
    else:
        payload, diag_hits = result
        out = payload
    # Host-side boundary warning. Skip when a tracer flows through (under
    # ``jax.vmap`` / ``jax.jit`` of ``sdeint`` itself); the warning will
    # surface from the outermost concrete invocation. ``jax.core.Tracer``
    # check costs nothing under vmap and avoids the per-element callback
    # dispatch we'd pay if the warn lived inside the JIT graph.
    if step_size_adaptor is not None and not isinstance(diag_hits, jax.core.Tracer):
        n_segments = int(jnp.atleast_1d(jnp.asarray(ts)).shape[0]) - 1
        warn_boundary_hits(step_size_adaptor, diag_hits, n_segments)
    return out
