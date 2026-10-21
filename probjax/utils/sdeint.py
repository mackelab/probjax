from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import jax.numpy as jnp
from jax import Array
from jaxtyping import Key, PyTree

from probjax.utils._solver_common import (
    ConfiguredSolver,
    SDEIntAux,
    solver_or_legacy_method,
)
from probjax.utils.functions import wrap_if_plain_callable
from probjax.utils.guards import is_tracing
from probjax.utils.sdeutil.adaptive import SDEStepSizeAdaptor
from probjax.utils.sdeutil.base import is_adaptive_solver
from probjax.utils.sdeutil.core import _sdeint
from probjax.utils.sdeutil.integrate_adaptive import warn_boundary_hits


@overload
def sdeint(
    rng: Key,
    drift: Callable[..., PyTree[Array]],
    diffusion: Callable[..., PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    *args: Any,
    solver: Union[str, type, ConfiguredSolver] = "euler_maruyama",
    method: Optional[str] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    sde_type: str = "ito",
    filter_state: Optional[Callable[[PyTree[Array]], PyTree[Array]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    step_size_adaptor: Optional[SDEStepSizeAdaptor] = None,
    return_aux: Literal[False] = False,
) -> PyTree[Array]: ...


@overload
def sdeint(
    rng: Key,
    drift: Callable[..., PyTree[Array]],
    diffusion: Callable[..., PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    *args: Any,
    solver: Union[str, type, ConfiguredSolver] = "euler_maruyama",
    method: Optional[str] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    sde_type: str = "ito",
    filter_state: Optional[Callable[[PyTree[Array]], PyTree[Array]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    step_size_adaptor: Optional[SDEStepSizeAdaptor] = None,
    return_aux: Literal[True],
) -> Tuple[PyTree[Array], SDEIntAux]: ...


def sdeint(
    rng: Key,
    drift: Callable[..., PyTree[Array]],
    diffusion: Callable[..., PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    *args: Any,
    solver: Union[str, type, ConfiguredSolver] = "euler_maruyama",
    method: Optional[str] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    sde_type: str = "ito",
    filter_state: Optional[Callable[[PyTree[Array]], PyTree[Array]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    step_size_adaptor: Optional[SDEStepSizeAdaptor] = None,
    return_aux: bool = False,
) -> Union[PyTree[Array], Tuple[PyTree[Array], SDEIntAux]]:
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
        solver: Integration method: a registered name, a solver class or a
            configured solver (``milstein(...)``). Registered names include:

            - ``"euler_maruyama"``: Euler-Maruyama (order 0.5)
            - ``"exp_euler_maruyama"``: Exponential Euler-Maruyama
              (requires :class:`~probjax.utils.functions.split_drift`)
            - ``"milstein"``: Milstein (order 1.0)
            - ``"sri1"``, ``"sri2"``: Stochastic Runge-Kutta methods
              (strong order 1.0); ``"srk"`` is an alias of ``"sri1"``
            - ``"linear_exact_sde"``: exact transition sampling for linear SDEs
              (requires :class:`~probjax.utils.functions.linear_drift` and
              :class:`~probjax.utils.functions.const_diffusion`)
            - ``"adaptive"``: step-doubling Euler-Maruyama driven by
              ``step_size_adaptor``
        method: Deprecated alias of ``solver``.
        dtype: Computation dtype (default ``float32``).
        sde_type: ``"ito"`` (default) or ``"stratonovich"``. Noise layout
            is inferred from the diffusion output shape — scalar/vector
            outputs are diagonal, matrix outputs are full (including
            rectangular).
        filter_state: Optional function selecting the state components to
            record. It must return a value; use ``collect_trace=False`` to
            record nothing at all.
        collect_trace: Record the filtered quantity at every time step
            (``True``, default) or return only the filtered terminal state
            (``False``).
        check_points: Optional index sequence for checkpointed grid
            integration.
        step_size_adaptor: Step-size controller for the adaptive integrator
            (``solver="adaptive"``). Pass
            :class:`~probjax.utils.sdeutil.adaptive.WeakStepSizeAdaptor`
            for distributional / weak quantities (fresh noise from the key)
            or :class:`~probjax.utils.sdeutil.adaptive.StrongStepSizeAdaptor`
            for per-path consistency (increments from a single virtual
            Brownian tree, so runs at different tolerances are comparable
            per path). Steps are never rejected on their own noise; the
            local error only sets the next step size. Currently restricted to
            **diagonal noise**. Defaults to ``None`` (fixed-step integration
            via ``solver``).
        return_aux: Return ``(value, aux)`` instead of the bare value, where
            ``aux`` is an :class:`~probjax.utils._solver_common.SDEIntAux`
            holding the Brownian path and the solver's final state. The
            Brownian path is ``None`` on the adaptive path and for
            non-diagonal noise.

    Returns:
        The filtered trajectory (when ``collect_trace=True``) or the filtered
        terminal state, or ``(value, aux)`` when ``return_aux=True``.

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
    solver = solver_or_legacy_method(solver, method, "sdeint", default="euler_maruyama")
    if step_size_adaptor is None and is_adaptive_solver(solver):
        step_size_adaptor = SDEStepSizeAdaptor()
    drift = wrap_if_plain_callable(drift)
    diffusion = wrap_if_plain_callable(diffusion)
    value, aux, diag_hits = _sdeint(
        rng,
        drift,
        diffusion,
        y0,
        ts,
        *args,
        method=solver,
        dtype=dtype,
        sde_type=sde_type,
        return_aux=return_aux,
        filter_state=filter_state,
        collect_trace=collect_trace,
        check_points=check_points,
        step_size_adaptor=step_size_adaptor,
    )
    # Host-side boundary warning. Skip when a tracer flows through (under
    # ``jax.vmap`` / ``jax.jit`` of ``sdeint`` itself); the warning will
    # surface from the outermost concrete invocation. ``jax.core.Tracer``
    # check costs nothing under vmap and avoids the per-element callback
    # dispatch we'd pay if the warn lived inside the JIT graph.
    if step_size_adaptor is not None and not is_tracing(diag_hits):
        n_segments = int(jnp.atleast_1d(jnp.asarray(ts)).shape[0]) - 1
        warn_boundary_hits(step_size_adaptor, diag_hits, n_segments)
    if return_aux:
        return value, aux
    return value
