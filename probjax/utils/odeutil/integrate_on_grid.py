from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
from jax import Array

from probjax.utils.jaxutils import nested_checkpoint_scan
from probjax.utils.odeutil.solvers import ODESolverAPI, ODEState


def _odeint_on_grid(
    method: ODESolverAPI,
    drift: Callable,
    y0: Array,
    ts: Array,
    *args,
    trace_filter: Optional[Callable] = None,
    check_points: Optional[Sequence[int]] = None,
    unroll: int | bool = False,
    _split_transpose: bool = False,
    collect_trace: bool = True,
) -> Tuple[ODEState, Any]:
    # Time steps
    solver = method(drift)  # type: ignore
    dts = ts[1:] - ts[:-1]

    t0 = ts[0]
    state = solver.init(t0, y0, *args)
    if not collect_trace:
        def body(i, carry):
            dt = dts[i]
            new_state, _ = solver.step(carry, dt, *args)
            return new_state

        state = jax.lax.fori_loop(0, dts.shape[0], body, state)
        traced = None
    else:
        def scan_fun(scan_state, dt):
            scan_state, info = solver.step(scan_state, dt, *args)
            if trace_filter is None:
                traced_value = scan_state.y0
            else:
                traced_value = trace_filter(scan_state.y0, (scan_state, info))
            return scan_state, traced_value

        if check_points is None:
            state, traced = jax.lax.scan(
                scan_fun, state, dts, unroll=unroll, _split_transpose=_split_transpose
            )
        else:
            state, traced = nested_checkpoint_scan(
                scan_fun,
                state,
                dts,
                nested_lengths=check_points,
                scan_fn=partial(
                    jax.lax.scan, unroll=unroll, _split_transpose=_split_transpose
                ),
            )

    return state, traced
