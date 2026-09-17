from typing import Any, Callable, Optional, Sequence, Tuple

from jax import Array

from probjax.utils._solver_common import scan_on_grid
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

    def body(i, carry):
        dt = dts[i]
        new_state, _ = solver.step(carry, dt, *args)
        return new_state

    def scan_fun(scan_state, dt):
        scan_state, info = solver.step(scan_state, dt, *args)
        if trace_filter is None:
            traced_value = scan_state.y0
        else:
            traced_value = trace_filter(scan_state.y0, (scan_state, info))
        return scan_state, traced_value

    return scan_on_grid(
        scan_fun,
        body,
        state,
        dts,
        dts.shape[0],
        check_points=check_points,
        unroll=unroll,
        _split_transpose=_split_transpose,
        collect_trace=collect_trace,
    )
