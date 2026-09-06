from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Key

from probjax.utils._solver_common import scan_on_grid
from probjax.utils.sdeutil import SDEState


def _sdeint_on_grid(
    method: Callable,
    drift: Callable,
    diffusion: Callable,
    rng: Key,
    y0: ArrayLike,
    ts: ArrayLike,
    filter_output: Optional[Callable] = None,
    check_points: Optional[Sequence[int]] = None,
    unroll: int | bool = False,
    _split_transpose: bool = False,
    collect_trace: bool = True,
) -> Tuple[SDEState, Any]:
    # Time steps
    ts = jnp.asarray(ts)
    solver = method(drift, diffusion)
    dts = ts[1:] - ts[:-1]

    def scan_fun(state, dt_key):
        dt, key = dt_key
        state, info = solver.step(key, state, dt)
        if filter_output is None:
            return state, state.y0
        else:
            return state, filter_output(state, info)

    t0 = ts[0]
    state = solver.init(t0, y0)
    keys = jax.random.split(rng, dts.shape[0])

    def body(i, carry):
        dt = dts[i]
        key = keys[i]
        new_state, _ = solver.step(key, carry, dt)
        return new_state

    return scan_on_grid(
        scan_fun,
        body,
        state,
        (dts, keys),
        dts.shape[0],
        check_points=check_points,
        unroll=unroll,
        _split_transpose=_split_transpose,
        collect_trace=collect_trace,
    )
