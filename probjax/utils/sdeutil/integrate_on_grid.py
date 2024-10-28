from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
from jax.typing import ArrayLike
from jaxtyping import Key

from probjax.utils.jaxutils import nested_checkpoint_scan
from probjax.utils.sdeutil import SDESolver, SDEState


def _sdeint_on_grid(
    method: SDESolver,
    drift: Callable,
    diffusion: Callable,
    rng: Key,
    y0: ArrayLike,
    ts: ArrayLike,
    *args,
    filter_output: Optional[Callable] = None,
    check_points: Optional[Sequence[int]] = None,
    unroll: int | bool = False,
    _split_transpose: bool = False,
) -> Tuple[SDEState, Any]:
    # Time steps
    solver = method(drift, diffusion)
    dts = ts[1:] - ts[:-1]

    def scan_fun(state, dt_key):
        dt, key = dt_key
        state, info = solver.step(key, state, dt, *args)
        if filter_output is None:
            return state, state.y0
        else:
            return state, filter_output(state, info)

    t0 = ts[0]
    state = solver.init(t0, y0, *args)
    keys = jax.random.split(rng, dts.shape)
    if check_points is None:
        state, traced = jax.lax.scan(
            scan_fun,
            state,
            (dts, keys),
            unroll=unroll,
            _split_transpose=_split_transpose,
        )
    else:
        state, traced = nested_checkpoint_scan(
            scan_fun,
            state,
            (dts, keys),
            nested_lengths=check_points,
            scan_fn=partial(
                jax.lax.scan, unroll=unroll, _split_transpose=_split_transpose
            ),
        )

    return state, traced
