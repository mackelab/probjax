import operator as op
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.util import safe_map, safe_zip

from probjax.utils.jaxutils import ravel_arg_fun, ravel_args
from probjax.utils.odeutil.util import (
    initial_step_size,
    interp_fit,
    mean_error_ratio,
    optimal_step_size,
)

map = safe_map
zip = safe_zip


@partial(
    jax.custom_vjp,
    nondiff_argnums=(
        0,
        1,
        2,
    ),
)
def odeint_adaptive(
    method: Callable,
    drift: Callable,
    kwargs: dict,
    y0: Array,
    ts: Array,
    *args,
):
    return _odeint_adaptive(method, drift, y0, ts, *args, **kwargs)


def _odeint_adaptive(
    method,
    drift,
    y0: Array,
    ts: Array,
    *args,
    dtinit: Optional[float] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    mxstep: int = jnp.inf,
    order: int = 5,
    dtmin: float = 0.0,
    dtmax: float = jnp.inf,
    maxerror: float = 1.0,
    safety: float = 0.9,
    ifactor: float = 10.0,
    dfactor: float = 0.2,
    error_norm: float = 2,
    interpolation_order: int = 3,
    filter_output: Optional[Callable] = None,
):
    y0 = jnp.asarray(y0)
    solver = method(drift)

    def scan_fun(carry, target_t):
        def cond_fun(state):
            i, state, dt, _, _ = state
            t = state.t0
            return (t < target_t) & (i < mxstep) & (dt > 0)

        def body_fun(carry):
            i, state, dt, last_t, interp_coeff = carry
            t, y, f = state.t0, state.y0, state.f0
            # Predicts the next step
            next_state, info = solver.step(state, dt, *args)
            next_y_error = info.y1_error
            y_mid = info.y1_mid
            next_y, next_f = next_state.y0, next_state.f0
            # Error estimation and step size control
            error_ratio = mean_error_ratio(
                next_y_error, y, next_y, rtol, atol, error_norm
            )
            new_interp_coeff = interp_fit(y, next_y, f, next_f, dt=dt, y_mid=y_mid)
            dt = jnp.clip(
                optimal_step_size(
                    dt,
                    error_ratio,
                    maxerror=maxerror,
                    safety=safety,
                    ifactor=ifactor,
                    dfactor=dfactor,
                    order=order,
                ),
                dtmin,
                dtmax,
            )

            cond = (error_ratio <= maxerror) | (dt == dtmin) | (dt == dtmax)

            def accept():
                return [i + 1, next_state, dt, t, new_interp_coeff]

            def reject():
                return [i + 1, state, dt, last_t, interp_coeff]

            return jax.lax.cond(cond, accept, reject)

        # old_state = carry[0]
        i, *carry = jax.lax.while_loop(cond_fun, body_fun, [0] + carry)
        # jax.debug.print(
        #     "i={i}",
        #     i=i,
        # )
        new_state, dt, last_t, interp_coeff = carry
        relative_output_time = (target_t - last_t) / (new_state.t0 - last_t)
        # jax.debug.print(
        #     "t_target={t_target}, t_last={t_last}, t_next={t_next}, t_rel = {t_rel},"
        #     "dt={dt}",
        #     t_target=target_t,
        #     t_last=last_t,
        #     t_next=new_state.t0,
        #     t_rel=relative_output_time,
        #     dt=dt,
        # )

        y_target = jnp.polyval(interp_coeff, relative_output_time)
        # Test polynomial
        # jax.debug.print(
        #     "y={y}, y_new={y_new}, y_est={y_est}, y_est_new={y_est_new}",
        #     y=old_state.y0,
        #     y_new=new_state.y0,
        #     y_est=jnp.polyval(interp_coeff, 0.0),
        #     y_est_new=jnp.polyval(interp_coeff, new_state.t0 - last_t),
        # )
        if filter_output is not None:
            y_target = filter_output(y_target)
        return carry, y_target

    t0 = ts[0]
    if dtinit is None:
        f0 = drift(t0, y0, *args)
        dt = jnp.clip(
            initial_step_size(drift, args, t0, y0, f0, order, rtol, atol),
            min=0.0,
            max=jnp.inf,
        )
    else:
        dt = dtinit

    state = solver.init(t0, y0, *args)
    interp_coeff = jnp.array([y0] * (interpolation_order + 1))
    init_carry = [state, dt, t0, interp_coeff]
    _, ys = jax.lax.scan(scan_fun, init_carry, ts[1:])

    return ys


def _odeint_adaptive_wrapper(
    method,
    drift,
    y0: Array,
    ts: Array,
    *args,
    **kwargs,
):
    print(y0)
    flat_y0, unravel = ravel_args(y0)
    drift_flat = ravel_arg_fun(drift, unravel, 1)
    ys = _odeint_adaptive(
        method,
        drift_flat,
        flat_y0,
        ts,
        *args,
        **kwargs,
    )

    return jax.vmap(unravel)(ys)


def _odeint_fwd(
    method,
    drift,
    kwargs,
    y0: Array,
    ts: Array,
    *args,
):
    ys = _odeint_adaptive(method, drift, y0, ts, *args, **kwargs)
    return ys, (ys, ts, args)


def _odeint_rev(
    method,
    drift,
    kwargs,
    res,
    g,
):
    ys, ts, args = res

    filter_output = kwargs.pop("filter_output", None)

    def aug_dynamics(t, augmented_state):
        y, y_bar, *_ = augmented_state
        # `t` here is negatice time, so we need to negate again to get back to
        # normal time. See the `odeint` invocation in `scan_fun` below.
        y_dot, vjpfun = jax.vjp(drift, -t, y, *args)
        return (-y_dot, *vjpfun(y_bar))

    y_bar = g[-1]
    ts_bar = []
    t0_bar = 0.0

    print(g)

    def scan_fun(carry, i):
        y_bar, t0_bar, args_bar = carry
        # Compute effect of moving measurement time
        # `t_bar` should not be complex as it represents time
        t_bar = jnp.dot(drift(ts[i], ys[i], *args), g[i]).real
        t0_bar = t0_bar - t_bar
        # Run augmented system backwards to previous observation
        augmented_state = (ys[i], y_bar, t0_bar, args_bar)
        _, y_bar, t0_bar, args_bar = _odeint_adaptive_wrapper(
            method,
            aug_dynamics,
            augmented_state,
            jnp.array([-ts[i], -ts[i - 1]]),
            **kwargs,
        )
        y_bar, t0_bar, args_bar = jax.tree_util.tree_map(
            op.itemgetter(1), (y_bar, t0_bar, args_bar)
        )
        # Add gradient from current output
        y_bar = y_bar + g[i - 1]

        if filter_output is not None:
            y_bar = filter_output(y_bar)

        return (y_bar, t0_bar, args_bar), t_bar

    init_carry = (y_bar, t0_bar, jax.tree_util.tree_map(jnp.zeros_like, args))
    (y_bar, t0_bar, args_bar), rev_ts_bar = jax.lax.scan(
        scan_fun, init_carry, jnp.arange(len(ts) - 1, 0, -1)
    )
    ts_bar = jnp.concatenate([jnp.array([t0_bar]), rev_ts_bar[::-1]])
    return (y_bar, ts_bar, *args_bar)


odeint_adaptive.defvjp(_odeint_fwd, _odeint_rev)
