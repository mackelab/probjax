import operator as op
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax._src.util import safe_map, safe_zip

from probjax.utils.jaxutils import ravel_arg_fun, ravel_args
from probjax.utils.odeutil.adaptive import AdaptiveParams, StepSizeAdapter
from probjax.utils.odeutil.util import (
    initial_step_size,
    interp_fit,
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
    # Extract AdaptiveParams if present, or create default one
    adaptive_params = kwargs.pop("adaptive_params", None)
    if adaptive_params is None:
        # Create default params using any rtol, atol, etc. from kwargs
        rtol = kwargs.pop("rtol", 1e-3)
        atol = kwargs.pop("atol", 1e-5)
        mxstep = kwargs.pop("mxstep", jnp.inf)
        dtmin = kwargs.pop("dtmin", 0.0)
        dtmax = kwargs.pop("dtmax", jnp.inf)
        maxerror = kwargs.pop("maxerror", 1.0)
        safety = kwargs.pop("safety", 0.9)
        ifactor = kwargs.pop("ifactor", 10.0)
        dfactor = kwargs.pop("dfactor", 0.2)
        error_norm = kwargs.pop("error_norm", 2)
        order = kwargs.pop("order", 5)

        adaptive_params = AdaptiveParams(
            rtol=rtol,
            atol=atol,
            mxstep=mxstep,
            dtmin=dtmin,
            dtmax=dtmax,
            maxerror=maxerror,
            safety=safety,
            ifactor=ifactor,
            dfactor=dfactor,
            error_norm=error_norm,
            order=order,
        )

    return _odeint_adaptive(
        method, drift, y0, ts, *args, adaptive_params=adaptive_params, **kwargs
    )


def _odeint_adaptive(
    method,
    drift,
    y0: Array,
    ts: Array,
    *args,
    adaptive_params: AdaptiveParams,
    dtinit: Optional[float] = None,
    interpolation_order: int = 3,
    return_state: bool = False,
    filter_output: Optional[Callable] = None,
    **kwargs,
):
    y0 = jnp.asarray(y0)
    solver = method(drift)

    # Create adapter for step size control
    adapter = StepSizeAdapter(adaptive_params)

    def scan_fun(carry, target_t):
        def cond_fun(state):
            i, state, dt, _, _ = state
            t = state.t0
            return (t < target_t) & (i < adaptive_params.mxstep) & (dt > 0)

        def body_fun(carry):
            i, state, dt, last_t, interp_coeff = carry
            t, y, f = state.t0, state.y0, state.f0
            # Predicts the next step
            next_state, info = solver.step(state, dt, *args)
            next_y_error = info.y1_error
            y_mid = info.y1_mid
            next_y, next_f = next_state.y0, next_state.f0

            # Use adapter for error estimation and step size control
            error_ratio = adapter.error_ratio(next_y_error, y, next_y)
            new_interp_coeff = interp_fit(y, next_y, f, next_f, dt=dt, y_mid=y_mid)
            dt = adapter.next_step_size(dt, error_ratio)
            cond = adapter.accept_step(error_ratio, dt)

            def accept():
                return [i + 1, next_state, dt, t, new_interp_coeff]

            def reject():
                return [i + 1, state, dt, last_t, interp_coeff]

            return jax.lax.cond(cond, accept, reject)

        i, *carry = jax.lax.while_loop(cond_fun, body_fun, [0] + carry)
        new_state, dt, last_t, interp_coeff = carry
        relative_output_time = (target_t - last_t) / (new_state.t0 - last_t)

        y_target = jnp.polyval(interp_coeff, relative_output_time)

        if filter_output is not None:
            y_target = filter_output(y_target, new_state)

        return carry, y_target

    t0 = ts[0]
    if dtinit is None:
        f0 = drift(t0, y0, *args)
        dt = jnp.clip(
            initial_step_size(
                drift,
                args,
                t0,
                y0,
                f0,
                adaptive_params.order,
                adaptive_params.rtol,
                adaptive_params.atol,
            ),
            min=0.0,
            max=jnp.inf,
        )
    else:
        dt = dtinit

    state = solver.init(t0, y0, *args)
    interp_coeff = jnp.array([y0] * (interpolation_order + 1))
    init_carry = [state, dt, t0, interp_coeff]
    final_carry, ys = jax.lax.scan(scan_fun, init_carry, ts[1:])
    state = final_carry[0]

    if return_state:
        return state, ys
    else:
        return ys


def _odeint_adaptive_wrapper(
    method,
    drift,
    y0: Array,
    ts: Array,
    *args,
    adaptive_params: AdaptiveParams,
    **kwargs,
):
    flat_y0, unravel = ravel_args(y0)
    drift_flat = ravel_arg_fun(drift, unravel, 1)
    ys = _odeint_adaptive(
        method,
        drift_flat,
        flat_y0,
        ts,
        *args,
        adaptive_params=adaptive_params,
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
    # Extract or create AdaptiveParams
    adaptive_params = kwargs.pop("adaptive_params", None)
    if adaptive_params is None:
        # Create default params using any rtol, atol, etc. from kwargs
        rtol = kwargs.pop("rtol", 1e-3)
        atol = kwargs.pop("atol", 1e-5)
        mxstep = kwargs.pop("mxstep", jnp.inf)
        dtmin = kwargs.pop("dtmin", 0.0)
        dtmax = kwargs.pop("dtmax", jnp.inf)
        maxerror = kwargs.pop("maxerror", 1.0)
        safety = kwargs.pop("safety", 0.9)
        ifactor = kwargs.pop("ifactor", 10.0)
        dfactor = kwargs.pop("dfactor", 0.2)
        error_norm = kwargs.pop("error_norm", 2)
        order = kwargs.pop("order", 5)

        adaptive_params = AdaptiveParams(
            rtol=rtol,
            atol=atol,
            mxstep=mxstep,
            dtmin=dtmin,
            dtmax=dtmax,
            maxerror=maxerror,
            safety=safety,
            ifactor=ifactor,
            dfactor=dfactor,
            error_norm=error_norm,
            order=order,
        )

    ys = _odeint_adaptive(
        method, drift, y0, ts, *args, adaptive_params=adaptive_params, **kwargs
    )
    return ys, (ys, ts, args, adaptive_params, kwargs)


def _odeint_rev(
    method,
    drift,
    kwargs,
    res,
    g,
):
    ys, ts, args, adaptive_params, kwargs = res

    filter_output = kwargs.pop("filter_output", None)
    return_state = kwargs.pop("return_state", False)

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
            adaptive_params=adaptive_params,
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
