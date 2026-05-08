import operator as op
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax._src.util import safe_map, safe_zip

from probjax.utils.jaxutils import ravel_arg_fun, ravel_args
from probjax.utils.odeutil.adaptive import StepSizeAdaptor
from probjax.utils.odeutil.util import interp_fit

map = safe_map
zip = safe_zip


def _extract_step_size_adaptor(kwargs: dict) -> tuple[StepSizeAdaptor, dict]:
    """Pop ``step_size_adaptor`` out of ``kwargs``, defaulting to a fresh one."""
    kwargs = dict(kwargs)
    adaptor = kwargs.pop("step_size_adaptor", None)
    if adaptor is None:
        adaptor = StepSizeAdaptor()
    return adaptor, kwargs


def odeint_adaptive(
    method: Callable,
    drift: Callable,
    kwargs: dict,
    y0: Array,
    ts: Array,
    *args,
):
    """Adaptive ODE integrator with custom VJP.

    ``drift`` is a pytree-registered callable (see
    :mod:`probjax.utils.functions`). Its array leaves flow as differentiable
    args through the custom VJP so ``jax.jit`` / ``jax.grad`` traces them
    cleanly; its static (callable / config) leaves ride as aux data.
    """
    leaves, treedef = jax.tree_util.tree_flatten(drift)
    return _odeint_adaptive_cvjp(
        method, treedef, kwargs, tuple(leaves), y0, ts, *args
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _odeint_adaptive_cvjp(
    method: Callable,
    drift_treedef: Any,
    kwargs: dict,
    drift_leaves: tuple,
    y0: Array,
    ts: Array,
    *args,
):
    drift = jax.tree_util.tree_unflatten(drift_treedef, list(drift_leaves))
    step_size_adaptor, kwargs = _extract_step_size_adaptor(kwargs)
    return _odeint_adaptive(
        method, drift, y0, ts, *args, step_size_adaptor=step_size_adaptor, **kwargs
    )


def _odeint_adaptive(
    method,
    drift,
    y0: Array,
    ts: Array,
    *args,
    step_size_adaptor: StepSizeAdaptor,
    dtinit: Optional[float] = None,
    interpolation_order: int = 3,
    filter_output: Optional[Callable] = None,
    collect_trace: bool = True,
    **kwargs,
):
    y0 = jnp.asarray(y0)
    solver = method(drift)
    adaptor = step_size_adaptor

    def scan_fun(carry, target_t):
        def cond_fun(state):
            i, state, dt, _, _ = state
            t = state.t0
            return (t < target_t) & (i < adaptor.mxstep) & (dt > 0)

        def body_fun(carry):
            i, state, dt, last_t, interp_coeff = carry
            t, y, f = state.t0, state.y0, state.f0
            next_state, info = solver.step(state, dt, *args)
            next_y_error = info.y1_error
            y_mid = info.y1_mid
            next_y, next_f = next_state.y0, next_state.f0

            error_ratio = adaptor.error_ratio(next_y_error, y, next_y)
            new_interp_coeff = interp_fit(y, next_y, f, next_f, dt=dt, y_mid=y_mid)
            dt = adaptor.next_step_size(dt, error_ratio)
            cond = adaptor.accept_step(error_ratio, dt)

            def accept():
                return [i + 1, next_state, dt, t, new_interp_coeff]

            def reject():
                return [i + 1, state, dt, last_t, interp_coeff]

            return jax.lax.cond(cond, accept, reject)

        i, *carry = jax.lax.while_loop(cond_fun, body_fun, [0] + carry)
        new_state, dt, last_t, interp_coeff = carry
        denom = new_state.t0 - last_t
        relative_output_time = jnp.where(
            denom == 0,
            jnp.zeros_like(denom),
            (target_t - last_t) / denom,
        )

        y_target = jnp.polyval(interp_coeff, relative_output_time)

        trace_value = (
            y_target if filter_output is None else filter_output(y_target, new_state)
        )

        return carry, trace_value

    t0 = ts[0]
    if dtinit is None:
        f0 = drift(t0, y0, *args)
        dt = adaptor.initial_step_size(drift, args, t0, y0, f0)
    else:
        dt = dtinit

    state = solver.init(t0, y0, *args)
    interp_coeff = jnp.array([y0] * (interpolation_order + 1))
    init_carry = [state, dt, t0, interp_coeff]
    targets = ts[1:]

    if collect_trace:
        final_carry, ys = jax.lax.scan(scan_fun, init_carry, targets)
    else:
        def body(i, carry):
            target = targets[i]
            carry, _ = scan_fun(carry, target)
            return carry

        final_carry = jax.lax.fori_loop(0, targets.shape[0], body, init_carry)
        ys = None

    state = final_carry[0]
    return state, ys


def _odeint_adaptive_wrapper(
    method,
    drift,
    y0: Array,
    ts: Array,
    *args,
    step_size_adaptor: StepSizeAdaptor,
    **kwargs,
):
    flat_y0, unravel = ravel_args(y0)
    drift_flat = ravel_arg_fun(drift, unravel, 1)
    state, ys = _odeint_adaptive(
        method,
        drift_flat,
        flat_y0,
        ts,
        *args,
        step_size_adaptor=step_size_adaptor,
        **kwargs,
    )

    if ys is None:
        raise ValueError("Tracing was disabled; cannot unwrap adaptive results.")
    return jax.vmap(unravel)(ys)


def _odeint_fwd(
    method,
    drift_treedef,
    kwargs,
    drift_leaves,
    y0: Array,
    ts: Array,
    *args,
):
    drift = jax.tree_util.tree_unflatten(drift_treedef, list(drift_leaves))
    step_size_adaptor, kwargs = _extract_step_size_adaptor(kwargs)
    result = _odeint_adaptive(
        method, drift, y0, ts, *args, step_size_adaptor=step_size_adaptor, **kwargs
    )
    state, ys = result
    return result, (ys, ts, args, drift_leaves, step_size_adaptor, kwargs)


def _odeint_rev(
    method,
    drift_treedef,
    kwargs,
    res,
    g,
):
    ys, ts, args, drift_leaves, step_size_adaptor, kwargs = res
    drift = jax.tree_util.tree_unflatten(drift_treedef, list(drift_leaves))

    filter_output = kwargs.pop("filter_output", None)
    collect_trace = kwargs.pop("collect_trace", True)

    g_state, g_traj = g

    if not collect_trace or ys is None or g_traj is None:
        raise ValueError(
            "Cannot compute gradients when trace output is disabled for adaptive solvers."
        )

    def aug_dynamics(t, augmented_state):
        y, y_bar, *_ = augmented_state
        # `t` here is negatice time, so we need to negate again to get back to
        # normal time. See the `odeint` invocation in `scan_fun` below.
        y_dot, vjpfun = jax.vjp(drift, -t, y, *args)
        return (-y_dot, *vjpfun(y_bar))

    y_bar = g_traj[-1]
    ts_bar = []
    t0_bar = 0.0

    def scan_fun(carry, i):
        y_bar, t0_bar, args_bar = carry
        # Compute effect of moving measurement time
        # `t_bar` should not be complex as it represents time
        t_bar = jnp.dot(drift(ts[i], ys[i], *args), g_traj[i]).real
        t0_bar = t0_bar - t_bar
        # Run augmented system backwards to previous observation
        augmented_state = (ys[i], y_bar, t0_bar, args_bar)
        _, y_bar, t0_bar, args_bar = _odeint_adaptive_wrapper(
            method,
            aug_dynamics,
            augmented_state,
            jnp.array([-ts[i], -ts[i - 1]]),
            step_size_adaptor=step_size_adaptor,
            **kwargs,
        )
        y_bar, t0_bar, args_bar = jax.tree_util.tree_map(
            op.itemgetter(1), (y_bar, t0_bar, args_bar)
        )
        # Add gradient from current output
        y_bar = y_bar + g_traj[i - 1]

        if filter_output is not None:
            y_bar = filter_output(y_bar)

        return (y_bar, t0_bar, args_bar), t_bar

    init_carry = (y_bar, t0_bar, jax.tree_util.tree_map(jnp.zeros_like, args))
    (y_bar, t0_bar, args_bar), rev_ts_bar = jax.lax.scan(
        scan_fun, init_carry, jnp.arange(len(ts) - 1, 0, -1)
    )
    ts_bar = jnp.concatenate([jnp.array([t0_bar]), rev_ts_bar[::-1]])
    # Cotangents for diff args: (drift_leaves, y0, ts, *args).
    # Gradient w.r.t. drift parameter leaves is not propagated here — return
    # zeros so jax.grad through adaptive solvers treats drift as a constant.
    drift_leaves_bar = tuple(jnp.zeros_like(leaf) for leaf in drift_leaves)
    return (drift_leaves_bar, y_bar, ts_bar, *args_bar)


_odeint_adaptive_cvjp.defvjp(_odeint_fwd, _odeint_rev)
