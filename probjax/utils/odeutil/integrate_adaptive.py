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
    # Integrate the raw (unfiltered) trace: the reverse pass needs the true
    # states for the adjoint dynamics and as VJP primals for the output
    # filter. The filtered trace is the primal output.
    filter_output = kwargs.get("filter_output", None)
    state, ys_raw = _odeint_adaptive(
        method,
        drift,
        y0,
        ts,
        *args,
        step_size_adaptor=step_size_adaptor,
        **dict(kwargs, filter_output=None),
    )
    if ys_raw is None or filter_output is None:
        ys = ys_raw
    else:
        ys = jax.vmap(lambda z: filter_output(z, None))(ys_raw)
    # Residuals carry array data only. Static configuration (the step-size
    # adaptor, the filter, tolerances, flags) is recovered from the static
    # `kwargs` argument in the reverse rule.
    return (state, ys), (y0, ys_raw, ts, args, drift_leaves)


def _odeint_rev(
    method,
    drift_treedef,
    kwargs,
    res,
    g,
):
    y0, ys_raw, ts, args, drift_leaves = res
    step_size_adaptor, rev_kwargs = _extract_step_size_adaptor(kwargs)
    filter_output = rev_kwargs.get("filter_output", None)
    collect_trace = rev_kwargs.get("collect_trace", True)
    interpolation_order = rev_kwargs.get("interpolation_order", 3)

    _, g_ys = g
    if not collect_trace or ys_raw is None or g_ys is None:
        raise ValueError(
            "Cannot compute gradients when trace output is disabled for adaptive solvers."
        )

    n_times = ts.shape[0]

    def apply_drift(leaves, t, y, args_tuple):
        d = jax.tree_util.tree_unflatten(drift_treedef, list(leaves))
        return d(t, y, *args_tuple)

    def jump_at(k):
        """Adjoint jump at ``ts[k + 1]``: VJP of the output filter applied to
        the cotangent of the filtered trace entry."""
        gk = g_ys[k]
        if filter_output is None:
            return gk
        _, vjpfun = jax.vjp(lambda z: filter_output(z, None), ys_raw[k])
        return vjpfun(gk)[0]

    def aug_dynamics(tau, aug):
        # Adjoint system in reversed time ``tau = -t``. The augmented state
        # is ``(y, y_bar, args_bar, leaves_bar)``. With ``F(tau, .)`` the
        # dynamics in ``tau`` and ``f`` the original drift,
        # ``d y_bar/dtau = +y_bar^T f_y`` (and likewise for the parameter
        # blocks), because ``F(tau, y) = -f(-tau, y)``.
        y, y_bar, args_bar, leaves_bar = aug
        t = -tau
        y_dot, vjpfun = jax.vjp(apply_drift, drift_leaves, t, y, args)
        leaves_ct, _t_ct, y_ct, args_ct = vjpfun(y_bar)
        return (-y_dot, y_ct, args_ct, leaves_ct)

    def backward_step(aug_init, t_hi, t_lo):
        trace = _odeint_adaptive_wrapper(
            method,
            aug_dynamics,
            aug_init,
            jnp.array([-t_hi, -t_lo]),
            step_size_adaptor=step_size_adaptor,
            interpolation_order=interpolation_order,
            filter_output=None,
            collect_trace=True,
        )
        # Single-entry trace holding the augmented state at ``tau = -t_lo``.
        return jax.tree_util.tree_map(lambda leaf: leaf[0], trace)

    def scan_fun(carry, i):
        # Going backward: ``i`` runs over ``n_times - 1, ..., 1`` and
        # ``ys_raw[i - 1]`` / ``g_ys[i - 1]`` pair with ``ts[i]``.
        y_bar, args_bar, leaves_bar = carry
        jump = jump_at(i - 1)
        # Gradient w.r.t. the observation time: ``dL/dt_i = jump_i^T f(t_i, y_i)``.
        t_bar = jnp.dot(
            apply_drift(drift_leaves, ts[i], ys_raw[i - 1], args), jump
        ).real
        aug_init = (ys_raw[i - 1], y_bar + jump, args_bar, leaves_bar)
        _, y_bar, args_bar, leaves_bar = backward_step(aug_init, ts[i], ts[i - 1])
        return (y_bar, args_bar, leaves_bar), t_bar

    init_carry = (
        jnp.zeros_like(y0),
        jax.tree_util.tree_map(jnp.zeros_like, args),
        jax.tree_util.tree_map(jnp.zeros_like, drift_leaves),
    )
    (y_bar, args_bar, leaves_bar), rev_t_bars = jax.lax.scan(
        scan_fun, init_carry, jnp.arange(n_times - 1, 0, -1)
    )
    # Gradient w.r.t. the initial time: ``dL/dt_0 = -y_bar(t_0)^T f(t_0, y_0)``.
    # ``y_bar`` here carries the jumps at ``ts[1:]``; the jump at ``ts[0]``
    # is handled by autodiff through the stacked initial state in ``core``.
    t0_bar = -jnp.dot(y_bar, apply_drift(drift_leaves, ts[0], y0, args)).real
    ts_bar = jnp.concatenate([t0_bar[None], rev_t_bars[::-1]])
    # Cotangents for the differentiable args ``(drift_leaves, y0, ts, *args)``.
    return (leaves_bar, y_bar, ts_bar, *args_bar)


_odeint_adaptive_cvjp.defvjp(_odeint_fwd, _odeint_rev)
