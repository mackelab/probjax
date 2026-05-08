"""Adaptive SDE integrator using step-doubling local error estimation.

Step doubling: at every attempt, compute one full ``dt`` Euler-Maruyama step
and two ``dt/2`` steps that share the same Brownian increment
(``dW_full = dW_first_half + dW_second_half``). The difference between the
two solutions is an estimate of the local truncation error. The step is
accepted iff the controller is happy with that error and dt is updated via
the standard Hairer–Wanner formula carried by
:class:`~probjax.utils.sdeutil.adaptive.SDEStepSizeAdaptor`.

Scope:
    Diagonal noise + Euler-Maruyama. SRK / Milstein already emit embedded
    error estimates (``y1_error``) and can plug into a separate adaptive
    path without step doubling — left for a follow-up.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Key

from probjax.utils.sdeutil.adaptive import SDEStepSizeAdaptor


def warn_boundary_hits(adaptor, total_hits, n_segments: int) -> None:
    """Host-side warning emitter — call **after** the JIT'd integrator returns.

    Lives outside the JIT graph because :func:`jax.debug.callback` dispatches
    once per :func:`jax.vmap` element, which crushes throughput on heavy
    batched workloads even when the callback ends up silent. By accepting
    a concrete (non-traced) ``total_hits`` after the JIT call, the threshold
    check costs nothing under vmap.

    "Often" means at least ``max(2, n_segments // 4)`` of the output
    segments exhausted ``max_inner_steps``. A handful of hits is ignored.
    Under vmap, ``total_hits`` is an array; we warn if any element crosses
    the threshold and report how many.
    """
    if not adaptor.warn_on_boundary or n_segments == 0:
        return

    import warnings

    threshold = max(2, n_segments // 4)
    max_steps = int(adaptor.max_inner_steps)
    hits_arr = jnp.asarray(total_hits)

    if hits_arr.ndim == 0:
        hits_i = int(hits_arr)
        if hits_i >= threshold:
            warnings.warn(
                f"Adaptive SDE integrator exhausted max_inner_steps={max_steps} "
                f"on {hits_i}/{n_segments} output segments. "
                "Increase max_inner_steps or relax rtol/atol.",
                RuntimeWarning,
                stacklevel=3,
            )
        return

    bad = int((hits_arr >= threshold).sum())
    if bad > 0:
        worst = int(hits_arr.max())
        warnings.warn(
            f"Adaptive SDE integrator exhausted max_inner_steps={max_steps} "
            f"on {bad}/{hits_arr.shape[0]} batched trajectories "
            f"(worst case: {worst}/{n_segments} segments). "
            "Increase max_inner_steps or relax rtol/atol.",
            RuntimeWarning,
            stacklevel=3,
        )


def _step_double_em(
    drift: Callable,
    diffusion: Callable,
    t: Array,
    y: Array,
    dt: Array,
    dW_full: Array,
    dW1: Array,
    dW2: Array,
) -> Tuple[Array, Array]:
    """One full + two half Euler-Maruyama steps; return ``(y_two_halves, error)``.

    For diagonal noise, ``g0 * dW`` is elementwise and the error is the
    elementwise difference between the two refinement levels.
    """
    f0 = drift(t, y)
    g0 = diffusion(t, y)
    y_full = y + dt * f0 + g0 * dW_full

    half = 0.5 * dt
    y_mid = y + half * f0 + g0 * dW1
    f1 = drift(t + half, y_mid)
    g1 = diffusion(t + half, y_mid)
    y_two = y_mid + half * f1 + g1 * dW2

    error = y_two - y_full
    return y_two, error


def _adaptive_to_target(
    drift: Callable,
    diffusion: Callable,
    adaptor: SDEStepSizeAdaptor,
    noise_shape: Tuple[int, ...],
    target_t: Array,
    carry,
):
    """Advance the integrator from ``carry`` until ``t >= target_t``.

    ``carry = (t, y, dt_free, brownian_state)``. ``dt_free`` is the
    controller's preferred step (un-clipped); the loop only clips the
    *effective* step so we don't overshoot the next output point, which
    keeps the controller from collapsing post-target.

    Implementation: a bounded :func:`lax.scan` of length
    ``adaptor.max_inner_steps`` with a :func:`lax.cond` no-op when
    ``t >= target_t``. The fixed length keeps the loop reverse-mode
    differentiable and vmap-friendly. Set ``max_inner_steps`` generously:
    rejected steps and tight tolerances both consume the budget.
    """

    def take_step(state):
        i, t, y, dt_free, bs = state
        dt_eff = jnp.minimum(dt_free, target_t - t)
        committed, dW_full, dW1, dW2 = adaptor.propose_increments(
            bs, t, dt_eff, noise_shape
        )
        # Pathwise-derivative recipe for SDE backprop: dt and dW are
        # hyperparameter-like to the differentiated trajectory — gradients
        # through them are spurious controller / tree noise. Block them.
        dt_eff = jax.lax.stop_gradient(dt_eff)
        dW_full = jax.lax.stop_gradient(dW_full)
        dW1 = jax.lax.stop_gradient(dW1)
        dW2 = jax.lax.stop_gradient(dW2)
        y_new, err = _step_double_em(
            drift, diffusion, t, y, dt_eff, dW_full, dW1, dW2
        )
        # The controller's accept/reject decision should not propagate
        # gradients either — its inputs (norm of err, error_ratio**(-1/p))
        # have NaN derivatives at zero, and under vmap ``lax.cond``
        # degenerates to ``select`` so the unselected branch's NaN poisons
        # the entire batch. ``stop_gradient`` on the controller-only tensors
        # confines the differentiated path to the trajectory itself.
        err = jax.lax.stop_gradient(err)
        ratio = adaptor.error_ratio(err, y, y_new)
        ratio = jax.lax.stop_gradient(ratio)
        controller_dt = adaptor.next_step_size(dt_eff, ratio)
        accept = adaptor.accept_step(ratio, controller_dt)

        was_clipped = dt_eff < dt_free
        next_dt_free = jnp.where(accept & was_clipped, dt_free, controller_dt)
        next_t = jnp.where(accept, t + dt_eff, t)
        next_y = jnp.where(accept, y_new, y)
        next_bs = jax.lax.cond(
            accept,
            lambda: committed,
            lambda: adaptor.on_reject(bs),
        )
        return (i + 1, next_t, next_y, next_dt_free, next_bs)

    def body(state, _):
        i, t, y, dt_free, bs = state
        not_done = (t < target_t) & (dt_free > 0) & (i < adaptor.mxstep)
        new_state = jax.lax.cond(not_done, take_step, lambda s: s, state)
        return new_state, None

    init = (0, *carry)
    final, _ = jax.lax.scan(
        body, init, None, length=adaptor.max_inner_steps, unroll=adaptor.unroll
    )
    n_used, t, y, dt_free, bs = final
    # Use the step counter rather than ``t < target_t``: ``t + dt_eff``
    # can land floating-point-shy of ``target_t`` even on a successful
    # integration. ``n_used == max_inner_steps`` means every iteration
    # was a real step and we never short-circuited via no-op — i.e.
    # we genuinely ran out of budget.
    boundary_hit = (n_used >= adaptor.max_inner_steps).astype(jnp.int32)
    return (t, y, dt_free, bs), boundary_hit


def _sdeint_adaptive(
    drift: Callable,
    diffusion: Callable,
    adaptor: SDEStepSizeAdaptor,
    rng: Key,
    y0: Array,
    ts: Array,
    noise_shape: Tuple[int, ...],
    collect_trace: bool = True,
):
    """Adaptive Euler-Maruyama integration driven by ``adaptor``.

    Args:
        drift, diffusion: Drift and diffusion functions ``(t, y) -> ...``.
            Diffusion must produce a 1D array (diagonal noise).
        adaptor: Step-size controller (Weak or Strong).
        rng: PRNG key.
        y0: Flattened initial state ``(state_dim,)``.
        ts: Output time grid ``(T,)`` (strictly increasing).
        noise_shape: Shape of the Brownian increment per step (typically
            ``(state_dim,)`` for diagonal noise).
        collect_trace: If True, return ``y`` at each ``ts[i>0]``; otherwise
            return only the terminal state.
    """
    t0 = ts[0]
    f0 = drift(t0, y0)
    dt_init = adaptor.initial_step_size(drift, (), t0, y0, f0)
    bs_init = adaptor.init_brownian(rng, t0, ts[-1], noise_shape)
    init_carry = (t0, y0, dt_init, bs_init)
    n_segments = ts[1:].shape[0]

    def scan_fn(carry, target_t):
        new_carry, hit = _adaptive_to_target(
            drift, diffusion, adaptor, noise_shape, target_t, carry
        )
        return new_carry, (new_carry[1], hit)

    if collect_trace:
        final_carry, (ys, hits) = jax.lax.scan(scan_fn, init_carry, ts[1:])
        total_hits = hits.sum()
    else:

        def body(i, state):
            outer_carry, total = state
            new_outer, hit = _adaptive_to_target(
                drift, diffusion, adaptor, noise_shape, ts[1:][i], outer_carry
            )
            return (new_outer, total + hit)

        final_carry, total_hits = jax.lax.fori_loop(
            0, n_segments, body, (init_carry, jnp.int32(0))
        )
        ys = None

    return final_carry[1], ys, total_hits
