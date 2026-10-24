"""Compiled execution for any kernel: one transition, a run and a warmup.

Every kernel shares the step signature ``(key, state, params, *args) ->
(state, info)``, so these three functions drive all of them:

- :func:`step` takes one transition and, optionally, one adaptor update;
- :func:`run` scans ``step``, keeps every ``thin``-th result and returns the
  requested fields;
- :func:`warmup` fits the kernel parameters with an adaptation policy.

Several chains run under ``jax.vmap``; wrap the adaptor in
:func:`~probjax.inference.adaptation.pooled` to share one adaptation across
them.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from probjax.inference.base import Adaptor, Kernel, RunResult, Warmup
from probjax.utils.jaxutils import WithProgressBarAPI, print_scan
from probjax.utils.typing import RngKey

__all__ = ["run", "step", "warmup"]

#: Stats shown by ``run(..., progress=True)`` when the kernel provides them.
DEFAULT_PROGRESS_STATS = ("logdensity", "acceptance_rate", "loss")


def _field(name: str, state: Any, info: Any) -> Any:
    """Read ``name`` from the state, else from the transition info."""
    for source in (state, info):
        if hasattr(source, name):
            return getattr(source, name)
    raise AttributeError(f"{name!r} is neither a state nor an info field.")


def _step(key, kernel, state, params, *args, adaptor=None, adaptor_state=None):
    state, info = kernel.step(key, state, params, *args)
    if adaptor is not None:
        adaptor_state, params, _ = adaptor.update(state, info, adaptor_state, params)
    return state, params, adaptor_state, info


@partial(jax.jit, static_argnames=("kernel", "adaptor"))
def step(
    key: RngKey,
    kernel: Kernel,
    state: Any,
    params: Any,
    *args: Any,
    adaptor: Adaptor | None = None,
    adaptor_state: Any = None,
) -> tuple[Any, Any, Any, Any]:
    """Take one transition, then apply one adaptor update if given.

    Returns ``(state, params, adaptor_state, info)``. Use it to write your own
    outer loop, for example one that alternates sampling with fitting.
    """
    return _step(
        key, kernel, state, params, *args, adaptor=adaptor, adaptor_state=adaptor_state
    )


def _progress_scan(body, carry, xs, length, names):
    """``lax.scan`` with a progress bar showing a moving average of ``names``."""

    def update_stats(stats, _carry, output):
        return tuple(
            0.9 * old + 0.1 * new for old, new in zip(stats, output[1], strict=True)
        )

    def print_fn(i, total, stats):
        WithProgressBarAPI._write_progress(WithProgressBarAPI, i, total, stats, names)

    return print_scan(
        body,
        carry,
        tuple(0.0 for _ in names),
        xs=xs,
        length=length,
        update_stats=update_stats,
        print_rate=length // 50 + 1,
        print_fn=print_fn,
    )


@partial(
    jax.jit,
    static_argnames=("kernel", "num_steps", "adaptor", "collect", "thin", "progress"),
)
def run(
    key: RngKey,
    kernel: Kernel,
    state: Any,
    params: Any,
    num_steps: int,
    *,
    args: tuple[Any, ...] | None = None,
    adaptor: Adaptor | None = None,
    collect: tuple[str, ...] = ("position",),
    thin: int = 1,
    progress: bool | tuple[str, ...] = False,
) -> RunResult:
    """Run ``num_steps`` transitions and keep every ``thin``-th.

    Args:
        key: Random key.
        kernel: The kernel to run.
        state: Initial kernel state.
        params: Kernel parameters.
        num_steps: Number of transitions; must be a multiple of ``thin``.
        args: Per-step extra arguments (e.g. minibatches), a tuple whose
            leaves have a leading axis of length ``num_steps``.
        adaptor: Adapts ``params`` after every transition. The returned
            params are the adaptor's finalized ones.
        collect: State or info fields to record at every kept step.
        thin: Keep every ``thin``-th transition.
        progress: Show a progress bar. ``True`` shows the log density and
            acceptance rate; a tuple of names shows those state/info fields.

    Returns:
        A :class:`RunResult` whose ``trace`` maps each name in ``collect`` to
        an array with ``num_steps // thin`` rows.
    """
    if num_steps % thin:
        raise ValueError(f"num_steps={num_steps} must be a multiple of thin={thin}.")
    num_kept = num_steps // thin
    keys = jax.random.split(key, (num_kept, thin))
    if args is None:
        args = ()
    args = jax.tree.map(lambda a: a.reshape((num_kept, thin) + a.shape[1:]), args)
    adaptor_state = None if adaptor is None else adaptor.init(state, params)

    names = DEFAULT_PROGRESS_STATS if progress is True else progress or ()
    if names:
        # Show only the stats this kernel has, found from the output shapes.
        first_args = jax.tree.map(lambda a: a[0, 0], args)
        state_shape, info_shape = jax.eval_shape(
            kernel.step, keys[0, 0], state, params, *first_args
        )
        names = tuple(
            name
            for name in names
            if hasattr(state_shape, name) or hasattr(info_shape, name)
        )

    def transition(carry, xs):
        state, params, adaptor_state = carry
        step_key, step_args = xs
        state, params, adaptor_state, info = _step(
            step_key,
            kernel,
            state,
            params,
            *step_args,
            adaptor=adaptor,
            adaptor_state=adaptor_state,
        )
        return (state, params, adaptor_state), info

    def keep_one(carry, xs):
        carry, info = jax.lax.scan(transition, tuple(carry), xs)
        state, last_info = carry[0], jax.tree.map(lambda v: v[-1], info)
        record = {name: _field(name, state, last_info) for name in collect}
        stats = tuple(
            jnp.mean(jnp.asarray(_field(name, state, last_info), jnp.float32))
            for name in names
        )
        return carry, (record, stats)

    carry = (state, params, adaptor_state)
    if names:
        carry, (trace, _) = _progress_scan(
            keep_one, carry, (keys, args), num_kept, names
        )
    else:
        carry, (trace, _) = jax.lax.scan(keep_one, carry, (keys, args))
    state, params, adaptor_state = carry
    if adaptor is not None:
        params, _ = adaptor.finalize(adaptor_state, params)
    return RunResult(state, params, trace if collect else None)


def warmup(
    key: RngKey,
    kernel: Kernel,
    state: Any,
    params: Any,
    num_steps: int,
    policy: Adaptor | Warmup | None = None,
) -> RunResult:
    """Fit the kernel parameters over ``num_steps`` transitions.

    ``policy`` is an :class:`Adaptor` (run with :func:`run`, then finalized) or
    a :class:`Warmup` procedure. ``None`` uses the kernel's
    ``default_adaptation``.
    """
    if policy is None:
        if kernel.default_adaptation is None:
            raise ValueError(
                "This kernel has no default adaptation; pass policy= explicitly."
            )
        policy = kernel.default_adaptation()
    if isinstance(policy, Warmup):
        return policy.run(key, kernel, state, params, num_steps)
    if isinstance(policy, Adaptor):
        return run(key, kernel, state, params, num_steps, adaptor=policy, collect=())
    raise TypeError(
        f"policy must be an Adaptor or a Warmup, got {type(policy).__name__}."
    )
