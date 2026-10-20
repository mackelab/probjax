"""The pure training step: optimizer construction and the init/step kernel."""

from typing import Any, Callable, Literal, NamedTuple, Union, cast

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from probjax.utils.typing import PyTree

__all__ = [
    "Schedule",
    "FitState",
    "FitInfo",
    "FitKernel",
    "build_fit_kernel",
]


#: A named built-in, or any optax schedule ``step -> learning_rate``.
Schedule = Union[Literal["constant", "warmup_cosine"], optax.Schedule]


def _build_optimizer(learning_rate, num_steps, schedule: Schedule, clip_norm):
    """Adam with an optional learning-rate schedule and global-norm clipping.

    Clipping is on by default because a single bad step is otherwise
    unrecoverable: the poisoned parameters persist for the rest of the run and
    nothing downstream detects them. At the default norm it rarely binds on
    healthy training.
    """
    if callable(schedule):
        lr = schedule
    elif schedule == "constant" or schedule == "warmup_cosine" and num_steps <= 1:
        lr = learning_rate
    elif schedule == "warmup_cosine":
        lr = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate / 100.0,
            peak_value=learning_rate,
            warmup_steps=max(1, num_steps // 20),
            decay_steps=num_steps,
            end_value=learning_rate / 1000.0,
        )
    else:
        raise ValueError(
            f"schedule must be 'constant', 'warmup_cosine' or an optax schedule; "
            f"got {schedule!r}."
        )

    adam = optax.adam(lr)
    if clip_norm is None:
        return adam
    if clip_norm <= 0:
        raise ValueError(f"clip_norm must be positive or None; got {clip_norm}.")
    return optax.chain(optax.clip_by_global_norm(clip_norm), adam)


class FitState(NamedTuple):
    """Training snapshot. Arrays stay on device unless a callback copies them.

    ``step`` counts completed updates. EMA starts at the initial parameters
    and is updated after each optimizer step. ``params`` always holds the raw
    optimizer parameters, even when EMA is selected for the returned model.
    """

    params: PyTree[Array]
    opt_state: Any
    rng: Array
    step: Array
    ema_params: PyTree[Array] | None = None
    model_state: Any = None
    stopped: Array | bool = False


class FitInfo(NamedTuple):
    """Per-update diagnostics; metrics may be any fixed-structure array pytree.

    Both loss and metrics describe the pre-update loss evaluation. Scanning
    stacks every array leaf along a leading update axis.
    """

    loss: Array
    metrics: Any = None


class FitKernel(NamedTuple):
    """Pure init/step interface, suitable for JIT and lax.scan.

    init(params, rng, model_state=None) -> FitState
    step(key, state, batch) -> (FitState, FitInfo)

    Each step must advance state.step by one. Custom kernels own their update,
    EMA and state policy; fit only supplies batches, callbacks and history.
    """

    init: Callable[..., FitState]
    step: Callable[..., tuple[FitState, FitInfo]]

    def __call__(self, key, state, batch):
        """One update: the same as ``kernel.step(key, state, batch)``."""
        return self.step(key, state, batch)


def build_fit_kernel(loss_fn, optimizer, *, ema_decay=None, has_aux=False):
    """Construct a pure Optax update with BlackJAX-style state/info separation.

    With has_aux=True, a stateless loss returns (loss, metrics). With mutable
    model state it returns (loss, (new_model_state, metrics)); otherwise the
    stateful return remains (loss, new_model_state). Auxiliary state and metrics
    are not differentiated. Each step splits its key into next, batch and loss
    keys, matching fit's minibatch sequence; the batch key is reserved for fit.
    """
    if ema_decay is not None and not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be in [0, 1) or None.")

    def init(params, rng, model_state=None):
        return FitState(
            params,
            optimizer.init(params),
            rng,
            jnp.asarray(0, jnp.int32),
            params if ema_decay is not None else None,
            model_state,
            jnp.asarray(False),
        )

    def step(key, state, batch):
        next_key, _, loss_key = jax.random.split(key, 3)
        args = (state.params, loss_key, batch)
        stateful = state.model_state is not None
        if stateful:
            args += (state.model_state,)
        if stateful or has_aux:
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(*args)
            if stateful:
                model_state, metrics = aux if has_aux else (aux, None)
            else:
                model_state, metrics = None, aux
        else:
            loss, grads = jax.value_and_grad(loss_fn)(*args)
            model_state, metrics = None, None
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = cast(PyTree[Array], optax.apply_updates(state.params, updates))
        ema = None
        if ema_decay is not None:
            decay = float(ema_decay)
            ema = jax.tree.map(
                lambda old, new: (decay * old + (1 - decay) * new).astype(new.dtype),
                state.ema_params,
                params,
            )
        return FitState(
            params, opt_state, next_key, state.step + 1, ema, model_state, state.stopped
        ), FitInfo(loss, metrics)

    return FitKernel(init, step)
