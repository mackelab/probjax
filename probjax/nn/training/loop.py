"""The training loop: plain parameter pytrees, optax, nothing about flax.

Two pieces. :func:`train_step` is one pure update ``(state, batch) -> (state,
info)`` for anyone who writes their own loop (sharding, donation, custom
logging). :func:`fit` runs that update: an array batch is minibatched on the
device inside a compiled scan, anything else iterable is a stream whose
batches are pulled on the host, one per update.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array

from probjax.utils.guards import ensure_int, is_tracing
from probjax.utils.typing import PyTree, RngKey

__all__ = [
    "FitInfo",
    "FitResult",
    "FitState",
    "default_optimizer",
    "fit",
    "init_state",
    "is_stream",
    "take_batches",
    "train_step",
]


class FitState(NamedTuple):
    """A training snapshot; pass it back as ``fit(..., state=)`` to resume.

    ``params`` are always the raw optimizer parameters; ``ema_params`` is their
    running average, or ``None`` without averaging. ``aux`` is carried through
    untouched by the functional API -- it is how a module's non-parameter
    state (BatchNorm statistics, counters) survives training.
    """

    params: PyTree[Array]
    opt_state: Any
    step: Array
    rng: Array
    ema_params: PyTree[Array] | None = None
    aux: Any = None


class FitInfo(NamedTuple):
    """The loss (and optional metrics) of one update, before it was applied."""

    loss: Array
    metrics: Any = None


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("state", "losses", "metrics"),
    meta_fields=("use_ema",),
)
@dataclass(frozen=True)
class FitResult:
    """The final state and the per-update loss (and metrics) history."""

    state: FitState
    losses: Array
    metrics: Any = None
    use_ema: bool = False

    @property
    def params(self):
        """The parameters to use: averaged if ``use_ema``, else raw."""
        return self.state.ema_params if self.use_ema else self.state.params


def default_optimizer(
    learning_rate: float = 1e-3,
    num_steps: int = 1000,
    schedule: str | optax.Schedule | None = "onecycle_cosine",
    clip_norm: float | None = 10.0,
) -> optax.GradientTransformation:
    """Adam with a learning-rate schedule and global-norm clipping.

    ``schedule`` is ``"onecycle_cosine"``, ``"warmup_cosine"``,
    ``"constant"``/``None``, or any optax schedule ``step -> rate``.
    """
    if not learning_rate > 0:
        raise ValueError(f"learning_rate must be positive; got {learning_rate!r}.")
    if clip_norm is not None and not clip_norm > 0:
        raise ValueError(f"clip_norm must be positive or None; got {clip_norm!r}.")
    if callable(schedule):
        rate = schedule
    elif schedule in (None, "constant") or num_steps <= 1:
        rate = learning_rate
    elif schedule == "onecycle_cosine":
        rate = optax.cosine_onecycle_schedule(num_steps, learning_rate)
    elif schedule == "warmup_cosine":
        rate = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate / 100.0,
            peak_value=learning_rate,
            warmup_steps=max(1, num_steps // 20),
            decay_steps=num_steps,
            end_value=learning_rate / 1000.0,
        )
    else:
        raise ValueError(
            "schedule must be 'onecycle_cosine', 'warmup_cosine', 'constant', "
            f"None or an optax schedule; got {schedule!r}."
        )
    adam = optax.adam(rate)
    if clip_norm is None:
        return adam
    return optax.chain(optax.clip_by_global_norm(clip_norm), adam)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def is_stream(data) -> bool:
    """Whether ``data`` yields batches rather than being one.

    A mapping, an array or a tuple (of arrays) is one batch; any other
    iterable -- a list, a generator, a data loader -- is a stream.
    """
    if isinstance(data, (Mapping, tuple, jax.Array, np.ndarray)):
        return False
    return hasattr(data, "__iter__")


def take_batches(source, count: int) -> list:
    """The first ``count`` batches of the iterator ``source``, advancing it."""
    batches = []
    for _ in range(count):
        try:
            batches.append(next(source))
        except StopIteration:
            break
    return batches


def _num_examples(batch) -> int:
    """The shared leading axis of a batch's leaves."""
    leaves = jax.tree.leaves(batch)
    if not leaves or any(leaf.ndim == 0 for leaf in leaves):
        raise ValueError("batch leaves must have a leading example axis.")
    size = leaves[0].shape[0]
    if size == 0 or any(leaf.shape[0] != size for leaf in leaves):
        raise ValueError("batch leaves must share a nonempty leading example axis.")
    return size


# ---------------------------------------------------------------------------
# The update
# ---------------------------------------------------------------------------


def _with_aux(loss_fn: Callable) -> Callable:
    """Lift ``loss_fn(params, rng, batch)`` to the loop's aux-carrying form."""

    def loss_with_aux(params, aux, rng, batch):
        out = loss_fn(params, rng, batch)
        loss, metrics = out if isinstance(out, tuple) else (out, None)
        return loss, (aux, metrics)

    return loss_with_aux


def _make_update(loss_with_aux: Callable, optimizer, ema_decay: Optional[float]):
    """One gradient step on a given batch, with the keys already drawn."""

    def update(state: FitState, batch, loss_key, next_rng):
        (loss, (aux, metrics)), grads = jax.value_and_grad(loss_with_aux, has_aux=True)(
            state.params, state.aux, loss_key, batch
        )
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        ema = None
        if ema_decay is not None:
            decay = float(ema_decay)
            ema = jax.tree.map(
                lambda old, new: (decay * old + (1 - decay) * new).astype(new.dtype),
                state.ema_params,
                params,
            )
        state = FitState(params, opt_state, state.step + 1, next_rng, ema, aux)
        return state, FitInfo(loss, metrics)

    return update


def _keys(rng):
    """Every update splits its key the same way: next state, batch, loss."""
    return jax.random.split(rng, 3)


def train_step(
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    *,
    ema_decay: Optional[float] = None,
) -> Callable[[FitState, Any], tuple[FitState, FitInfo]]:
    """One pure update ``(state, batch) -> (state, info)``.

    ``loss_fn(params, rng, batch)`` returns the loss or ``(loss, metrics)``.
    Start from :func:`init_state`; jit, shard and donate as you like.
    """
    update = _make_update(_with_aux(loss_fn), optimizer, ema_decay)

    def step(state, batch):
        next_rng, _, loss_key = _keys(state.rng)
        return update(state, batch, loss_key, next_rng)

    return step


def init_state(
    params: PyTree[Array],
    optimizer: optax.GradientTransformation,
    rng: RngKey,
    *,
    ema: bool = False,
    aux: Any = None,
) -> FitState:
    """A fresh training state for ``params``."""
    return FitState(
        params,
        optimizer.init(params),
        jnp.asarray(0, jnp.int32),
        rng,
        params if ema else None,
        aux,
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _stack(infos: list[FitInfo]) -> tuple[Array, Any]:
    """The history of a run from its chunks (each with a leading step axis)."""
    if not infos:
        return jnp.zeros((0,)), None
    losses = jnp.concatenate([info.loss for info in infos])
    metrics = jax.tree.map(
        lambda *leaves: jnp.concatenate(leaves), *(info.metrics for info in infos)
    )
    return losses, metrics


def _last(info: FitInfo) -> FitInfo:
    return jax.tree.map(lambda leaf: leaf[-1], info)


def _warn_if_non_finite(losses: Array) -> None:
    finite = jnp.isfinite(losses)
    if not bool(jnp.all(finite)):
        warnings.warn(
            f"Training loss became non-finite at step {int(jnp.argmin(finite))} of "
            f"{losses.shape[0]}; the returned parameters are unusable. Lower the "
            "learning rate, tighten clip_norm, or check the model for an "
            "unbounded transform.",
            RuntimeWarning,
            stacklevel=3,
        )


def _run_array(update, state, batch, batch_size, num_steps, every, callback):
    """Minibatch on the device; one compiled scan per callback chunk."""
    batch = jax.tree.map(jnp.asarray, batch)
    size = _num_examples(batch)
    full = batch_size is None or batch_size >= size

    def body(state, _):
        next_rng, batch_key, loss_key = _keys(state.rng)
        if full:
            minibatch = batch
        else:
            index = jax.random.randint(batch_key, (batch_size,), 0, size)
            minibatch = jax.tree.map(lambda leaf: leaf[index], batch)
        return update(state, minibatch, loss_key, next_rng)

    run = jax.jit(
        lambda state, length: jax.lax.scan(body, state, None, length=length),
        static_argnums=1,
    )
    chunk = every if callback is not None else max(num_steps, 1)
    infos = []
    for start in range(0, num_steps, chunk):
        state, info = run(state, min(chunk, num_steps - start))
        infos.append(info)
        if callback is not None and callback(state, _last(info)) is False:
            break
    return state, infos


def _run_stream(update, state, stream, num_steps, every, callback):
    """Pull one batch per update from the caller's iterator."""

    @jax.jit
    def step(state, batch):
        next_rng, _, loss_key = _keys(state.rng)
        return update(state, batch, loss_key, next_rng)

    stream = iter(stream)
    infos = []
    for index in range(num_steps):
        try:
            batch = next(stream)
        except StopIteration:
            raise RuntimeError(
                f"the data stream ran out after {index} of {num_steps} batches. "
                "A stream is not restarted; lower num_steps or pass an endless "
                "iterator."
            ) from None
        state, info = step(state, jax.tree.map(jnp.asarray, batch))
        infos.append(jax.tree.map(lambda leaf: leaf[None], info))
        due = (index + 1) % every == 0 or index + 1 == num_steps
        if callback is not None and due and callback(state, info) is False:
            break
    return state, infos


def _fit(
    loss_with_aux: Callable,
    params: PyTree[Array],
    rng: RngKey,
    data,
    *,
    aux: Any = None,
    num_steps: int = 1000,
    batch_size: Optional[int] = 512,
    optimizer: Optional[optax.GradientTransformation] = None,
    learning_rate: float = 1e-3,
    schedule: str | optax.Schedule | None = "onecycle_cosine",
    clip_norm: Optional[float] = 10.0,
    ema_decay: Optional[float] = None,
    use_ema: bool = False,
    callback: Optional[Callable[[FitState, FitInfo], Any]] = None,
    every: int = 1,
    state: Optional[FitState] = None,
) -> FitResult:
    """The loop behind :func:`fit` and :func:`~probjax.nn.training.fit_module`."""
    ensure_int("num_steps", num_steps, min_value=0)
    ensure_int("every", every, min_value=1)
    if batch_size is not None:
        ensure_int("batch_size", batch_size, min_value=1)
    if ema_decay is not None and not 0 <= ema_decay < 1:
        raise ValueError(f"ema_decay must be in [0, 1) or None; got {ema_decay!r}.")
    if use_ema and ema_decay is None:
        raise ValueError("use_ema requires ema_decay.")
    if callback is not None and not callable(callback):
        raise TypeError(f"callback must be callable; got {type(callback).__name__}.")
    if optimizer is None:
        optimizer = default_optimizer(learning_rate, num_steps, schedule, clip_norm)

    if state is None:
        state = init_state(params, optimizer, rng, ema=ema_decay is not None, aux=aux)
    elif ema_decay is not None and state.ema_params is None:
        raise ValueError(
            "this state has no averaged parameters to continue; resume without "
            "ema_decay or start a fresh run."
        )

    traced = any(is_tracing(leaf) for leaf in jax.tree.leaves(state))
    if traced and callback is not None:
        raise ValueError("callbacks run on the host and cannot be used inside jit.")

    update = _make_update(loss_with_aux, optimizer, ema_decay)
    if is_stream(data):
        state, infos = _run_stream(update, state, data, num_steps, every, callback)
    else:
        state, infos = _run_array(
            update, state, data, batch_size, num_steps, every, callback
        )
    losses, metrics = _stack(infos)
    if not traced:
        _warn_if_non_finite(losses)
    return FitResult(state, losses, metrics, use_ema)


def fit(
    loss_fn: Callable,
    params: PyTree[Array],
    rng: RngKey,
    data,
    **options,
) -> FitResult:
    """Minimize ``loss_fn`` over ``params`` with minibatch gradient descent.

    Args:
        loss_fn: ``loss_fn(params, rng, batch) -> loss`` or ``(loss,
            metrics)``; metrics are stacked into the result.
        params: Pytree of trainable parameters (ignored when resuming).
        rng: Key for minibatching and the loss (ignored when resuming).
        data: One batch -- a mapping, array or tuple whose leaves share a
            leading example axis, minibatched on the device -- or any other
            iterable of batches, pulled one per update (see :func:`is_stream`).
        **options: ``num_steps`` (1000), ``batch_size`` (512; ``None`` is the
            full batch; ignored for a stream), ``optimizer`` (default:
            :func:`default_optimizer` from ``learning_rate`` (1e-3),
            ``schedule`` ("onecycle_cosine") and ``clip_norm`` (10.0)),
            ``ema_decay`` / ``use_ema`` (parameter averaging), ``callback``
            (``callback(state, info)`` every ``every`` updates; returning
            False stops the run) and ``state`` (resume; ``num_steps`` then
            counts additional updates).

    With array data and no callback, ``fit`` can run inside ``jax.jit`` and be
    differentiated.
    """
    return _fit(_with_aux(loss_fn), params, rng, data, **options)
