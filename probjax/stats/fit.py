"""
Gradient-based fitting (:mod:`probjax.stats.fit`)
=================================================

A framework-agnostic training loop: :func:`fit` minimizes any
``loss_fn(params, rng, batch)`` over a params pytree with optax, where
``batch`` is either one batch -- a bare array, or a dict such as
``{"data": x, "context": c}`` whose leaves share the leading example axis --
or an iterable of batches, for data that does not fit in memory:

>>> params, losses = fit(loss_fn, params, key, {"data": x})   # whole array
>>> params, losses = fit(loss_fn, params, key, my_dataloader)  # streamed

Nothing here assumes a particular NN library. Without general callbacks the
loop is a single ``jax.lax.scan``. General callbacks run between compiled scan
chunks; streamed batches and the legacy loss-only ``on_step`` hook use ordered
``io_callback`` calls. Optional EMA stays on the device alongside raw parameters.

Module-backed models (the families in :mod:`probjax.nn.generative`) get the
convenient ``model.fit(rng, data)`` via :class:`FitMixin`, which snapshots
the current module graph and carries non-parameter state through each update:

>>> flow = maf(2, 5, rngs=nnx.Rngs(0))
>>> losses = flow.fit(jax.random.key(0), samples)
>>> flow.logpdf(samples)  # trained in place

This is the object-layer counterpart of the scipy-style classmethod
``rv_generic.fit`` (closed-form / optimizer MLE for parametric families).
"""

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jaxtyping import Array

from probjax.utils.typing import ArrayLike, RngKey

__all__ = [
    "fit",
    "FitMixin",
    "FitState",
    "FitInfo",
    "FitKernel",
    "build_fit_kernel",
    "FitResult",
    "FitCallback",
    "is_batch_stream",
    "take_batches",
]

Schedule = Literal["constant", "warmup_cosine"]

#: Default minibatch size when ``batch_size="auto"``.
_AUTO_BATCH = 512
#: How many times ``num_steps="auto"`` aims to pass over the data.
_AUTO_EPOCHS = 200
_AUTO_MIN_STEPS, _AUTO_MAX_STEPS = 1000, 20000


def _resolve_batch_size(batch_size, num_examples: int) -> Optional[int]:
    """``"auto"`` -> a minibatch; ``None`` -> the full dataset, explicitly."""
    if batch_size == "auto":
        return min(num_examples, _AUTO_BATCH)
    return batch_size


def _resolve_num_steps(num_steps, num_examples: int, batch_size) -> int:
    """``"auto"`` -> enough steps for a fixed number of passes over the data."""
    if num_steps != "auto":
        if (
            not isinstance(num_steps, int)
            or isinstance(num_steps, bool)
            or num_steps < 0
        ):
            raise ValueError("num_steps must be a nonnegative integer or 'auto'.")
        return num_steps
    per_epoch = max(1, num_examples // (batch_size or num_examples))
    return int(min(max(_AUTO_EPOCHS * per_epoch, _AUTO_MIN_STEPS), _AUTO_MAX_STEPS))


# =============================================================================
# Data feeders
# =============================================================================


def is_batch_stream(data) -> bool:
    """Whether ``data`` is an iterable of batches rather than one batch pytree.

    Both readings are pytrees, so nothing about the *structure* separates a
    list of batches from one batch made of several arrays. Rather than guess
    from shapes -- which fails silently and in whichever direction the guess
    went -- the rule is fixed and stated:

    * a **bare array** or a **dict** is one batch;
    * a **list** or **tuple** is a sequence of batches;
    * anything else with ``__iter__`` or ``__next__`` (a generator, a
      ``DataLoader``) is a stream of batches.

    So a single batch that groups several arrays must be a dict --
    ``{"data": x, "context": c}`` -- not a tuple.
    """
    if hasattr(data, "shape") or isinstance(data, (dict, int, float, complex)):
        return False
    return hasattr(data, "__iter__") or hasattr(data, "__next__")


class _BatchStream:
    """Pulls exactly the requested number of batches from an iterable.

    Re-iterates a finite iterable so a plain list of batches works, and says so
    plainly when it cannot -- a spent generator is a common mistake and the
    default error for it is unhelpful.
    """

    def __init__(self, source, name: str = "data"):
        self._source = source
        self._name = name
        self._it = iter(source)
        self._restarts = 0
        self._pending = None

    def peek(self):
        """The next batch, left in place for the following ``__next__``."""
        if self._pending is None:
            self._pending = next(self)
        return self._pending

    def __next__(self):
        if self._pending is not None:
            batch, self._pending = self._pending, None
            return batch
        try:
            return next(self._it)
        except StopIteration:
            pass
        try:
            self._it = iter(self._source)
            self._restarts += 1
            return next(self._it)
        except (TypeError, StopIteration) as exc:
            raise RuntimeError(
                f"{self._name} ran out of batches and could not be restarted. "
                "Pass a re-iterable object (a list, or a DataLoader) or an "
                "infinite iterator, or lower num_steps."
            ) from exc

    def take(self, count: int) -> list:
        return [next(self) for _ in range(count)]


def _maybe_len(source):
    """``len(source)`` when it has one, else None -- ``hasattr`` is not enough,
    since a wrapper may define ``__len__`` that defers to a source without one."""
    try:
        return len(source)
    except TypeError:
        return None


class _BatchAdapter:
    """Normalises what a loader yields into the ``{"data": ..., ...}`` batch dict.

    Loaders come in three shapes and all three are worth supporting: a bare
    array of examples, a ``(data, context)`` tuple, and a dict that is already
    a batch. Re-iterable and length-preserving, so restarting and
    ``num_steps="auto"`` keep working through the wrapper.
    """

    def __init__(self, source, context_key: str = "context"):
        self._source = source
        self._context_key = context_key

    def __len__(self):
        # Deliberately propagates the source's TypeError: reporting a length
        # the source does not have would make num_steps="auto" invent one.
        return len(self._source)

    def __iter__(self):
        for item in self._source:
            yield self._as_batch(item)

    def _as_batch(self, item):
        if isinstance(item, dict):
            if "data" not in item:
                raise ValueError(
                    f"a batch dict must have a 'data' key; got keys {sorted(item)}."
                )
            return item
        if isinstance(item, tuple) and len(item) == 2:
            return {"data": item[0], self._context_key: item[1]}
        return {"data": item}


def take_batches(source, count: int) -> list:
    """The first ``count`` batches of ``source``, restarting it if it is short.

    Exposed for callers that must see some data before training starts --
    fitting a standardising transform, say -- without giving up the ability to
    train on the same iterable afterwards.
    """
    return _BatchStream(source).take(count)


def _as_device_batch(batch):
    return jax.tree.map(jnp.asarray, batch)


def _batch_spec(batch):
    return jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(jnp.shape(a), jnp.asarray(a).dtype), batch
    )


def _check_same_spec(spec, batch, step: int) -> None:
    got = _batch_spec(batch)
    if jax.tree.structure(got) != jax.tree.structure(spec) or any(
        (a.shape, a.dtype) != (b.shape, b.dtype)
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(spec), strict=False)
    ):
        raise ValueError(
            f"batch {step} has a different shape or dtype than the first one "
            f"({got} vs {spec}). The compiled step cannot accept it; pass "
            "drop_last=True to the loader, or pad the final batch."
        )


def _build_optimizer(learning_rate, num_steps, schedule: Schedule, clip_norm):
    """Adam with optional warmup-cosine decay and global-norm clipping.

    Clipping is on by default because a single bad step is otherwise
    unrecoverable: the poisoned parameters persist for the rest of the run and
    nothing downstream detects them. At the default norm it rarely binds on
    healthy training.
    """
    import optax

    if schedule == "constant" or schedule == "warmup_cosine" and num_steps <= 1:
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
            f"schedule must be 'constant' or 'warmup_cosine'; got {schedule!r}."
        )

    adam = optax.adam(lr)
    if clip_norm is None:
        return adam
    if clip_norm <= 0:
        raise ValueError(f"clip_norm must be positive or None; got {clip_norm}.")
    return optax.chain(optax.clip_by_global_norm(clip_norm), adam)


# =============================================================================
# The scanned training loop
# =============================================================================


class FitState(NamedTuple):
    """Training snapshot. Arrays stay on device unless a callback copies them.

    ``step`` counts completed updates. EMA starts at the initial parameters
    and is updated after each optimizer step. ``params`` always holds the raw
    optimizer parameters, even when EMA is selected for the returned model.
    """

    params: object
    opt_state: object
    rng: object
    step: object
    ema_params: object = None
    model_state: object = None
    stopped: object = False


class FitInfo(NamedTuple):
    """Per-update diagnostics; metrics may be any fixed-structure array pytree.

    Both loss and metrics describe the pre-update loss evaluation. Scanning
    stacks every array leaf along a leading update axis.
    """

    loss: Array
    metrics: object = None


class FitKernel(NamedTuple):
    """Pure init/step interface, suitable for JIT and lax.scan.

    init(params, rng, model_state=None) -> FitState
    step(key, state, batch) -> (FitState, FitInfo)

    Each step must advance state.step by one. Custom kernels own their update,
    EMA and state policy; fit only supplies batches, callbacks and history.
    """

    init: object
    step: object


def build_fit_kernel(loss_fn, optimizer, *, ema_decay=None, has_aux=False):
    """Construct a pure Optax update with BlackJAX-style state/info separation.

    With has_aux=True, a stateless loss returns (loss, metrics). With mutable
    model state it returns (loss, (new_model_state, metrics)); otherwise the
    stateful return remains (loss, new_model_state). Auxiliary state and metrics
    are not differentiated. Each step splits its key into next, batch and loss
    keys, matching fit's minibatch sequence; the batch key is reserved for fit.
    """
    import optax

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
        params = optax.apply_updates(state.params, updates)
        ema = (
            None
            if ema_decay is None
            else jax.tree.map(
                lambda old, new: (ema_decay * old + (1 - ema_decay) * new).astype(
                    new.dtype
                ),
                state.ema_params,
                params,
            )
        )
        return FitState(
            params, opt_state, next_key, state.step + 1, ema, model_state, state.stopped
        ), FitInfo(loss, metrics)

    return FitKernel(init, step)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("state", "losses", "info", "valid_steps"),
    meta_fields=("use_ema",),
)
@dataclass(frozen=True)
class FitResult:
    """Final state and stacked diagnostics from functional or module fitting.

    losses aliases info.loss; state.step is absolute. valid_steps counts this
    call's updates, and valid masks any padding in a fixed-shape JIT history.
    params selects raw or EMA weights. use_ema is static pytree metadata.
    """

    state: FitState
    losses: Array
    use_ema: bool = False
    info: FitInfo | None = None
    valid_steps: object = None

    @property
    def valid(self):
        """Mask identifying executed updates, including in padded JIT histories."""
        count = self.losses.shape[0] if self.valid_steps is None else self.valid_steps
        return jnp.arange(self.losses.shape[0]) < count

    @property
    def params(self):
        """Parameters selected by ``use_ema`` for inference."""
        return self.state.ema_params if self.use_ema else self.state.params


class FitCallback:
    """Host hooks for validation, logging, checkpoints and early stopping.

    Override any hook. ``on_fit_begin`` and ``on_step_end`` may return False
    to stop. Snapshots are read-only: mutate neither their containers nor the
    training model. Hooks run outside JIT, so they can evaluate JAX programs
    or save parameters. Only explicit host conversions copy parameter arrays.
    Exceptions propagate normally; ``on_fit_end`` runs on successful completion
    (including early stopping), not after a failed hook or training step.
    """

    def on_fit_begin(self, state: FitState):
        pass

    def on_step_end(self, state: FitState, info: FitInfo):
        pass

    def on_fit_end(self, result: FitResult):
        pass


def _make_step(
    loss_fn,
    tx,
    fetch,
    on_step,
    log_every: int,
    *,
    ema_decay=None,
    extended=False,
    loss_dtype=None,
    kernel=None,
    info_spec=None,
    allow_stop=False,
):
    """Adapt a pure kernel to a scanned loop, retaining legacy loss callbacks."""
    kernel = kernel or build_fit_kernel(loss_fn, tx, ema_decay=ema_decay)

    def host_callback(step, loss):
        result = on_step(int(step), float(loss))
        return np.asarray(result is not None and not bool(result))

    def body(state, _):
        def run(state):
            _, batch_key, _ = jax.random.split(state.rng, 3)
            return kernel.step(state.rng, state, fetch(batch_key))

        if on_step is None and not allow_stop:
            new_state, info = run(state)
        else:

            def skip(state):
                if info_spec is None:
                    return state, FitInfo(jnp.asarray(jnp.nan, loss_dtype))
                return state, jax.tree.map(
                    lambda spec: jnp.zeros(spec.shape, spec.dtype), info_spec
                )

            new_state, info = jax.lax.cond(state.stopped, skip, run, state)
        if on_step is not None:
            fire = ~state.stopped & ((state.step % log_every) == 0)
            stop = jax.lax.cond(
                fire,
                lambda: io_callback(
                    host_callback,
                    jax.ShapeDtypeStruct((), bool),
                    state.step,
                    info.loss,
                    ordered=True,
                ),
                lambda: jnp.asarray(False),
            )
            new_state = new_state._replace(stopped=state.stopped | stop)
        return new_state, info

    if extended:
        return body

    def legacy_body(carry, item):
        params, opt_state, rng, stop, step = carry
        state, info = body(FitState(params, opt_state, rng, step, stopped=stop), item)
        return (
            state.params,
            state.opt_state,
            state.rng,
            state.stopped,
            state.step,
        ), info.loss

    return legacy_body


def _call_callbacks(callbacks, method, *args):
    stop = False
    for callback in callbacks:
        hook = getattr(callback, method, None)
        if hook is None and method == "on_step_end" and callable(callback):
            hook = callback
        if hook is not None:
            result = hook(*args)
            stop |= result is not None and not bool(result)
    return stop


def _io_callbacks(callbacks, method, *args):
    """Explicit host boundary; serialize typed PRNG keys without losing their impl."""

    def has_hook(callback):
        hook = getattr(callback, method, None)
        if isinstance(callback, FitCallback) and (
            getattr(type(callback), method) is getattr(FitCallback, method)
        ):
            return False
        return callable(hook) or (method == "on_step_end" and callable(callback))

    selected = tuple(c for c in callbacks if has_hook(c))
    if not selected:
        return jnp.asarray(False)
    leaves, tree = jax.tree.flatten(args)
    impls = [
        jax.random.key_impl(x)
        if hasattr(x, "dtype") and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
        else None
        for x in leaves
    ]
    buffers = [
        jax.random.key_data(x) if impl is not None else x
        for x, impl in zip(leaves, impls, strict=True)
    ]
    cpu = jax.devices("cpu")[0]

    def invoke(*buffers):
        # Host snapshots are NumPy arrays. Typed keys alone are reconstructed
        # on CPU because their dtype cannot cross the io_callback ABI directly.
        with jax.default_device(cpu):
            restored = [
                jax.random.wrap_key_data(jnp.asarray(x), impl=impl)
                if impl is not None
                else np.asarray(x)
                for x, impl in zip(buffers, impls, strict=True)
            ]
            values = jax.tree.unflatten(tree, restored)
            return np.asarray(_call_callbacks(selected, method, *values))

    return io_callback(invoke, jax.ShapeDtypeStruct((), bool), *buffers, ordered=True)


def _compiled_fit(state, body, num_steps, callbacks, callback_every, use_ema):
    """Fixed-shape execution with optional, explicitly requested host effects."""
    start = state.step
    stop = _io_callbacks(callbacks, "on_fit_begin", state)
    state = state._replace(stopped=state.stopped | stop)

    def step(state, _):
        updated, info = body(state, None)
        if callbacks:
            completed = updated.step - start
            fire = ~state.stopped & (
                (completed % callback_every == 0)
                | (completed == num_steps)
                | updated.stopped
            )
            requested = jax.lax.cond(
                fire,
                lambda: _io_callbacks(callbacks, "on_step_end", updated, info),
                lambda: jnp.asarray(False),
            )
            updated = updated._replace(stopped=updated.stopped | requested)
        return updated, info

    state, info = jax.lax.scan(step, state, None, length=num_steps)
    result = FitResult(state, info.loss, use_ema, info, state.step - start)
    _io_callbacks(callbacks, "on_fit_end", result)
    return result


def _array_fetch(batch, batch_size, num_examples):
    """Minibatch by gathering on device; the dataset never leaves the accelerator."""
    if batch_size is None or batch_size >= num_examples:
        return lambda key: batch
    return lambda key: jax.tree.map(
        lambda a: a[jax.random.randint(key, (batch_size,), 0, num_examples)], batch
    )


def _stream_fetch(stream, spec):
    """Minibatch by pulling from a host iterator.

    ``ordered=True`` is required: without it the runtime may reorder the
    callbacks and hand the loop its batches out of sequence. Measured overhead
    is about 0.04 ms per step against an on-device gather.
    """
    state = {"n": 0, "error": None}

    def pull():
        try:
            batch = _as_device_batch(next(stream))
            state["n"] += 1
            _check_same_spec(spec, batch, state["n"])
        except Exception as exc:  # noqa: BLE001 - re-raised by fit, see below
            # An exception here escapes as an opaque XLA callback failure, so
            # keep the original for fit to re-raise with its own traceback.
            state["error"] = exc
            raise
        return batch

    fetch = lambda key: io_callback(pull, spec, ordered=True)  # noqa: E731
    fetch.stream_state = state
    return fetch


def fit(
    loss_fn,
    params,
    rng: RngKey,
    batch: object,
    *,
    num_steps: "int | Literal['auto']" = "auto",
    batch_size: "int | None | Literal['auto']" = "auto",
    learning_rate: float = 1e-3,
    schedule: Schedule = "constant",
    clip_norm: Optional[float] = 10.0,
    optimizer=None,
    on_step=None,
    log_every: int = 1,
    ema_decay: Optional[float] = None,
    use_ema: bool = False,
    callbacks=(),
    callback_every: int = 1,
    callback_mode: Literal["host", "io"] = "host",
    return_result: bool = False,
    model_state=None,
    has_aux: bool = False,
    initial_state: FitState | None = None,
    kernel: FitKernel | None = None,
) -> Tuple[object, Array] | FitResult:
    """Minimize ``loss_fn`` over ``params`` with minibatch gradient descent.

    Args:
        loss_fn: ``loss_fn(params, rng, batch) -> scalar``. With model_state,
            accepts a fourth state argument and returns ``(loss, new_state)``.
            With has_aux, also returns metrics as described below. Ignored
            when a complete kernel is supplied.
        params: Pytree of trainable parameters.
        rng: PRNG key consumed for minibatching and the per-step loss.
        batch: Either one batch -- a bare array, or a dict of arrays such as
            ``{"data": x, "context": c}`` whose leaves share the leading
            example axis -- or an **iterable of batches**: a list, a tuple, a
            generator, a ``DataLoader``. Note that a list or tuple is always
            read as a sequence of batches, so a single batch grouping several
            arrays must be a dict. With an iterable, minibatching is the
            loader's job: every batch must have the same shapes and dtypes as
            the first (one compiled step serves them all), and a finite
            iterable is restarted as many times as ``num_steps`` requires.
        num_steps: Number of gradient steps, or ``"auto"`` (the default) to
            scale with the dataset: enough steps for a fixed number of passes
            over it, clamped to [1000, 20000]. For an iterable with a
            ``__len__``, the dataset size is taken as
            ``len(batch) * batch_examples``; without one there is nothing to
            scale from and ``"auto"`` means 1000.
        batch_size: Minibatch size, or ``"auto"`` (the default) for
            ``min(num_examples, 512)``. ``None`` still means the full dataset
            every step, which was the previous default and stops being viable
            as the dataset grows. Must not be set for an iterable ``batch``.
        learning_rate: Adam learning rate, used when ``optimizer`` is None.
        schedule: ``"constant"`` or ``"warmup_cosine"`` (5% warmup, cosine decay
            to ``learning_rate / 1000``). Ignored when ``optimizer`` is given.
        clip_norm: Global gradient-norm clip; ``None`` disables. Ignored when
            ``optimizer`` is given.
        optimizer: Optional ``optax.GradientTransformation``. Supplying it takes
            full control, bypassing ``learning_rate``, ``schedule`` and
            ``clip_norm``.
        on_step: Optional ``on_step(step, loss) -> bool | None`` called on the
            host every ``log_every`` steps. Returning ``False`` stops training
            early. Parameters are deliberately not passed: the callback runs
            inside the compiled loop, so handing it the tree would copy every
            parameter back to the host on each call.
        log_every: Cadence for the legacy loss-only ``on_step`` callback.
        ema_decay: Optional fixed EMA decay in [0, 1). Starts at initial params
            and averages after each update. Disabled by default, with no extra
            parameter copy or averaging work.
        use_ema: Return/apply EMA parameters instead of raw optimizer parameters.
            Requires ``ema_decay``. Raw and averaged weights remain separately
            available through ``return_result=True``.
        callbacks: Sequence of :class:`FitCallback` objects or callables
            ``callback(state, info)``. Run on the host after each chunk of
            ``callback_every`` updates and after the final partial chunk.
            Returning False stops before the next chunk. Hooks receive device
            arrays and may run validation or save checkpoints. They observe
            snapshots; change the update rule through ``optimizer`` or ``kernel``.
        callback_every: Number of updates between general callbacks. Larger
            values reduce Python dispatch overhead. With no general callbacks,
            the entire run remains a single scan.
        callback_mode: "host" (default) runs general callbacks between scan
            chunks outside JIT, with device-array snapshots. "io" explicitly
            stages ordered Python callbacks inside a single compiled scan and
            transfers snapshots to the host (NumPy arrays; typed keys are
            reconstructed on CPU). Use "io" for general callbacks under JIT.
            This callback path does not support autodiff or vmap; use host mode
            outside JIT for callbacks that run substantial JAX computations.
        return_result: Return a :class:`FitResult` instead of the legacy tuple.
        model_state: Optional non-parameter state. When supplied, loss_fn must
            accept ``(params, rng, batch, model_state)`` and return
            ``(loss, updated_model_state)``. State updates are carried between
            steps without differentiating them. Available in FitResult.state.
        has_aux: Forward auxiliary loss outputs into FitInfo.metrics. Stateless
            losses return ``(loss, metrics)``; stateful losses return
            ``(loss, (updated_model_state, metrics))``. Metrics may be any fixed
            pytree of arrays; all leaves are stacked in the returned history.
        initial_state: Continue from this FitState, preserving raw parameters,
            optimizer state, RNG, EMA and module state. params/rng are ignored
            and may be None. num_steps is the number of additional updates.
            The stopped flag is cleared. Supply the same optimizer/kernel and
            EMA settings; state does not store or validate their configuration.
            Iterable sources resume at their current position, not a saved one.
        kernel: Optional FitKernel with init(params, rng, model_state=None) and
            step(key, state, batch) -> (FitState, FitInfo). Owns the optimizer,
            loss, EMA and RNG update policy; optimizer may not also be supplied.
            Other optimizer/loss configuration arguments are ignored. Each step
            must increment state.step once and preserve the state/info structure.
            Use this for custom updates and reuse it when resuming schedules.

    Returns:
        ``(trained_params, losses)``, or FitResult when return_result=True.
        Outside JIT in host mode, histories are trimmed to completed updates.
        Under JIT or in io mode, histories have fixed length num_steps and
        skipped entries are zero-filled. Use return_result=True and result.valid
        or result.valid_steps to identify actual updates after early stopping.
        FitResult.info contains stacked FitInfo diagnostics; result.losses is
        the same array as result.info.loss. A general callback observes the
        post-update state and the last pre-update FitInfo in its chunk.

    Warns:
        RuntimeWarning: in eager host mode, if a loss is non-finite. No Python
            checks or warnings are emitted from the pure JIT path; inspect
            result.losses with result.valid to check losses there. The parameters
            are returned as-is rather than repaired -- once a NaN gradient has
            been applied the run is dead, and silently continuing would hide it.

    Note:
        Array-data fitting can be enclosed in jax.jit. Keep configuration such
        as num_steps, batch_size, callbacks and EMA options static (e.g. close
        over them). FitResult is a pytree. Python iterables remain host-only.
        No callbacks means no host effects, and the fit can be differentiated.
        The explicit legacy on_step hook uses io_callback under JIT too.
        Custom kernels must obey the documented state-step contract; runtime
        Python validation of that contract only occurs in eager host mode.
    """
    for name, value in (("log_every", log_every), ("callback_every", callback_every)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    if ema_decay is not None and not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be in [0, 1) or None.")
    if use_ema and ema_decay is None and kernel is None:
        raise ValueError("use_ema requires ema_decay.")
    if callback_mode not in ("host", "io"):
        raise ValueError("callback_mode must be 'host' or 'io'.")
    callbacks = tuple(callbacks)
    for callback in callbacks:
        if not callable(callback) and not any(
            callable(getattr(callback, name, None))
            for name in ("on_fit_begin", "on_step_end", "on_fit_end")
        ):
            raise TypeError("callbacks must contain callables or FitCallback hooks.")
    if is_batch_stream(batch):
        if isinstance(batch_size, int):
            raise ValueError(
                "batch_size cannot be set when batch is an iterable: the "
                "iterable decides its own batch size. Pass an array pytree "
                "instead, or drop batch_size."
            )
        stream = _BatchStream(batch, "batch")
        first = _as_device_batch(stream.peek())
        spec = _batch_spec(first)
        leaves = jax.tree.leaves(first)
        if not leaves:
            raise ValueError("batch must contain at least one array leaf.")
        # Only the *stream* length tells us the dataset size; a bare iterator
        # has no such information and "auto" falls back to the floor.
        if any(a.ndim == 0 for a in leaves):
            raise ValueError("batch leaves must have a leading example axis.")
        per_batch = leaves[0].shape[0]
        if per_batch == 0 or any(a.shape[0] != per_batch for a in leaves):
            raise ValueError("batch leaves must share a nonempty leading example axis.")
        source_len = _maybe_len(batch)
        num_steps = _resolve_num_steps(
            num_steps,
            per_batch * source_len if source_len else per_batch,
            per_batch,
        )
        fetch = _stream_fetch(stream, spec)
    else:
        batch = _as_device_batch(batch)
        leaves = jax.tree.leaves(batch)
        if not leaves:
            raise ValueError("batch must contain at least one array leaf.")
        if any(a.ndim == 0 for a in leaves):
            raise ValueError("batch leaves must have a leading example axis.")
        num_examples = leaves[0].shape[0]
        if num_examples == 0 or any(a.shape[0] != num_examples for a in leaves):
            raise ValueError("batch leaves must share a nonempty leading example axis.")
        if batch_size not in (None, "auto") and (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be positive, None, or 'auto'.")
        batch_size = _resolve_batch_size(batch_size, num_examples)
        num_steps = _resolve_num_steps(num_steps, num_examples, batch_size)
        fetch = _array_fetch(batch, batch_size, num_examples)

    if num_steps < 0:
        raise ValueError("num_steps must be nonnegative.")

    if kernel is not None and optimizer is not None:
        raise ValueError("Supply kernel or optimizer, not both.")
    if (
        kernel is None
        and initial_state is not None
        and ((initial_state.ema_params is not None) != (ema_decay is not None))
    ):
        raise ValueError(
            "Resuming requires the same EMA configuration as initialization."
        )
    if initial_state is not None and model_state is not None:
        raise ValueError("model_state is already supplied by initial_state.")
    if kernel is None:
        tx = (
            optimizer
            if optimizer is not None
            else _build_optimizer(learning_rate, num_steps, schedule, clip_norm)
        )
        kernel = build_fit_kernel(loss_fn, tx, ema_decay=ema_decay, has_aux=has_aux)
    if initial_state is None:
        state = kernel.init(params, rng, model_state)
    else:
        state = initial_state._replace(stopped=jnp.asarray(False))
    if use_ema and state.ema_params is None:
        raise ValueError("use_ema requires EMA parameters in the training state.")
    if initial_state is not None and ema_decay is not None and state.ema_params is None:
        raise ValueError("Cannot resume EMA without an EMA accumulator.")
    traced = any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(state))
    if traced and is_batch_stream(batch):
        raise ValueError(
            "Jitted fit requires array batches; iterate data loaders outside JIT."
        )
    if traced and callbacks and callback_mode != "io":
        raise ValueError(
            "Use callback_mode='io' for explicit Python callbacks inside JIT."
        )
    example = first if is_batch_stream(batch) else batch
    info_spec = jax.eval_shape(kernel.step, state.rng, state, example)[1]
    if not isinstance(info_spec, FitInfo) or info_spec.loss.shape != ():
        raise ValueError("kernel.step must return FitInfo with a scalar loss.")
    body = _make_step(
        loss_fn,
        None,
        fetch,
        on_step,
        log_every,
        extended=True,
        kernel=kernel,
        info_spec=info_spec,
        allow_stop=bool(callbacks) and callback_mode == "io",
    )
    if traced or callback_mode == "io":
        result = _compiled_fit(
            state, body, num_steps, callbacks, callback_every, use_ema
        )
        return result if return_result else (result.params, result.losses)
    start_step = int(state.step)
    target_step = start_step + num_steps
    chunks = []
    stop = _call_callbacks(callbacks, "on_fit_begin", state)
    # Reuse the same compiled chunk across callbacks; only a shorter final
    # chunk needs another compilation. No parameter transfer is required.
    run = jax.jit(
        lambda state, length: jax.lax.scan(body, state, None, length=length),
        static_argnums=(1,),
    )
    try:
        if not callbacks:
            state, info = run(state, num_steps)
            completed = int(state.step) - start_step
            if not 0 <= completed <= num_steps or (
                not bool(state.stopped) and completed != num_steps
            ):
                raise ValueError(
                    "kernel.step must advance state.step by one per update."
                )
        else:
            while int(state.step) < target_step and not stop:
                length = min(callback_every, target_step - int(state.step))
                before = int(state.step)
                state, chunk = run(state, length)
                completed = int(state.step) - before
                if not 1 <= completed <= length or (
                    not bool(state.stopped) and completed != length
                ):
                    raise ValueError(
                        "kernel.step must advance state.step by one per update."
                    )
                chunks.append(
                    jax.tree.map(lambda x, completed=completed: x[:completed], chunk)
                )
                # Ensure callback exceptions and side effects finish here.
                jax.block_until_ready(state)
                requested_stop = _call_callbacks(
                    callbacks,
                    "on_step_end",
                    state,
                    jax.tree.map(
                        lambda x, completed=completed: x[completed - 1], chunk
                    ),
                )
                stop = requested_stop or bool(state.stopped)
            info = (
                jax.tree.map(lambda *xs: jnp.concatenate(xs), *chunks)
                if chunks
                else jax.tree.map(
                    lambda spec: jnp.empty((0, *spec.shape), spec.dtype), info_spec
                )
            )
            state = state._replace(stopped=jnp.asarray(stop))
        # Also surface asynchronous iterator failures inside this try block.
        jax.block_until_ready((state, info))
    except Exception:
        error = getattr(fetch, "stream_state", {}).get("error")
        if error is not None:
            raise error from None
        raise

    info = jax.tree.map(lambda x: x[: int(state.step) - start_step], info)
    losses = info.loss
    finite = jnp.isfinite(losses)
    if not bool(jnp.all(finite)):
        first = int(jnp.argmin(finite))
        warnings.warn(
            f"Training loss became non-finite at step {first} of "
            f"{losses.shape[0]}; the returned parameters are unusable. Lower "
            "the learning rate, tighten clip_norm, or check the model for an "
            "unbounded transform.",
            RuntimeWarning,
            stacklevel=2,
        )
    result = FitResult(state, losses, use_ema, info, state.step - start_step)
    _call_callbacks(callbacks, "on_fit_end", result)
    return result if return_result else (result.params, losses)


class FitMixin:
    """Adds ``model.fit(rng, data, ...)`` for modules with a ``loss`` method."""

    def _default_fit_kwargs(self) -> dict:
        """Model-family defaults for :func:`fit`, overridable per subclass.

        Anything the caller passes explicitly wins, so this only shifts the
        starting point for a family whose loss landscape is known to want
        something other than plain constant-rate Adam.
        """
        return {}

    def _prepare_fit(self, data):
        """Prepare model state before splitting, e.g. fit standardization."""
        return data

    def _fit_loss(self, rng, batch):
        """Override batch-to-loss dispatch without replacing the fit loop."""
        if isinstance(batch, dict):
            return self.loss(
                rng, batch["data"], **{k: v for k, v in batch.items() if k != "data"}
            )
        return self.loss(rng, batch)

    def _fit_param_filter(self):
        """NNX filter selecting trainable parameters (default: all Param)."""
        from flax import nnx

        return nnx.Param

    def _fit_callbacks(self):
        """Model callbacks prepended to those explicitly supplied by the caller."""
        return ()

    def fit(
        self,
        rng: RngKey,
        data: ArrayLike,
        *,
        context: Optional[ArrayLike] = None,
        weights: Optional[ArrayLike] = None,
        **fit_kwargs,
    ) -> Array | FitResult:
        """Train in place; return losses, or FitResult with return_result=True.

        ``ema_decay`` enables averaging; ``use_ema=True`` installs the averaged
        parameters at completion. Non-parameter state (e.g. BatchNorm statistics
        and RNG counters) is carried through training and installed without EMA.
        Existing train/eval flags are respected; call ``model.train()`` first
        when needed. With has_aux=True, loss returns (loss, metrics). Each call
        starts a fresh optimizer and EMA unless initial_state is supplied.
        Resuming uses that snapshot's variables, not subsequent model edits.
        Compatible with nnx.jit when preprocessing hooks are traceable and data
        is an array pytree; custom host preprocessing should run before JIT.
        """
        from flax import nnx

        fit_kwargs = {**self._default_fit_kwargs(), **fit_kwargs}
        data = self._prepare_fit(data)
        param_filter = self._fit_param_filter()
        graphdef, params, rest = nnx.split(self, param_filter, ...)

        def loss_fn(params, rng, batch, model_state):
            model = nnx.merge(graphdef, params, model_state, copy=True)
            output = model._fit_loss(rng, batch)
            rest = nnx.state(model, nnx.Not(param_filter))
            if fit_kwargs.get("has_aux", False):
                loss, metrics = output
                return loss, (rest, metrics)
            return output, rest

        fit_kwargs["callbacks"] = (
            *self._fit_callbacks(),
            *fit_kwargs.get("callbacks", ()),
        )
        want_result = fit_kwargs.pop("return_result", False)
        if is_batch_stream(data):
            if weights is not None:
                raise ValueError(
                    "weights cannot be passed alongside an iterable data source; "
                    "include them in each batch dict instead."
                )
            if context is not None:
                raise ValueError(
                    "context cannot be passed alongside an iterable data "
                    "source; yield (data, context) pairs or batch dicts from "
                    "the iterable instead."
                )
            batch = _BatchAdapter(data)
        else:
            batch = {"data": data}
            if context is not None:
                batch["context"] = context
            if weights is not None:
                batch["weights"] = weights
        result = fit(
            loss_fn,
            params,
            rng,
            batch,
            model_state=None if fit_kwargs.get("initial_state") is not None else rest,
            return_result=True,
            **fit_kwargs,
        )
        nnx.update(self, result.params, result.state.model_state)
        return result if want_result else result.losses
