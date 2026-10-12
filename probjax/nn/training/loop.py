"""The scanned training loop and its callbacks."""

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jaxtyping import Array

from probjax.nn.training.batches import (
    _BatchStream,
    _array_fetch,
    _as_device_batch,
    _batch_spec,
    _maybe_len,
    _resolve_batch_size,
    _resolve_num_steps,
    _stream_fetch,
    is_batch_stream,
)
from probjax.nn.training.kernel import (
    FitInfo,
    FitKernel,
    FitState,
    Schedule,
    _build_optimizer,
    build_fit_kernel,
)
from probjax.utils.typing import RngKey

__all__ = [
    "fit",
    "FitResult",
    "FitCallback",
]


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
    params selects raw or EMA params. use_ema is static pytree metadata.
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
        batch: Either one batch -- any pytree of arrays whose leaves share
            the leading example axis, e.g. a bare array or a dict such as
            ``{"x": x, "y": y}`` -- or an **iterable of batches**: a list, a tuple, a
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
        schedule: ``"constant"``, ``"warmup_cosine"`` (5% warmup, cosine decay
            to ``learning_rate / 1000``), or any optax schedule ``step ->
            learning_rate``, used as is (``learning_rate`` is then ignored).
            Ignored when ``optimizer`` is given.
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
            Requires ``ema_decay``. Raw and averaged params remain separately
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
