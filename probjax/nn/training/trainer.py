"""The training loop.

One engine, two shapes. Outside jit it runs compiled chunks and calls Python
callbacks between them; inside jit it is a single scan with no host effects at
all. Nothing here knows about flax, and nothing here knows what a batch
*means* -- the data contract says how a batch reaches the loss.
"""

import itertools
import warnings
from typing import Callable, Optional, cast

import jax
import jax.numpy as jnp
import optax
from jax.experimental import io_callback
from jaxtyping import Array

from probjax.nn.training.config import (
    DataSpec,
    FitConfig,
    check_config,
)
from probjax.nn.training.state import FitInfo, FitResult, FitState
from probjax.utils.guards import is_mapping, is_tracing
from probjax.utils.jaxutils import spec_from_value
from probjax.utils.typing import PyTree, RngKey

__all__ = ["fit", "FitCallback", "take_batches"]


class FitCallback:
    """Host hooks for validation, logging, checkpoints and early stopping.

    Override any hook. ``on_fit_begin`` and ``on_step_end`` may return False to
    stop the run. Hooks run on the host between compiled chunks, so they may
    evaluate JAX programs or save parameters; the snapshots they receive are
    read-only, and mutate neither them nor the model being trained.

    Exceptions propagate normally. ``on_fit_end`` runs after a successful run,
    including one that stopped early, but not after a failed hook or a failed
    training step.
    """

    def on_fit_begin(self, state: FitState):
        """Called once, before the first update. Return False to stop."""

    def on_step_end(self, state: FitState, info: FitInfo):
        """Called after each chunk of updates. Return False to stop."""

    def on_fit_end(self, result: FitResult):
        """Called once, when the run has finished."""


def _hook(callback, name: str):
    """The callback's implementation of a hook, or None if it has none.

    A plain callable is an ``on_step_end`` hook. A :class:`FitCallback` only
    counts as implementing a hook if it overrides the base method, so a
    callback that does not care about a hook does not pay for a call.
    """
    if isinstance(callback, FitCallback):
        if getattr(type(callback), name) is getattr(FitCallback, name):
            return None
        return getattr(callback, name)
    if name == "on_step_end" and callable(callback):
        return callback
    return None


def _dispatch(callbacks, name: str, *args) -> bool:
    """Call one hook on every callback that implements it.

    Returns whether any of them asked to stop.
    """
    stop = False
    for callback in callbacks:
        hook = _hook(callback, name)
        if hook is None:
            continue
        result = hook(*args)
        stop = stop or (result is not None and not bool(result))
    return stop


# ---------------------------------------------------------------------------
# Batch sources
# ---------------------------------------------------------------------------


def take_batches(source, count: int) -> list:
    """The first ``count`` batches of ``source``.

    ``source`` is an iterator of batches and is advanced past the batches
    returned; a caller that needs the rest of it keeps the iterator.
    """
    source = iter(source)
    batches = []
    for _ in range(count):
        try:
            batches.append(next(source))
        except StopIteration:
            break
    return batches


def _check_batch(batch) -> int:
    """The number of examples in a batch, checking that the leaves agree."""
    leaves = jax.tree.leaves(batch)
    if not leaves:
        raise ValueError("batch must contain at least one array leaf.")
    if any(leaf.ndim == 0 for leaf in leaves):
        raise ValueError("batch leaves must have a leading example axis.")
    size = leaves[0].shape[0]
    if size == 0 or any(leaf.shape[0] != size for leaf in leaves):
        raise ValueError("batch leaves must share a nonempty leading example axis.")
    return size


class _ArrayFetcher:
    """Gathers the minibatch on device; the dataset never leaves the accelerator.

    ``batch_size`` of None means the whole dataset, every step.
    """

    def __init__(self, batch, batch_size: Optional[int], num_examples: int):
        self._batch = batch
        self._batch_size = batch_size
        self._num_examples = num_examples
        self.error: Optional[Exception] = None

    def __call__(self, key):
        if self._batch_size is None:
            return self._batch
        indices = jax.random.randint(key, (self._batch_size,), 0, self._num_examples)
        return jax.tree.map(lambda leaf: leaf[indices], self._batch)


class _StreamFetcher:
    """Pulls the batch from the user's iterator, one per step.

    The pull happens inside the compiled step, where an exception would
    surface as an opaque callback failure. The original is kept so the caller
    can re-raise it with its own traceback.
    """

    def __init__(self, stream, spec):
        self._stream = stream
        self._spec = spec
        self.error: Optional[Exception] = None
        self.count = 0

    def __call__(self, key):
        return io_callback(self._pull, self._spec, ordered=True)

    def _pull(self):
        try:
            batch = next(self._stream)
        except StopIteration as exc:
            self.error = RuntimeError(
                "the data stream ran out of batches. A stream is the caller's "
                "iterator and is not restarted, so it has to hold at least as "
                "many batches as the run takes steps; lower max_steps or pass "
                "an endless iterator."
            )
            raise self.error from exc
        try:
            batch = jax.tree.map(jnp.asarray, batch)
            self.count += 1
        except Exception as exc:  # noqa: BLE001 - re-raised by the caller
            self.error = exc
            raise
        return batch


def _make_fetcher(batch, config: FitConfig):
    """The per-step batch producer for this run."""
    if config.data.stream:
        if "batch_size" in config.explicit:
            raise ValueError(
                "batch_size cannot be set for a stream: the stream decides its "
                "own batch size. Pass an array instead, or drop batch_size."
            )
        source = iter(batch)
        try:
            first = next(source)
        except StopIteration as exc:
            raise ValueError("the data stream is empty.") from exc
        first = jax.tree.map(jnp.asarray, first)
        _check_batch(first)
        # The first batch was only looked at to see that the format is right;
        # it is also the first batch of the run.
        stream = itertools.chain([first], source)
        return _StreamFetcher(stream, spec_from_value(first))
    batch = jax.tree.map(jnp.asarray, batch)
    size = _check_batch(batch)
    batch_size = config.batch_size
    if batch_size is None or batch_size >= size:
        return _ArrayFetcher(batch, None, size)
    return _ArrayFetcher(batch, batch_size, size)


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------


def _serve(spec: DataSpec, batch):
    """The arguments the loss receives for one batch.

    The entry named by the spec is handed over positionally, every other entry
    as a keyword argument. A spec with no primary hands over the batch whole.
    """
    if spec.primary is None:
        return batch, {}
    if not is_mapping(batch):
        raise ValueError(
            f"the data spec names {spec.primary!r}, so a batch must be a "
            f"mapping; got {type(batch).__name__}."
        )
    if spec.primary not in batch:
        raise ValueError(
            f"a batch must have a {spec.primary!r} entry; got keys {sorted(batch)}."
        )
    return batch[spec.primary], {
        name: value for name, value in batch.items() if name != spec.primary
    }


def _make_body(loss_with_aux, tx, ema_decay, spec: DataSpec, fetcher):
    """The pure update: one step of the loop.

    The key is split three ways per step -- next state, batch, loss -- which
    is what fixes the random sequence of a run.
    """

    def body(state, _):
        next_rng, batch_key, loss_key = jax.random.split(state.rng, 3)
        batch = fetcher(batch_key)
        primary, context = _serve(spec, batch)
        (loss, (aux, metrics)), grads = jax.value_and_grad(loss_with_aux, has_aux=True)(
            state.params, loss_key, primary, state.aux, **context
        )
        current = cast(optax.Params, state.params)
        updates, opt_state = tx.update(grads, state.opt_state, current)
        params = cast(PyTree[Array], optax.apply_updates(current, updates))
        ema = None
        if ema_decay is not None:
            decay = float(ema_decay)
            ema = jax.tree.map(
                lambda old, new: (decay * old + (1 - decay) * new).astype(new.dtype),
                state.ema_params,
                params,
            )
        new_state = state._replace(
            params=params,
            opt_state=opt_state,
            step=state.step + 1,
            rng=next_rng,
            ema_params=ema,
            aux=aux,
        )
        return new_state, FitInfo(loss, metrics)

    return body


def _last(info: FitInfo) -> FitInfo:
    """The last update of a chunk."""
    metrics = None
    if info.metrics is not None:
        metrics = jax.tree.map(lambda leaf: leaf[-1], info.metrics)
    return FitInfo(info.loss[-1], metrics)


def _concat(chunks):
    """The history of a run: the losses stacked, and the metrics with them."""
    if not chunks:
        return jnp.zeros((0,)), None
    losses = jnp.concatenate([chunk.loss for chunk in chunks])
    metrics = jax.tree.map(
        lambda *leaves: jnp.concatenate(leaves), *(chunk.metrics for chunk in chunks)
    )
    return losses, metrics


def _warn_if_non_finite(losses: Array) -> None:
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


def _fit(
    loss_with_aux: Callable,
    params: PyTree[Array],
    rng: RngKey,
    batch,
    *,
    config: FitConfig,
    state: Optional[FitState],
    callbacks,
    aux=None,
) -> FitResult:
    """The loop: everything a run needs, given a loss that carries the aux."""
    check_config(config)
    for callback in callbacks:
        if not isinstance(callback, FitCallback) and not callable(callback):
            raise TypeError(
                "callbacks must be FitCallback objects or callables taking "
                f"(state, info); got {type(callback).__name__}."
            )
    fetcher = _make_fetcher(batch, config)
    tx = cast(optax.GradientTransformation, config.optimizer)
    body = _make_body(loss_with_aux, tx, config.ema_decay, config.data, fetcher)
    if state is None:
        state = FitState(
            params,
            tx.init(cast(optax.Params, params)),
            jnp.asarray(0, jnp.int32),
            rng,
            params if config.ema_decay is not None else None,
            aux,
        )

    try:
        traced = any(is_tracing(leaf) for leaf in jax.tree.leaves(state))
        if traced:
            if callbacks:
                raise ValueError(
                    "Callbacks run on the host between compiled chunks, so they "
                    "cannot be used inside jit. Call fit outside jit, or drop the "
                    "callbacks."
                )
            if config.max_steps:
                scanned, info = jax.lax.scan(body, state, None, length=config.max_steps)
                state = cast(FitState, scanned)
                losses, metrics = info.loss, info.metrics
            else:
                losses, metrics = jnp.zeros((0,)), None
            return FitResult(state, losses, metrics, config.use_ema)

        target = int(state.step) + config.max_steps
        chunks = []
        stop = _dispatch(callbacks, "on_fit_begin", state)
        if config.max_steps:
            # One compiled chunk, reused for every chunk of the run; only a
            # shorter final chunk needs another compilation.
            run = jax.jit(
                lambda carry, length: jax.lax.scan(body, carry, None, length=length),
                static_argnums=(1,),
            )
            while int(state.step) < target and not stop:
                length = min(config.callback_every, target - int(state.step))
                stepped, chunk = run(state, length)
                state = cast(FitState, stepped)
                chunks.append(chunk)
                stop = _dispatch(callbacks, "on_step_end", state, _last(chunk))
        losses, metrics = _concat(chunks)
        _warn_if_non_finite(losses)
        result = FitResult(state, losses, metrics, config.use_ema)
        _dispatch(callbacks, "on_fit_end", result)
        return result
    except Exception:
        # Also surface asynchronous iterator failures with their own traceback.
        if fetcher.error is not None:
            raise fetcher.error from None
        raise


def fit(
    loss_fn: Callable,
    params: PyTree[Array],
    rng: RngKey,
    batch,
    *,
    config: Optional[FitConfig] = None,
    state: Optional[FitState] = None,
    callbacks=(),
) -> FitResult:
    """Minimize ``loss_fn`` over ``params`` with minibatch gradient descent.

    Args:
        loss_fn: ``loss_fn(params, rng, primary, **context) -> loss``, where
            ``primary`` is the batch entry the data spec names and
            ``context`` the remaining ones. It may return ``(loss, metrics)``
            instead, and the metrics are stacked into the result.
        params: Pytree of trainable parameters. Ignored when ``state`` is
            given.
        rng: PRNG key consumed for minibatching and the per-step loss.
            Ignored when ``state`` is given.
        batch: One batch, or -- when the data spec says so -- an iterator of
            batches. A batch is a mapping from names to arrays whose leaves
            share a leading example axis.
        config: How to train, and how the data reaches the loss. Defaults to
            :class:`FitConfig`, which has no data spec and so must be
            completed before it can be used.
        state: Continue from this state, keeping the parameters, optimizer
            state, RNG and averaging. ``max_steps`` is the number of
            *additional* updates.
        callbacks: :class:`FitCallback` objects, or plain callables
            ``callback(state, info)``. They run on the host between compiled
            chunks; returning False stops the run.

    Returns:
        A :class:`FitResult` holding the final state, the losses and whatever
        metrics the loss reported.

    Note:
        Array-data fitting can be enclosed in ``jax.jit``, in which case it is
        a single ``lax.scan`` with no host effects and can be differentiated.
        Callbacks are not available there: they need the host. Module-backed
        models are fitted through
        :meth:`probjax.nn.training.FitMixin.fit`.
    """
    config = FitConfig() if config is None else config

    def loss_with_aux(params, rng, primary, aux, **context):
        out = loss_fn(params, rng, primary, **context)
        loss, metrics = out if isinstance(out, tuple) else (out, None)
        return loss, (aux, metrics)

    try:
        return _fit(
            loss_with_aux,
            params,
            rng,
            batch,
            config=config,
            state=state,
            callbacks=callbacks,
        )
    except Exception:
        raise
