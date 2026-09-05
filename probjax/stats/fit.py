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

Nothing here assumes a particular NN library. The loop is a single
``jax.lax.scan``: it compiles once no matter how many steps are requested, and
runs end to end without returning to Python -- a streamed batch arrives through
an ordered ``io_callback``, and ``on_step`` reports progress the same way.

Module-backed models (the families in :mod:`probjax.nn.generative`) get the
convenient ``model.fit(rng, data)`` via :class:`FitMixin`, which lazily builds
the pure ``loss_fn`` + params from the module once per instance:

>>> flow = maf(2, 5, rngs=nnx.Rngs(0))
>>> losses = flow.fit(jax.random.key(0), samples)
>>> flow.logpdf(samples)  # trained in place

This is the object-layer counterpart of the scipy-style classmethod
``rv_generic.fit`` (closed-form / optimizer MLE for parametric families).
"""

import warnings
import weakref
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jaxtyping import Array

from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["fit", "FitMixin", "is_batch_stream", "take_batches"]

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
        return int(num_steps)
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

    if schedule == "constant":
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


def _make_step(loss_fn, tx, fetch, on_step, log_every: int):
    """One training step, written for ``lax.scan``.

    ``fetch(key)`` returns the minibatch: an on-device gather for an array
    dataset, or an ``io_callback`` into a host iterator for a stream.
    """
    import optax

    want_callback = on_step is not None

    def host_callback(step, loss):
        result = on_step(int(step), float(loss))
        return np.asarray(result is False)

    def body(carry, _):
        params, opt_state, rng, stop, step = carry
        # Split in the same order and arity as the original Python loop so the
        # scanned version reproduces it exactly for a given seed.
        rng, rng_batch, rng_loss = jax.random.split(rng, 3)

        def run(_):
            minibatch = fetch(rng_batch)
            loss, grads = jax.value_and_grad(loss_fn)(params, rng_loss, minibatch)
            updates, new_opt = tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_opt, loss

        if not want_callback:
            params, opt_state, loss = run(None)
        else:

            def skip(_):
                # Early stopping cannot break a scan; the remaining iterations
                # run but do no work, and the caller drops their losses.
                nan = jnp.asarray(jnp.nan, jnp.result_type(float))
                return params, opt_state, nan

            params, opt_state, loss = jax.lax.cond(stop, skip, run, operand=None)

        if want_callback:
            fire = jnp.logical_and(~stop, (step % log_every) == 0)
            stop = jnp.logical_or(
                stop,
                jax.lax.cond(
                    fire,
                    lambda: io_callback(
                        host_callback,
                        jax.ShapeDtypeStruct((), bool),
                        step,
                        loss,
                        ordered=True,
                    ),
                    lambda: jnp.asarray(False),
                ),
            )

        return (params, opt_state, rng, stop, step + 1), loss

    return body


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
) -> Tuple[object, Array]:
    """Minimize ``loss_fn`` over ``params`` with minibatch gradient descent.

    Args:
        loss_fn: ``loss_fn(params, rng, batch) -> scalar``. Must be a stable
            function object across calls to benefit from the cached jitted
            step (avoid rebuilding it per call).
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
        log_every: Cadence for ``on_step``. Ignored when ``on_step`` is None.

    Returns:
        ``(trained_params, losses)``. ``losses`` has shape ``(num_steps,)``,
        or is truncated at the stopping step if ``on_step`` asked to stop.

    Warns:
        RuntimeWarning: if any step produced a non-finite loss. The parameters
            are returned as-is rather than repaired -- once a NaN gradient has
            been applied the run is dead, and silently continuing would hide it.

    Note:
        The loop is a single ``jax.lax.scan``, so it compiles once regardless of
        ``num_steps`` and runs without returning to Python. Two consequences:
        losses arrive only when the run finishes rather than step by step (use
        ``on_step`` to watch it live), and a diverged run still executes its
        remaining iterations.
    """
    if log_every < 1:
        raise ValueError(f"log_every must be at least 1; got {log_every}.")

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
        per_batch = leaves[0].shape[0]
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
        num_examples = leaves[0].shape[0]
        batch_size = _resolve_batch_size(batch_size, num_examples)
        num_steps = _resolve_num_steps(num_steps, num_examples, batch_size)
        fetch = _array_fetch(batch, batch_size, num_examples)

    if optimizer is not None:
        tx = optimizer
    else:
        tx = _build_optimizer(learning_rate, num_steps, schedule, clip_norm)
    opt_state = tx.init(params)

    body = _make_step(loss_fn, tx, fetch, on_step, log_every)
    init = (params, opt_state, rng, jnp.asarray(False), jnp.asarray(0, jnp.int32))
    try:
        # Not wrapped in jit: the scan is one XLA computation either way, and
        # jitting here would key the cache on a closure rebuilt every call.
        (params, _, _, stopped, _), losses = jax.lax.scan(
            body, init, None, length=num_steps
        )
    except Exception:
        # A bad batch fails inside the callback, where JAX wraps it in a
        # JaxRuntimeError over a traceback through the whole scan machinery.
        # The stream's own error is the one the user can act on.
        error = getattr(fetch, "stream_state", {}).get("error")
        if error is not None:
            raise error from None
        raise

    if on_step is not None and bool(stopped):
        # Steps after the stop ran as no-ops and reported NaN; drop them rather
        # than hand back losses that look like divergence.
        ran = int(jnp.sum(jnp.asarray(~jnp.isnan(losses), jnp.int32)))
        losses = losses[:ran]

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
    return params, losses


# Per-model pure loss functions, built lazily once per instance so their
# identity is stable (keeps _jitted_train_step's cache warm across fit calls).
_PURE_LOSS_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _pure_loss_fn(model):
    from flax import nnx  # the only flax-aware spot: split/merge boundary

    cached = _PURE_LOSS_CACHE.get(model)
    if cached is not None:
        return cached

    graphdef, _, rest = nnx.split(model, nnx.Param, ...)

    def loss_fn(params, rng, batch):
        m = nnx.merge(graphdef, params, rest)
        if isinstance(batch, dict):
            kwargs = {k: v for k, v in batch.items() if k != "data"}
            return m.loss(rng, batch["data"], **kwargs)
        return m.loss(rng, batch)

    _PURE_LOSS_CACHE[model] = loss_fn
    return loss_fn


class FitMixin:
    """Adds ``model.fit(rng, data, ...)`` for modules with a ``loss`` method."""

    def _default_fit_kwargs(self) -> dict:
        """Model-family defaults for :func:`fit`, overridable per subclass.

        Anything the caller passes explicitly wins, so this only shifts the
        starting point for a family whose loss landscape is known to want
        something other than plain constant-rate Adam.
        """
        return {}

    def fit(
        self,
        rng: RngKey,
        data: ArrayLike,
        *,
        context: Optional[ArrayLike] = None,
        weights: Optional[ArrayLike] = None,
        **fit_kwargs,
    ) -> Array:
        """Train this model in place; returns per-step losses."""
        from flax import nnx

        fit_kwargs = {**self._default_fit_kwargs(), **fit_kwargs}
        loss_fn = _pure_loss_fn(self)
        params = nnx.state(self, nnx.Param)
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
        params, losses = fit(loss_fn, params, rng, batch, **fit_kwargs)
        nnx.update(self, params)
        return losses
