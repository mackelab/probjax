"""Batch plumbing for the training loop: sizing defaults, loader detection and
the fetchers that turn a batch source into per-step batches.

A batch is any pytree whose leaves share a leading example axis. Nothing here
knows what the leaves mean.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import io_callback

from probjax.nn.export import spec_from_value
from probjax.utils.guards import ensure_int, is_array_like

__all__ = [
    "is_batch_stream",
    "take_batches",
]


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
        return ensure_int(
            "num_steps",
            num_steps,
            min_value=0,
            message="num_steps must be a nonnegative integer or 'auto'.",
        )
    per_epoch = max(1, num_examples // (batch_size or num_examples))
    return int(min(max(_AUTO_EPOCHS * per_epoch, _AUTO_MIN_STEPS), _AUTO_MAX_STEPS))


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

    So a single batch that groups several arrays must be a dict (or another
    pytree that is not a list or tuple) -- ``{"x": x, "y": y}`` -- not a tuple.
    """
    if is_array_like(data) or isinstance(data, (dict, int, float, complex)):
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
    return spec_from_value(batch)


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

    def fetch(key):
        return io_callback(pull, spec, ordered=True)

    fetch.stream_state = state  # type: ignore[attr-defined]
    return fetch
