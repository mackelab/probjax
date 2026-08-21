"""
Gradient-based fitting (:mod:`probjax.stats.fit`)
=================================================

A framework-agnostic training loop: :func:`fit` minimizes any
``loss_fn(params, rng, batch)`` over a params pytree with optax, where
``batch`` is an arbitrary pytree (e.g. ``{"data": x, "context": c}``) whose
leaves share the leading example axis. Nothing here assumes a particular NN
library. The loop is a single ``jax.lax.scan``: it compiles once no matter
how many steps are requested, and runs end to end without returning to Python.

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
from jaxtyping import Array

from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["fit", "FitMixin"]

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


def _make_step(loss_fn, tx, fetch):
    """One training step, written for ``lax.scan``.

    ``fetch(key)`` returns the minibatch for the step.
    """
    import optax

    def body(carry, _):
        params, opt_state, rng = carry
        # Split in the same order and arity as the Python loop this replaces,
        # so the scanned version reproduces it exactly for a given seed.
        rng, rng_batch, rng_loss = jax.random.split(rng, 3)
        minibatch = fetch(rng_batch)
        loss, grads = jax.value_and_grad(loss_fn)(params, rng_loss, minibatch)
        updates, opt_state = tx.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, rng), loss

    return body


def _array_fetch(batch, batch_size, num_examples):
    """Minibatch by gathering on device; the dataset never leaves the accelerator."""
    if batch_size is None or batch_size >= num_examples:
        return lambda key: batch
    return lambda key: jax.tree.map(
        lambda a: a[jax.random.randint(key, (batch_size,), 0, num_examples)], batch
    )


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
) -> Tuple[object, Array]:
    """Minimize ``loss_fn`` over ``params`` with minibatch gradient descent.

    Args:
        loss_fn: ``loss_fn(params, rng, batch) -> scalar``. Must be a stable
            function object across calls to benefit from the cached jitted
            step (avoid rebuilding it per call).
        params: Pytree of trainable parameters.
        rng: PRNG key consumed for minibatching and the per-step loss.
        batch: Pytree of training arrays (e.g. ``{"data": x, "context": c}``
            or a bare array); all leaves share the leading example axis.
        num_steps: Number of gradient steps, or ``"auto"`` (the default) to
            scale with the dataset: enough steps for a fixed number of passes
            over it, clamped to [1000, 20000].
        batch_size: Minibatch size, or ``"auto"`` (the default) for
            ``min(num_examples, 512)``. ``None`` still means the full dataset
            every step, which was the previous default and stops being viable
            as the dataset grows.
        learning_rate: Adam learning rate, used when ``optimizer`` is None.
        schedule: ``"constant"`` or ``"warmup_cosine"`` (5% warmup, cosine decay
            to ``learning_rate / 1000``). Ignored when ``optimizer`` is given.
        clip_norm: Global gradient-norm clip; ``None`` disables. Ignored when
            ``optimizer`` is given.
        optimizer: Optional ``optax.GradientTransformation``. Supplying it takes
            full control, bypassing ``learning_rate``, ``schedule`` and
            ``clip_norm``.

    Returns:
        ``(trained_params, losses)`` where ``losses`` has shape ``(num_steps,)``.

    Warns:
        RuntimeWarning: if any step produced a non-finite loss. The parameters
            are returned as-is rather than repaired -- once a NaN gradient has
            been applied the run is dead, and silently continuing would hide it.

    Note:
        The loop is a single ``jax.lax.scan``, so it compiles once regardless of
        ``num_steps`` and runs without returning to Python. Two consequences:
        losses arrive only when the run finishes rather than step by step, and a
        diverged run still executes its remaining iterations.
    """
    batch = jax.tree.map(jnp.asarray, batch)
    leaves = jax.tree.leaves(batch)
    if not leaves:
        raise ValueError("batch must contain at least one array leaf.")
    num_examples = leaves[0].shape[0]
    batch_size = _resolve_batch_size(batch_size, num_examples)
    num_steps = _resolve_num_steps(num_steps, num_examples, batch_size)

    if optimizer is not None:
        tx = optimizer
    else:
        tx = _build_optimizer(learning_rate, num_steps, schedule, clip_norm)
    opt_state = tx.init(params)

    body = _make_step(loss_fn, tx, _array_fetch(batch, batch_size, num_examples))
    # Not wrapped in jit: the scan is one XLA computation either way, and
    # jitting here would key the cache on a closure rebuilt every call.
    (params, _, _), losses = jax.lax.scan(
        body, (params, opt_state, rng), None, length=num_steps
    )

    finite = jnp.isfinite(losses)
    if not bool(jnp.all(finite)):
        first = int(jnp.argmin(finite))
        warnings.warn(
            f"Training loss became non-finite at step {first} of {num_steps}; "
            "the returned parameters are unusable. Lower the learning rate, "
            "tighten clip_norm, or check the model for an unbounded transform.",
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
        **fit_kwargs,
    ) -> Array:
        """Train this model in place; returns per-step losses."""
        from flax import nnx

        fit_kwargs = {**self._default_fit_kwargs(), **fit_kwargs}
        loss_fn = _pure_loss_fn(self)
        params = nnx.state(self, nnx.Param)
        batch = (
            {"data": data} if context is None else {"data": data, "context": context}
        )
        params, losses = fit(loss_fn, params, rng, batch, **fit_kwargs)
        nnx.update(self, params)
        return losses
