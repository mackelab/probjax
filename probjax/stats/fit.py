"""
Gradient-based fitting (:mod:`probjax.stats.fit`)
=================================================

A framework-agnostic training loop: :func:`fit` minimizes any
``loss_fn(params, rng, batch)`` over a params pytree with optax, where
``batch`` is an arbitrary pytree (e.g. ``{"data": x, "context": c}``) whose
leaves share the leading example axis. Nothing here assumes a particular NN
library.

Module-backed models (the families in :mod:`probjax.nn.generative`) get the
convenient ``model.fit(rng, data)`` via :class:`FitMixin`, which lazily builds
the pure ``loss_fn`` + params from the module once per instance — the stable
function identity keeps the jitted train step cached across calls:

>>> flow = maf(2, 5, rngs=nnx.Rngs(0))
>>> losses = flow.fit(jax.random.key(0), samples)
>>> flow.logpdf(samples)  # trained in place

This is the object-layer counterpart of the scipy-style classmethod
``rv_generic.fit`` (closed-form / optimizer MLE for parametric families).
"""

import weakref
from functools import lru_cache
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["fit", "FitMixin"]


@lru_cache(maxsize=64)
def _jitted_train_step(loss_fn, tx):
    """Build (and cache) the jitted optimization step for a (loss_fn, optimizer) pair."""
    import optax

    @jax.jit
    def train_step(params, opt_state, rng, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, rng, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def fit(
    loss_fn,
    params,
    rng: RngKey,
    batch: object,
    *,
    num_steps: int = 1000,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-3,
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
        num_steps: Number of gradient steps.
        batch_size: Minibatch size; ``None`` uses the full dataset each step.
        learning_rate: Adam learning rate, used when ``optimizer`` is None.
        optimizer: Optional ``optax.GradientTransformation`` overriding the
            default ``optax.adam(learning_rate)``.

    Returns:
        ``(trained_params, losses)`` where ``losses`` has shape ``(num_steps,)``.
    """
    import optax

    batch = jax.tree.map(jnp.asarray, batch)
    leaves = jax.tree.leaves(batch)
    if not leaves:
        raise ValueError("batch must contain at least one array leaf.")
    num_examples = leaves[0].shape[0]

    tx = optimizer if optimizer is not None else optax.adam(learning_rate)
    opt_state = tx.init(params)
    train_step = _jitted_train_step(loss_fn, tx)

    losses = []
    for _ in range(num_steps):
        rng, rng_batch, rng_loss = jax.random.split(rng, 3)
        if batch_size is None or batch_size >= num_examples:
            minibatch = batch
        else:
            idx = jax.random.randint(rng_batch, (batch_size,), 0, num_examples)
            minibatch = jax.tree.map(lambda a: a[idx], batch)
        params, opt_state, loss = train_step(params, opt_state, rng_loss, minibatch)
        losses.append(loss)

    return params, jnp.stack(losses)


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

        loss_fn = _pure_loss_fn(self)
        params = nnx.state(self, nnx.Param)
        batch = {"data": data} if context is None else {"data": data, "context": context}
        params, losses = fit(loss_fn, params, rng, batch, **fit_kwargs)
        nnx.update(self, params)
        return losses
