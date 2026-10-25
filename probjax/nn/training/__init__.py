"""
Gradient-based training (:mod:`probjax.nn.training`)
====================================================

Three levels, each built on the one before:

* :func:`train_step` -- one pure update ``(state, batch) -> (state, info)``
  for your own loop (sharding, donation, custom logging).
* :func:`fit` -- minimise ``loss_fn(params, rng, batch)`` over a params
  pytree. An array batch is minibatched on the device; any other iterable is
  a stream, one batch per update.
* :func:`fit_module` -- the same for an ``nnx`` module, trained in place.
  Generative models wrap it as ``model.fit(rng, data, **options)``.

>>> result = fit(loss_fn, params, key, {"x": x, "y": y}, num_steps=300)
>>> result = fit(loss_fn, params, key, my_dataloader, num_steps=300)
>>> flow.fit(key, samples, learning_rate=3e-4)

Options are plain keyword arguments (``num_steps``, ``batch_size``,
``optimizer`` or ``learning_rate``/``schedule``/``clip_norm``, ``ema_decay``,
``use_ema``, ``callback``, ``every``, ``state``); see :func:`fit`.

Only ``module`` imports flax.
"""

from probjax.nn.training.loop import (
    FitInfo,
    FitResult,
    FitState,
    default_optimizer,
    fit,
    init_state,
    is_stream,
    take_batches,
    train_step,
)
from probjax.nn.training.module import fit_module

__all__ = [
    "FitInfo",
    "FitResult",
    "FitState",
    "default_optimizer",
    "fit",
    "fit_module",
    "init_state",
    "is_stream",
    "take_batches",
    "train_step",
]
