"""
Gradient-based training (:mod:`probjax.nn.training`)
====================================================

A framework-agnostic training loop: :func:`fit` minimizes any
``loss_fn(params, rng, batch)`` over a params pytree with optax, where
``batch`` is either one batch -- any pytree whose leaves share the leading
example axis, such as a bare array or ``{"x": x, "y": y}`` -- or an iterable of
batches, for data that does not fit in memory:

>>> params, losses = fit(loss_fn, params, key, {"x": x})  # whole array
>>> params, losses = fit(loss_fn, params, key, my_dataloader)  # streamed

Nothing in the loop assumes a particular NN library or batch structure. Without
general callbacks it is a single ``jax.lax.scan``. General callbacks run between
compiled scan chunks; streamed batches and the legacy loss-only ``on_step`` hook
use ordered ``io_callback`` calls. Optional EMA stays on the device alongside raw
parameters.

Module-backed models (the families in :mod:`probjax.nn.generative`) get the
convenient ``model.fit(rng, data)`` via :class:`FitMixin`, which snapshots
the current module graph and carries non-parameter state through each update:

>>> flow = maf(2, 5, rngs=nnx.Rngs(0))
>>> losses = flow.fit(jax.random.key(0), samples)
>>> flow.logpdf(samples)  # trained in place

This is the object-layer counterpart of the scipy-style classmethod
``rv_generic.fit`` (closed-form / optimizer MLE for parametric families).

Extension points
----------------
Each part is replaced by passing a different one, not by editing the loop:

* **Loss**: any ``loss_fn(params, rng, batch)``; add ``model_state`` for mutable
  state and ``has_aux=True`` for metrics.
* **Optimizer**: ``optimizer=`` takes any ``optax.GradientTransformation``.
  Without it, ``learning_rate``, ``schedule`` (a name or any optax schedule) and
  ``clip_norm`` configure the default.
* **Step**: ``kernel=`` takes a complete :class:`FitKernel`; build one with
  :func:`build_fit_kernel` or write your own ``init``/``step`` pair.
* **Observation**: :class:`FitCallback` objects (``callbacks=``) or a plain
  ``on_step`` function; resume from a :class:`FitState` via ``initial_state=``.
* **Data**: batches are opaque pytrees, and a loader is any re-iterable of them.
* **Modules**: subclass behaviour of :class:`FitMixin` through its ``_fit_batch``,
  ``_fit_loss``, ``_prepare_fit``, ``_default_fit_kwargs``, ``_fit_param_filter``
  and ``_fit_callbacks`` hooks.

Layout: ``batches``, ``kernel`` and ``loop`` are flax-free; only ``mixin``
imports flax.
"""

from probjax.nn.training.batches import is_batch_stream, take_batches
from probjax.nn.training.kernel import (
    FitInfo,
    FitKernel,
    FitState,
    build_fit_kernel,
)
from probjax.nn.training.loop import FitCallback, FitResult, fit
from probjax.nn.training.mixin import FitMixin

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
