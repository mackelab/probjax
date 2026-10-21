"""
Gradient-based training (:mod:`probjax.nn.training`)
====================================================

A framework-agnostic training loop. :func:`fit` minimises any
``loss_fn(params, rng, primary, **context)`` over a params pytree with optax.

A run is described by a :class:`FitConfig`, which says how to train *and* how
the data reaches the loss: a :class:`DataSpec` names the batch entry handed to
the loss positionally, and says whether the source is one batch or a stream of
them. Nothing in the loop assumes a particular NN library or batch structure:

>>> config = FitConfig(data=DataSpec(primary="data"), max_steps=300)
>>> result = fit(loss_fn, params, key, {"data": x})       # one array
>>> result = fit(loss_fn, params, key, my_dataloader,      # a stream of batches
...              config=DataSpec(primary="data", stream=True))

Module-backed models (the families in :mod:`probjax.nn.generative`) get the
convenient ``model.fit(rng, data)`` via :class:`FitMixin`, which snapshots the
current module graph and carries non-parameter state through each update:

>>> flow = maf(2, 5, rngs=nnx.Rngs(0))
>>> result = flow.fit(jax.random.key(0), samples)
>>> flow.logpdf(samples)  # trained in place

This is the object-layer counterpart of the scipy-style classmethod
``rv_generic.fit`` (closed-form / optimizer MLE for parametric families).

Extension points
----------------
Each part is replaced by passing a different one, not by editing the loop:

* **Loss**: any ``loss_fn(params, rng, primary, **context)``; return
  ``(loss, metrics)`` to have the metrics stacked into the result.
* **Optimizer**: ``FitConfig.optimizer`` takes any
  ``optax.GradientTransformation``. Without one, the configuration's own
  ``learning_rate``, ``schedule`` (a name or any optax schedule) and
  ``clip_norm`` build the default.
* **Data**: :class:`DataSpec` says how a batch reaches the loss. An array is
  one batch; a stream yields them and picks its own batch size.
* **Observation**: :class:`FitCallback` objects run on the host between
  compiled chunks; returning False stops the run. Resume from a
  :class:`FitState` with ``state=``.
* **Modules**: subclass behaviour of :class:`FitMixin` through its
  ``default_config``, ``_fit_batch``, ``_fit_loss``, ``_prepare_fit``,
  ``_fit_param_filter`` and ``_fit_callbacks`` hooks.

Layout: ``config``, ``state``, ``trainer`` and ``mixin``; only ``mixin``
imports flax.
"""

from probjax.nn.training.config import DataSpec, FitConfig
from probjax.nn.training.mixin import FitMixin
from probjax.nn.training.state import FitInfo, FitResult, FitState
from probjax.nn.training.trainer import FitCallback, fit, take_batches

__all__ = [
    "DataSpec",
    "FitCallback",
    "FitConfig",
    "FitInfo",
    "FitMixin",
    "FitResult",
    "FitState",
    "fit",
    "take_batches",
]
