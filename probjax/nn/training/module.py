"""Training an ``nnx`` module in place: the only flax-aware part of training."""

from __future__ import annotations

from typing import Any, Callable, Optional

from flax import nnx

from probjax.nn.training.loop import FitResult, _fit
from probjax.utils.typing import RngKey

__all__ = ["fit_module"]


def _call_loss(model, rng, batch):
    return model.loss(rng, batch)


def fit_module(
    model: nnx.Module,
    rng: RngKey,
    data,
    *,
    loss: Optional[Callable[[Any, RngKey, Any], Any]] = None,
    trainable: Any = nnx.Param,
    **options,
) -> FitResult:
    """Train ``model`` in place and return the run's result.

    Args:
        model: The module to train.
        rng: Key for minibatching and the loss.
        data: A batch or a stream of batches, as for
            :func:`~probjax.nn.training.fit`.
        loss: ``loss(model, rng, batch) -> loss`` or ``(loss, metrics)``.
            Defaults to ``model.loss(rng, batch)``.
        trainable: ``nnx`` filter of the variables to train. All other state
            (BatchNorm statistics, RNG counters, frozen parameters) is carried
            through training and written back without averaging.
        **options: The options of :func:`~probjax.nn.training.fit`. With
            ``state=`` the run continues from that snapshot, not from the
            model's current values.
    """
    loss = _call_loss if loss is None else loss
    graphdef, params, rest = nnx.split(model, trainable, ...)

    def loss_with_aux(params, aux, rng, batch):
        merged = nnx.merge(graphdef, params, aux, copy=True)
        out = loss(merged, rng, batch)
        value, metrics = out if isinstance(out, tuple) else (out, None)
        return value, (nnx.state(merged, nnx.Not(trainable)), metrics)

    result = _fit(loss_with_aux, params, rng, data, aux=rest, **options)
    nnx.update(model, result.params, result.state.aux)
    return result
