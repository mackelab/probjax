"""``FitMixin``: ``model.fit(rng, data)`` for ``nnx`` modules.

This is the only module of the training package that knows about flax; the loop
itself works on plain parameter pytrees.
"""

import inspect
from typing import Any, Callable

from flax import nnx
from jaxtyping import Array

from probjax.nn.training.loop import FitResult, fit
from probjax.utils.typing import RngKey

__all__ = ["FitMixin"]

# Arguments that configure the loop. Any other keyword passed to ``model.fit`` is
# an extra for the model's own batch, handed to ``FitMixin._fit_batch``.
_LOOP_ARGUMENTS = frozenset(inspect.signature(fit).parameters) - {
    "loss_fn",
    "params",
    "rng",
    "batch",
}


class FitMixin:
    """Adds ``model.fit(rng, data, ...)`` to an ``nnx.Module`` with a ``loss``.

    The loop trains on whatever pytree (or stream of pytrees) the model hands it,
    so nothing here fixes what a batch looks like. A subclass decides by
    overriding any of:

    * ``_fit_batch(data, **extras)``: build the batch from ``fit``'s arguments.
      Keyword arguments of ``model.fit`` that the loop does not know arrive as
      ``extras``. Default: ``data`` unchanged, and extras are rejected.
    * ``_fit_loss(rng, batch)``: map one batch to a loss (or ``(loss, metrics)``
      with ``has_aux=True``). Default: ``self.loss(rng, batch)``.
    * ``_prepare_fit(data)``: set up state from the raw data before training.
    * ``_default_fit_kwargs()``: family defaults for the loop's arguments.
    * ``_fit_param_filter()``: which variables are trained (default: ``nnx.Param``).
    * ``_fit_callbacks()``: callbacks run in addition to the caller's.
    """

    #: Provided by the model: the per-batch loss the loop minimises.
    loss: Callable[..., Any]

    def _default_fit_kwargs(self) -> dict:
        """Model-family defaults for :func:`fit`, overridable per subclass.

        Anything the caller passes explicitly wins, so this only shifts the
        starting point for a family whose loss landscape is known to want
        something other than plain constant-rate Adam.
        """
        return {}

    def _prepare_fit(self, data):
        """Prepare model state before splitting, e.g. fit standardization."""
        return data

    def _fit_batch(self, data, **extras):
        """Build the batch pytree (or batch stream) the loop trains on."""
        if extras:
            raise TypeError(
                f"{type(self).__name__}.fit() got unexpected keyword arguments "
                f"{sorted(extras)}."
            )
        return data

    def _fit_loss(self, rng, batch):
        """Override batch-to-loss dispatch without replacing the fit loop."""
        return self.loss(rng, batch)

    def _fit_param_filter(self):
        """NNX filter selecting trainable parameters (default: all Param)."""
        return nnx.Param

    def _fit_callbacks(self):
        """Model callbacks prepended to those explicitly supplied by the caller."""
        return ()

    def fit(self, rng: RngKey, data, **kwargs) -> Array | FitResult:
        """Train in place; return losses, or FitResult with return_result=True.

        ``kwargs`` are the arguments of :func:`fit` (``num_steps``,
        ``learning_rate``, ``schedule``, ``callbacks``, ...); anything else goes
        to the model's own batch, see ``_fit_batch``.

        ``ema_decay`` enables averaging; ``use_ema=True`` installs the averaged
        parameters at completion. Non-parameter state (e.g. BatchNorm statistics
        and RNG counters) is carried through training and installed without EMA.
        Existing train/eval flags are respected; call ``model.train()`` first
        when needed. With has_aux=True, loss returns (loss, metrics). Each call
        starts a fresh optimizer and EMA unless initial_state is supplied.
        Resuming uses that snapshot's variables, not subsequent model edits.
        Compatible with nnx.jit when preprocessing hooks are traceable and data
        is an array pytree; custom host preprocessing should run before JIT.
        """
        extras = {k: kwargs.pop(k) for k in list(kwargs) if k not in _LOOP_ARGUMENTS}
        fit_kwargs = {**self._default_fit_kwargs(), **kwargs}
        data = self._prepare_fit(data)
        batch = self._fit_batch(data, **extras)
        param_filter = self._fit_param_filter()
        graphdef, params, rest = nnx.split(self, param_filter, ...)

        def loss_fn(params, rng, batch, model_state):
            model = nnx.merge(graphdef, params, model_state, copy=True)
            output = model._fit_loss(rng, batch)
            rest = nnx.state(model, nnx.Not(param_filter))
            if fit_kwargs.get("has_aux", False):
                loss, metrics = output
                return loss, (rest, metrics)
            return output, rest

        fit_kwargs["callbacks"] = (
            *self._fit_callbacks(),
            *fit_kwargs.get("callbacks", ()),
        )
        want_result = fit_kwargs.pop("return_result", False)
        result = fit(
            loss_fn,
            params,
            rng,
            batch,
            model_state=None if fit_kwargs.get("initial_state") is not None else rest,
            return_result=True,
            **fit_kwargs,
        )
        assert isinstance(result, FitResult)
        nnx.update(self, result.params, result.state.model_state)
        return result if want_result else result.losses
