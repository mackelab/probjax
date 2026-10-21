"""``FitMixin``: ``model.fit(rng, data)`` for ``nnx`` modules.

This is the only module of the training package that knows about flax; the loop
itself works on plain parameter pytrees.
"""

from typing import Any, Callable, Optional

from flax import nnx

from probjax.nn.training.config import DataSpec, FitConfig, merge_config
from probjax.nn.training.state import FitResult
from probjax.nn.training.trainer import _fit
from probjax.utils.typing import RngKey

__all__ = ["FitMixin"]


class FitMixin:
    """Adds ``model.fit(rng, data)`` to an ``nnx.Module`` with a ``loss``.

    The loop trains on whatever pytree the model hands it, so nothing here
    fixes what a batch looks like. A subclass decides by overriding any of:

    * ``default_config()``: the recipe this model trains with.
    * ``_fit_batch(data, config, **extras)``: build the batch from ``fit``'s
      arguments. Keyword arguments of ``model.fit`` that the loop does not
      know arrive as ``extras``. Default: ``data`` unchanged, extras rejected.
    * ``_fit_loss(rng, primary, **context)``: map one batch to a loss (or
      ``(loss, metrics)``). Default: ``self.loss(rng, primary, **context)``.
    * ``_prepare_fit(data, config)``: set up state from the raw data before
      training.
    * ``_fit_param_filter()``: which variables are trained (default:
      ``nnx.Param``).
    * ``_fit_callbacks()``: callbacks run in addition to the caller's.
    """

    #: Provided by the model: the per-batch loss the loop minimises.
    loss: Callable[..., Any]

    def default_config(self) -> FitConfig:
        """The recipe this model trains with.

        A configuration passed to :meth:`fit` overrides only the fields it
        names, so the rest of this recipe survives.
        """
        return FitConfig(data=DataSpec())

    def _prepare_fit(self, data, config: FitConfig):
        """Prepare model state before splitting, e.g. fit standardization."""
        return data

    def _fit_batch(self, data, config: FitConfig, **extras):
        """Build the batch pytree the loop trains on."""
        if extras:
            raise TypeError(
                f"{type(self).__name__}.fit() got unexpected keyword arguments "
                f"{sorted(extras)}. Training options go in the config= "
                "argument; any other keyword is passed to the model's batch."
            )
        return data

    def _fit_loss(self, rng, primary, **context):
        """Override batch-to-loss dispatch without replacing the fit loop."""
        return self.loss(rng, primary, **context)

    def _fit_param_filter(self):
        """NNX filter selecting trainable parameters (default: all Param)."""
        return nnx.Param

    def _fit_callbacks(self):
        """Model callbacks prepended to those explicitly supplied by the caller."""
        return ()

    def fit(
        self,
        rng: RngKey,
        data,
        *,
        config: Optional[FitConfig] = None,
        state=None,
        callbacks=(),
        **extras,
    ) -> FitResult:
        """Train in place; return the result.

        ``config`` says how to train and how the data reaches the loss; it
        overrides only the fields it names, so the model's own recipe is kept.
        Any other keyword argument is passed to the model's batch.

        ``ema_decay`` enables averaging and ``use_ema`` installs the averaged
        parameters at completion. Non-parameter state (e.g. BatchNorm
        statistics and RNG counters) is carried through training and installed
        without averaging. Existing train/eval flags are respected; call
        ``model.train()`` first when needed.

        Passing ``state`` continues from that snapshot -- its parameters,
        optimizer state, RNG and averaging -- rather than from the model's
        current state, so later edits to the model are not picked up. Each
        call otherwise starts a fresh optimizer.
        """
        base = self.default_config()
        config = base if config is None else merge_config(base, config)
        callbacks = (*self._fit_callbacks(), *callbacks)
        data = self._prepare_fit(data, config)
        batch = self._fit_batch(data, config, **extras)
        param_filter = self._fit_param_filter()
        graphdef, params, rest = nnx.split(self, param_filter, ...)

        def loss_with_aux(params, rng, primary, aux, **context):
            model = nnx.merge(graphdef, params, aux, copy=True)
            out = model._fit_loss(rng, primary, **context)
            loss, metrics = out if isinstance(out, tuple) else (out, None)
            return loss, (nnx.state(model, nnx.Not(param_filter)), metrics)

        result = _fit(
            loss_with_aux,
            params,
            rng,
            batch,
            config=config,
            state=state,
            callbacks=callbacks,
            aux=rest,
        )
        nnx.update(self, result.params, result.state.aux)
        return result
