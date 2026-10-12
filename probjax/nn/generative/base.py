"""Base class for generative-model families.

:class:`GenerativeModel` packs the cached, shape-polymorphic distribution
machinery from :mod:`probjax.nn.generative.sampling` so downstream families
(diffusion, flow matching, mean flow, normalizing flows, ...) only supply
what is genuinely theirs: how to turn base noise into data. The expensive,
shared parts — ``nnx.split``, ``jax.export`` with a symbolic batch axis,
and per-model caching / invalidation — live here.

A family implements ``loss`` and delegates its family-specific sampling
kernel to :meth:`GenerativeModel._build_exported_sampler`. Event descriptions
are pytree specs, so structured data works like a single flat array.
"""

from __future__ import annotations

from numbers import Integral
from typing import Callable

import jax
from flax import nnx

from probjax.nn.distribution import _ExportedLogPDF, _ModelDistribution, _export_logpdf
from probjax.nn.export import (
    clear_export_cache,
    get_cached_export,
    normalize_spec,
    spec_cache_key,
)
from probjax.nn.generative.sampling import (
    _ExportedSampler,
    _export_sampler,
    sample_normal,
)
from probjax.nn.training import FitMixin, FitResult, is_batch_stream
from probjax.utils.typing import Array, PyTree, RngKey

__all__ = ["GenerativeModel"]


def _affine_sample_base(self, rng, sample_shape, spec):
    """Gaussian base ``N(mu0, std0**2)``; flow models assign this as ``_sample_base``."""
    return sample_normal(
        rng,
        sample_shape,
        spec,
        loc=self.mu0.get_value(),
        scale=self.std0.get_value(),
    )


class _DataContextBatches:
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


class GenerativeModel(nnx.Module, FitMixin):
    """Shared base for generative families with cached exported samplers."""

    _event_dtype = jax.numpy.float32

    def _normalize_event_spec(self, event_spec, dtype=None):
        """Normalize shape shorthand without tying the network to that shape."""
        if event_spec is None:
            raise TypeError(
                "event_spec must specify an event dimension, shape, or pytree."
            )
        dtype = self._event_dtype if dtype is None else dtype

        def is_shape(value):
            return isinstance(value, (tuple, list)) and all(
                isinstance(dim, Integral) for dim in value
            )

        def to_spec(value):
            if isinstance(value, jax.ShapeDtypeStruct):
                spec = value
            elif isinstance(value, Integral) and not isinstance(value, bool):
                spec = jax.ShapeDtypeStruct((int(value),), dtype)
            elif is_shape(value):
                spec = jax.ShapeDtypeStruct(tuple(value), dtype)
            else:
                raise TypeError(
                    "event_spec leaves must be dimensions, shapes, or ShapeDtypeStruct."
                )
            if any(
                not isinstance(dim, Integral) or isinstance(dim, bool) or dim <= 0
                for dim in spec.shape
            ):
                raise ValueError(
                    "event_spec dimensions must be positive concrete integers; "
                    "override the event spec when changing event dimensions."
                )
            return spec

        spec = jax.tree.map(
            to_spec,
            event_spec,
            is_leaf=lambda value: (
                value is None
                or is_shape(value)
                or isinstance(value, jax.ShapeDtypeStruct)
            ),
        )
        if not jax.tree.leaves(spec):
            raise ValueError("event_spec must contain at least one array event.")
        return spec

    @property
    def event_spec(self):
        """Default event description; a shape/pytree override can be set later."""
        stored = getattr(self, "_default_event_spec", None)
        return None if stored is None else jax.tree.unflatten(*stored)

    @event_spec.setter
    def event_spec(self, value):
        self.set_event_spec(value)

    @property
    def event_shape(self):
        """Default event shape or shape pytree, excluding sample/batch axes."""
        spec = self.event_spec
        return None if spec is None else jax.tree.map(lambda leaf: leaf.shape, spec)

    def set_event_spec(self, event_spec):
        """Change the default for future views without resizing network params.

        Existing distribution views keep their bound event specs. Cached exports
        are invalidated because this changes static model metadata.
        """
        spec = self._normalize_event_spec(event_spec)
        leaves, tree = jax.tree.flatten(spec)
        self._default_event_spec = nnx.static((tree, tuple(leaves)))
        self._clear_distribution_cache()

    def loss(self, rng: RngKey, data: PyTree[Array], *args, **kwargs) -> Array:
        """Monte-Carlo training loss for a batch of data."""
        raise NotImplementedError

    def fit(
        self,
        rng: RngKey,
        data,
        *,
        context=None,
        weights=None,
        **fit_kwargs,
    ) -> Array | FitResult:
        """Train in place on ``data``: an array pytree or a stream of batches.

        ``context`` conditions the model and ``weights`` weight the examples;
        both are passed to :meth:`loss` as keyword arguments. With a stream,
        yield ``(data, context)`` pairs or dicts with a ``"data"`` key instead.
        The remaining arguments are those of :func:`probjax.nn.training.fit`.
        """
        extras = {
            name: value
            for name, value in (("context", context), ("weights", weights))
            if value is not None
        }
        return super().fit(rng, data, **extras, **fit_kwargs)

    def _fit_batch(self, data, *, context=None, weights=None):
        """Batches are ``{"data": ..., "context": ..., "weights": ...}`` dicts."""
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
            return _DataContextBatches(data)
        batch = {"data": data}
        if context is not None:
            batch["context"] = context
        if weights is not None:
            batch["weights"] = weights
        return batch

    def _fit_loss(self, rng, batch):
        if isinstance(batch, dict):
            extras = {k: v for k, v in batch.items() if k != "data"}
            return self.loss(rng, batch["data"], **extras)
        return self.loss(rng, batch)

    def _distribution_sampler(self, event_spec, **kwargs) -> _ExportedSampler:
        """Build the family-specific sampling operation."""
        raise NotImplementedError

    def _clear_distribution_cache(self) -> None:
        """Invalidate every compiled distribution operation for this model."""
        clear_export_cache(self)

    def _sample_base(self, rng, sample_shape, spec):
        """Draw this model family's initial sampling state."""
        return sample_normal(rng, sample_shape, spec)

    def _build_exported_sampler(
        self,
        key: tuple,
        event_spec,
        make_sample_fn: Callable[[nnx.GraphDef, bool], Callable],
        *,
        dtype=jax.numpy.float32,
        stochastic: bool = False,
        context_spec=None,
        trace: bool = False,
        base_sample_kwargs=None,
    ) -> _ExportedSampler:
        """Normalize shapes and export a specialized sampler once per key.

        ``make_sample_fn`` receives the model graphdef and whether context is
        present, then returns the
        function ``(state, [rng,] eps, [context]) -> samples`` to export,
        where data and context may both be pytrees.
        """
        spec = normalize_spec(event_spec, dtype)
        context_spec = (
            None if context_spec is None else normalize_spec(context_spec, dtype)
        )
        key = key + (
            spec_cache_key(spec),
            spec_cache_key(context_spec),
            stochastic,
            trace,
            tuple(sorted((base_sample_kwargs or {}).items())),
        )

        def build():
            graphdef, state = nnx.split(self)
            exported = _export_sampler(
                state,
                spec,
                make_sample_fn(graphdef, context_spec is not None),
                stochastic=stochastic,
                context_spec=context_spec,
            )
            return _ExportedSampler(
                self,
                graphdef,
                exported,
                spec,
                stochastic=stochastic,
                context_spec=context_spec,
                trace=trace,
                base_sample_kwargs=base_sample_kwargs,
            )

        return get_cached_export(self, key, build)

    def _build_exported_logpdf(
        self,
        key: tuple,
        event_spec,
        make_logpdf_fn: Callable[[nnx.GraphDef], Callable],
        *,
        dtype=jax.numpy.float32,
        context_spec=None,
    ) -> _ExportedLogPDF:
        """Export ``make_logpdf_fn(graphdef)`` once per ``key`` and cache it."""
        spec = normalize_spec(event_spec, dtype)
        context_spec = (
            None if context_spec is None else normalize_spec(context_spec, dtype)
        )
        key = key + (spec_cache_key(spec), spec_cache_key(context_spec))

        def build():
            graphdef, state = nnx.split(self)
            exported = _export_logpdf(
                state,
                spec,
                make_logpdf_fn(graphdef),
                context_spec=context_spec,
            )
            return _ExportedLogPDF(
                self,
                graphdef,
                exported,
                spec,
                context_spec=context_spec,
            )

        return get_cached_export(self, key, build)

    def as_dist(
        self,
        event_spec=None,
        *,
        context_spec=None,
        context=None,
        **kwargs,
    ):
        """Create a lazy compiled distribution view of this model.

        Args:
            event_spec: Shape or pytree specification for one event. Omit to
                use the model's default. Overrides only this distribution view.
            context_spec: Context specification including optional per-leaf
                dtypes.
            context: Optional context to bind to the returned distribution.
            **kwargs: Static sampling options such as ``num_steps`` or ``mode``.
        """
        if event_spec is None:
            event_spec = self.event_spec
            if event_spec is None:
                raise TypeError(
                    "event_spec is required: pass it to as_dist() or set a "
                    "default at construction / via set_event_spec()."
                )
            if kwargs.get("dtype") is not None:
                event_spec = jax.tree.map(
                    lambda leaf: jax.ShapeDtypeStruct(leaf.shape, kwargs["dtype"]),
                    event_spec,
                )
        event_spec = self._normalize_event_spec(event_spec, kwargs.get("dtype"))
        return _ModelDistribution(
            self,
            event_spec,
            context_spec=context_spec,
            context=context,
            sampler_options=kwargs,
            name=type(self).__name__,
        )
