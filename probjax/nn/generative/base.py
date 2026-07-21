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
from probjax.stats.fit import FitMixin
from probjax.utils.typing import Array, PyTree, RngKey

__all__ = ["GenerativeModel"]


class GenerativeModel(nnx.Module, FitMixin):
    """Shared base for generative families with cached exported samplers."""

    def loss(self, rng: RngKey, data: PyTree[Array], *args, **kwargs) -> Array:
        """Monte-Carlo training loss for a batch of data."""
        raise NotImplementedError

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
        event_spec,
        *,
        context_spec=None,
        context=None,
        **kwargs,
    ):
        """Create a lazy compiled distribution view of this model.

        Args:
            event_spec: Shape or pytree specification for one event.
            context_spec: Context specification including optional per-leaf
                dtypes.
            context: Optional context to bind to the returned distribution.
            **kwargs: Static sampling options such as ``num_steps`` or ``mode``.
        """
        return _ModelDistribution(
            self,
            event_spec,
            context_spec=context_spec,
            context=context,
            sampler_options=kwargs,
            name=type(self).__name__,
        )
