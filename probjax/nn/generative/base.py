"""Base class for generative-model families.

A family implements plain methods, each batched over one leading axis:

* ``loss(rng, data, **inputs)`` -- the training objective;
* ``sample_base(rng, sample_shape, spec, **options)`` -- the starting noise;
* ``sample_from_noise(noise, rng=None, inputs=None, **options)`` -- noise to
  samples (the solver loop lives here);
* ``logpdf(x, inputs=None)`` -- only where the density is tractable.

They are ordinary JAX functions: jit, vmap and differentiate them freely.
:meth:`GenerativeModel.as_dist` serves them through :meth:`_compiled`, one
cached ``jax.jit`` per method and options. Event descriptions are pytree
specs, so structured data works like a single flat array.
"""

from __future__ import annotations

import weakref
from numbers import Integral
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.distribution import ModelDistribution
from probjax.nn.generative.sampling import sample_normal
from probjax.nn.training import FitResult, fit_module, is_stream
from probjax.utils.guards import is_mapping
from probjax.utils.typing import Array, PyTree, RngKey

__all__ = ["GenerativeModel"]

#: model -> {(method, options): (graphdef, compiled)}. Weak, so a cache entry
#: never keeps its model alive; entries only close over the graphdef.
_COMPILED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _affine_sample_base(self, rng, sample_shape, spec, **options):
    """Gaussian base ``N(mu0, std0**2)``; flow models assign this as ``sample_base``."""
    del options
    return sample_normal(
        rng,
        sample_shape,
        spec,
        loc=self.mu0.get_value(),
        scale=self.std0.get_value(),
    )


def _as_batch(item):
    """A batch is a dict whose ``"data"`` entry is what the model generates.

    Every other entry is a named model input (``context``, ``local_context``,
    ``weights``, ...); a bare array or pytree is shorthand for ``{"data": item}``.
    """
    if is_mapping(item):
        if "data" not in item:
            raise ValueError(
                f"a batch dict must have a 'data' key; got keys {sorted(item)}."
            )
        return dict(item)
    return {"data": item}


def _batch_loss(model, rng, batch):
    """``model.loss(rng, data, **inputs)`` for one batch dict."""
    inputs = dict(batch)
    return model.loss(rng, inputs.pop("data"), **inputs)


class GenerativeModel(nnx.Module):
    """Shared base for generative families: training and compiled distribution views."""

    _event_dtype = jax.numpy.float32
    #: Set by the continuous-time models, whose event dtypes must be floating
    #: point; ``None`` accepts any dtype.
    _floating_event_dtype_error: Optional[str] = None

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
        if self._floating_event_dtype_error is not None and any(
            not jnp.issubdtype(leaf.dtype, jnp.floating)
            for leaf in jax.tree.leaves(spec)
        ):
            raise TypeError(self._floating_event_dtype_error)
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

        Existing distribution views keep their bound event specs.
        """
        spec = self._normalize_event_spec(event_spec)
        leaves, tree = jax.tree.flatten(spec)
        self._default_event_spec = nnx.static((tree, tuple(leaves)))

    def loss(self, rng: RngKey, data: PyTree[Array], *args, **kwargs) -> Array:
        """Monte-Carlo training loss for a batch of data."""
        raise NotImplementedError

    def fit(self, rng: RngKey, data, *, weights=None, **options) -> FitResult:
        """Train in place on ``data``.

        ``data`` is an array pytree, a dict with a ``"data"`` entry plus any
        named model inputs (``{"data": x, "context": c}``), or a stream of
        either. The inputs reach :meth:`loss`, and so the network, as keyword
        arguments. ``weights`` weight the examples in the loss. The remaining
        keywords are the options of :func:`probjax.nn.training.fit`, on top of
        this model's :meth:`fit_defaults`.
        """
        if is_stream(data):
            if weights is not None:
                raise ValueError(
                    "weights cannot be passed alongside a stream; give each "
                    "batch dict a 'weights' entry instead."
                )
            data = (_as_batch(item) for item in data)
        else:
            data = _as_batch(data)
            if weights is not None:
                data = {**data, "weights": weights}
        data = self._before_fit(data)
        options = {**self.fit_defaults(), **options}
        return fit_module(self, rng, data, loss=_batch_loss, **options)

    def fit_defaults(self) -> dict:
        """This family's training recipe; ``fit`` keywords override it."""
        return {}

    def _before_fit(self, data):
        """Set up state from the training data (e.g. standardisation)."""
        return data

    def sample_base(self, rng: RngKey, sample_shape, spec, **options):
        """Draw this family's starting noise; ``options`` are the view's."""
        del options
        return sample_normal(rng, sample_shape, spec)

    def sample_from_noise(self, noise, rng=None, inputs=None, **options):
        """Map a batch of noise to a batch of samples."""
        raise NotImplementedError

    def _compiled(self, method: str, **options):
        """``(fn, state)``: jitted ``fn(state, *args)`` runs ``self.<method>(*args, **options)``.

        Cached per ``(method, options)``, so views with the same options share
        compilations. A model whose structure changed -- a new solver config,
        a replaced submodule -- has a new graphdef and is compiled again; new
        weights are just a new ``state``.
        """
        graphdef, state = nnx.split(self)
        key = (method, tuple(sorted(options.items())))
        cache = _COMPILED.setdefault(self, {})
        entry = cache.get(key)
        if entry is None or entry[0] != graphdef:

            def call(state, *args):
                model = nnx.merge(graphdef, state)
                return getattr(model, method)(*args, **options)

            entry = (graphdef, jax.jit(call))
            cache[key] = entry
        return entry[1], state

    def as_dist(self, event_spec=None, **options) -> ModelDistribution:
        """View this model as a distribution over one event.

        Args:
            event_spec: Shape or pytree specification for one event. Omit to
                use the model's default. Overrides only this view.
            **options: Static sampling options such as ``num_steps`` or
                ``mode``; ``dtype`` sets the event dtype.

        Bind named inputs shared by every sample with ``.condition(...)``.
        """
        dtype = options.pop("dtype", None)
        if event_spec is None:
            event_spec = self.event_spec
            if event_spec is None:
                raise TypeError(
                    "event_spec is required: pass it to as_dist() or set a "
                    "default at construction / via set_event_spec()."
                )
            if dtype is not None:
                event_spec = jax.tree.map(
                    lambda leaf: jax.ShapeDtypeStruct(leaf.shape, dtype), event_spec
                )
        spec = self._normalize_event_spec(event_spec, dtype)
        return ModelDistribution(self, spec, options=options)
