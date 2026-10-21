"""Data standardisation for density models.

Every flexible bijector in this stack carries a domain assumption -- the spline
knots span ``[-5, 5]``, the histogram grid has explicit bounds -- and data that
sits far outside it lands in the linear tails, where the model has almost no
capacity. Standardising removes that assumption without the caller having to
know it exists.

The shift and scale are ordinary :class:`nnx.Variable` state, not hidden
training history: they serialise with the model, travel through
``nnx.split`` / ``nnx.merge``, and are visible via
:meth:`StandardizingMixin.standardization`.
"""

import itertools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.training import DataSpec, take_batches
from probjax.utils.guards import is_mapping, is_tracing
from probjax.utils.jaxutils import where

__all__ = ["StandardizingMixin"]


#: Batches consumed to estimate the standardising transform from a stream.
_STD_BATCHES = 8


def _standardization_sample(data, num_features: int, spec: DataSpec):
    """Examples to estimate shift and scale from an array, a dict or a stream.

    Returns the sample and the source to train on: a stream is advanced to
    look at it, so the batches that were looked at are put back in front.
    """
    primary = spec.primary or "data"
    if spec.stream:
        source = iter(data)
        batches = take_batches(source, _STD_BATCHES)
        rest = itertools.chain(batches, source)
        parts = [b[primary] if is_mapping(b) else b for b in batches]
        sample = jnp.concatenate([
            jnp.asarray(p).reshape(-1, num_features) for p in parts
        ])
        return sample, rest
    examples = data[primary] if is_mapping(data) else data
    return jnp.asarray(examples, jnp.float32).reshape(-1, num_features), data


class StandardizingMixin:
    """Adds a fixed affine reparameterisation fitted once from the data.

    The model learns the density of ``z = (x - shift) / scale`` and reports
    densities in the original space by carrying the Jacobian:

        ``log p(x) = log p_inner(z) - sum(log scale)``

    which is what keeps ``logpdf`` a proper density rather than a density of
    the standardised variable.
    """

    # Provided by the generative model base class.
    _clear_distribution_cache: Callable[[], None]

    def _init_standardization(self, input_dim: int, standardize: bool = True) -> None:
        self.standardize = bool(standardize)
        self._std_shift = nnx.Variable(jnp.zeros((input_dim,)))
        self._std_scale = nnx.Variable(jnp.ones((input_dim,)))
        self._std_fitted = nnx.Variable(jnp.zeros((), dtype=bool))

    # -- state ---------------------------------------------------------------

    @property
    def standardization(self) -> Tuple[jax.Array, jax.Array]:
        """The ``(shift, scale)`` currently applied."""
        return self._std_shift.get_value(), self._std_scale.get_value()

    @property
    def is_standardized(self) -> bool:
        return bool(self._std_fitted.get_value())

    def set_standardization(self, shift, scale) -> None:
        """Set the transform explicitly, marking it as fitted."""
        shift = jnp.broadcast_to(jnp.asarray(shift, jnp.float32), self._std_shift.shape)
        scale = jnp.broadcast_to(jnp.asarray(scale, jnp.float32), self._std_scale.shape)
        if jnp.any(scale <= 0):
            raise ValueError("standardization scale must be strictly positive.")
        self._std_shift[...] = shift
        self._std_scale[...] = scale
        self._std_fitted[...] = jnp.asarray(True)
        self._clear_distribution_cache()

    def fit_standardization(self, data, spec: DataSpec = DataSpec()):
        """Fit the transform from data. A no-op once already fitted.

        Idempotent on purpose: a second ``fit`` call must not silently move the
        model's coordinate system out from under an already-trained density.

        ``spec`` says how the data is laid out: a stream of batches is
        summarised from the first :data:`_STD_BATCHES` of them -- the whole
        point of streaming is that the full dataset does not fit, and an
        estimate from a few thousand examples is accurate enough for what this
        transform is for.

        Returns the data to train on, which for a stream is the same batches
        again: looking at them must not take them away from the run.
        """
        if not self.standardize:
            return data
        fitted = self._std_fitted.get_value()
        traced = is_tracing(fitted)
        if not traced and bool(fitted):
            return data
        if traced and spec.stream:
            raise ValueError("Jitted standardization requires an array batch.")
        sample, data = _standardization_sample(data, self._std_shift.shape[0], spec)

        def estimate():
            shift = jnp.mean(sample, axis=0)
            scale = jnp.std(sample, axis=0)
            # Constant columns keep unit scale so the map remains invertible.
            scale = where(scale > 1e-6, scale, 1.0)
            return (
                shift.astype(self._std_shift.dtype),
                scale.astype(self._std_scale.dtype),
            )

        shift, scale = jax.lax.cond(fitted, lambda: self.standardization, estimate)
        self._std_shift[...] = shift
        self._std_scale[...] = scale
        self._std_fitted[...] = jnp.asarray(True)
        if not traced:
            self._clear_distribution_cache()
        return data

    # -- the transform --------------------------------------------------------

    def _standardize(self, x):
        shift, scale = self.standardization
        return (jnp.asarray(x) - shift) / scale

    def _unstandardize(self, z):
        shift, scale = self.standardization
        return jnp.asarray(z) * scale + shift

    def _log_scale_correction(self):
        """``sum(log scale)`` -- the Jacobian of the standardising map."""
        return jnp.sum(jnp.log(self._std_scale.get_value()))
