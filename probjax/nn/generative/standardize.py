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
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.training import is_stream, take_batches
from probjax.utils.guards import is_mapping, is_tracing
from probjax.utils.jaxutils import where

__all__ = ["StandardizingMixin"]


#: Batches consumed to estimate the standardising transform from a stream.
_STD_BATCHES = 8


def _examples(batch):
    """The generated values of a batch: its ``"data"`` entry, or the batch."""
    return batch["data"] if is_mapping(batch) else batch


def _standardization_sample(data, num_features: int):
    """Examples to estimate shift and scale from a batch or a stream.

    Returns the sample and the source to train on: a stream is advanced to
    look at it, so the batches that were looked at are put back in front.
    """
    if is_stream(data):
        source = iter(data)
        batches = take_batches(source, _STD_BATCHES)
        sample = jnp.concatenate([
            jnp.asarray(_examples(b)).reshape(-1, num_features) for b in batches
        ])
        return sample, itertools.chain(batches, source)
    sample = jnp.asarray(_examples(data), jnp.float32).reshape(-1, num_features)
    return sample, data


class StandardizingMixin:
    """Adds a fixed affine reparameterisation fitted once from the data.

    The model learns the density of ``z = (x - shift) / scale`` and reports
    densities in the original space by carrying the Jacobian:

        ``log p(x) = log p_inner(z) - sum(log scale)``

    which is what keeps ``logpdf`` a proper density rather than a density of
    the standardised variable.
    """

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

    def fit_standardization(self, data):
        """Fit the transform from data. A no-op once already fitted.

        Idempotent on purpose: a second ``fit`` call must not silently move the
        model's coordinate system out from under an already-trained density.

        ``data`` is an array, a batch dict with a ``"data"`` entry, or a
        stream of either (see :func:`probjax.nn.training.is_stream`). A
        stream is summarised from the first :data:`_STD_BATCHES` of them -- the whole
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
        if traced and is_stream(data):
            raise ValueError("Jitted standardization requires an array batch.")
        sample, data = _standardization_sample(data, self._std_shift.shape[0])

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
