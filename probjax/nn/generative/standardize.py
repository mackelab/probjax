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

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = ["StandardizingMixin"]


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
        self._clear_distribution_cache()

        # The pure loss closure captures the non-Param state, so a shift/scale
        # set after it was built would be stale. Dropping it is cheap: it is
        # rebuilt on the next fit.
        from probjax.stats.fit import _PURE_LOSS_CACHE

        _PURE_LOSS_CACHE.pop(self, None)

    def fit_standardization(self, data) -> None:
        """Fit the transform from data. A no-op once already fitted.

        Idempotent on purpose: a second ``fit`` call must not silently move the
        model's coordinate system out from under an already-trained density.
        """
        if not self.standardize or self.is_standardized:
            return
        data = jnp.asarray(data, jnp.float32).reshape(-1, self._std_shift.shape[0])
        shift = jnp.mean(data, axis=0)
        scale = jnp.std(data, axis=0)
        # A constant column has zero spread and would divide by zero; leaving it
        # at unit scale keeps the map invertible and the column simply unscaled.
        scale = jnp.where(scale > 1e-6, scale, 1.0)
        self.set_standardization(shift, scale)

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
