"""
Transform protocols (:mod:`probjax.stats.bijective.protocols`)
===============================================================

The transform contract used across probjax, plus an object-layer
``TransformedDistribution`` for composing transforms with any
:class:`~probjax.stats.base.DistributionAPI` base, including learned-model
views returned by ``model.as_dist()``.

A transform is any forward callable ``y = T(x)``. If it additionally
provides ``inverse`` / ``inverse_and_logdet`` methods (as the
autoregressive flow modules do), those are used directly; otherwise the
inverse is derived automatically from the forward jaxpr via
:func:`probjax.core.inverse_and_logabsdet`.

Note that the bijections in this package are *not* the fast path here: a
:class:`~probjax.core.custom_inverse` object exposes ``inv`` /
``inv_and_logdet`` / ``value_and_logdet``, not ``inverse`` /
``inverse_and_logdet``, so :func:`ensure_invertible` always wraps them in
``_AutoInvertedTransform``. That still resolves to the registered inverse —
tracing the forward hits ``custom_inverse_call_p``, whose rule pulls the
registered thunk — it just goes via the jaxpr rather than a direct method call.
"""

from typing import Optional, Protocol, Tuple, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array

from probjax.core import inverse_and_logabsdet
from probjax.core.custom_primitives.sharded_primitive import batch_shard
from probjax.stats.base import DistributionAPI
from probjax.utils.typing import ArrayLike, RngKey

__all__ = [
    "TransformProtocol",
    "InvertibleTransformProtocol",
    "ensure_invertible",
    "forward_and_logdet",
    "TransformedDistribution",
]


@runtime_checkable
class TransformProtocol(Protocol):
    """A forward transform ``y = T(x)``."""

    def __call__(self, x: ArrayLike, *args, **kwargs) -> Array: ...


@runtime_checkable
class InvertibleTransformProtocol(TransformProtocol, Protocol):
    """A transform that knows its own inverse and log-determinant."""

    def inverse(self, y: ArrayLike, *args, **kwargs) -> Array: ...

    def inverse_and_logdet(
        self, y: ArrayLike, *args, **kwargs
    ) -> Tuple[Array, Array]: ...


class _AutoInvertedTransform:
    """Wrap a plain forward callable with a jaxpr-derived inverse.

    The inverse is derived per call rather than stored: the derived jaxpr
    bakes module weights in as constants, so storing it would serve stale
    weights after the wrapped module is trained in place.
    """

    def __init__(self, forward):
        self._forward = forward

    def __call__(self, x, *args, **kwargs):
        return self._forward(x, *args, **kwargs)

    def inverse(self, y, *args, **kwargs):
        return self.inverse_and_logdet(y, *args, **kwargs)[0]

    def inverse_and_logdet(self, y, *args, **kwargs):
        return inverse_and_logabsdet(self._forward)(y, *args, **kwargs)


def ensure_invertible(transform) -> InvertibleTransformProtocol:
    """Return ``transform`` if it satisfies :class:`InvertibleTransformProtocol`,
    otherwise wrap the forward callable with an auto-derived inverse."""
    if isinstance(transform, InvertibleTransformProtocol):
        return transform
    return _AutoInvertedTransform(transform)


def forward_and_logdet(transform, x: ArrayLike) -> Tuple[Array, Array]:
    """Compute ``y = T(x)`` and ``log |det dT/dx|`` at ``x``.

    Uses ``transform.forward_and_logdet`` if the transform provides it;
    otherwise evaluates the (possibly auto-derived) inverse log-determinant
    at ``y`` and negates it.
    """
    explicit = getattr(transform, "forward_and_logdet", None)
    if explicit is not None:
        return explicit(x)
    y = transform(x)
    _, logdet_inv = ensure_invertible(transform).inverse_and_logdet(y)
    return y, -logdet_inv


class TransformedDistribution(DistributionAPI):
    """Push a base :class:`DistributionAPI` through a transform.

    Sampling applies the forward transform to base samples; ``logpdf`` uses
    the change-of-variables formula with the transform's inverse (explicit
    or auto-derived).

    Args:
        base: Base distribution (anything satisfying ``DistributionAPI`` —
            scipy-style frozen distributions, learned-model distribution
            views, or another ``TransformedDistribution``).
        transform: Forward callable, optionally satisfying
            :class:`InvertibleTransformProtocol`.
        event_shape: Override when the transform changes the event shape;
            defaults to the base distribution's event shape.
    """

    def __init__(
        self,
        base: DistributionAPI,
        transform,
        *,
        event_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.base = base
        self.transform = transform
        self._invertible = ensure_invertible(transform)
        self._event_shape = (
            tuple(event_shape) if event_shape is not None else tuple(base.event_shape)
        )

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return tuple(self.base.batch_shape)

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return self._event_shape

    def _flatten(self, x):
        """Flatten leading dims to a single vmap axis, keeping event dims."""
        x = jnp.asarray(x)
        event_ndim = len(self._event_shape)
        if event_ndim:
            if tuple(x.shape[-event_ndim:]) != self._event_shape:
                raise ValueError(
                    "Trailing dimensions of the input must match the event shape."
                )
            leading_shape = tuple(x.shape[:-event_ndim])
        else:
            leading_shape = tuple(x.shape)
        return jnp.reshape(x, (-1,) + self._event_shape), leading_shape

    def rvs(
        self,
        rng: RngKey,
        shape: Tuple[int, ...] = (),
        name: Optional[str] = None,
        **kwargs,
    ) -> Array:
        samples = self.base.rvs(rng, shape=shape, **kwargs)
        base_event_ndim = len(tuple(self.base.event_shape))
        flat = jnp.reshape(
            samples, (-1,) + tuple(samples.shape[samples.ndim - base_event_ndim :])
        )
        leading_shape = tuple(samples.shape[: samples.ndim - base_event_ndim])
        transformed_flat = batch_shard(jax.vmap(self.transform))(flat)
        return jnp.reshape(transformed_flat, leading_shape + self._event_shape)

    def logpdf(self, x: ArrayLike) -> Array:
        x_flat, leading_shape = self._flatten(x)
        inv_flat, logdet_flat = batch_shard(
            jax.vmap(self._invertible.inverse_and_logdet)
        )(x_flat)
        inv = jnp.reshape(inv_flat, leading_shape + tuple(self.base.event_shape))
        logdet = jnp.reshape(logdet_flat, leading_shape)
        return self.base.logpdf(inv) + logdet
