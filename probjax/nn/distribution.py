"""Adapter that exposes a learned generative model as a Distribution.

Most generative-model classes in :mod:`probjax.nn.diffusion` don't carry an
intrinsic ``event_shape`` (a flow matcher trained on R^d looks the same as
one trained on R^k), and their sampling pipelines have free parameters
(``num_steps``, ``mode="ode"`` vs ``"sde"``). Rather than baking those into
the model classes, we hand the user a small adapter:

>>> ddpm = EDM(net, ...)              # already trained
>>> dist = ddpm.as_distribution(event_shape=(d,), num_steps=50)
>>> samples = dist.rvs(key, shape=(N,))             # plug into SMC / filter / ...
>>> # dist.logpdf(x) raises — diffusion logpdf is opt-in via logpdf_fn=...

Normalizing flows already implement :class:`DistributionAPI` directly
(both ``logpdf`` and ``rvs`` are tractable in closed form via change-of-
variables), so they don't need this adapter — use the flow class directly.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from probjax.stats.base import DistributionAPI
from probjax.utils.typing import Array, ArrayLike, RngKey


class LearnedDistribution(DistributionAPI):
    """Distribution view of a learned generative model.

    A thin adapter that satisfies :class:`DistributionAPI` by delegating
    sampling (and optionally density evaluation) to user-supplied callables.

    Args:
        event_shape: The trailing shape of one sample (e.g. ``(d,)`` for a
            flow on ``R^d``, ``(C, H, W)`` for an image diffusion model).
        sampler_fn: ``(rng, batch_shape) -> samples``. Should produce an
            array of shape ``batch_shape + event_shape``. The model's
            ``sample_ode`` / ``sample_sde`` / ``sample`` method goes here,
            usually wrapped in a small closure that draws the starting
            noise.
        logpdf_fn: Optional ``(x) -> log p(x)``. If absent, :meth:`logpdf`
            raises a :class:`NotImplementedError` with a hint on how to
            wire one (typically ODE-based change of variables for
            diffusion / flow matching).
        batch_shape: Static batch shape of the underlying distribution
            (rarely useful for learned models — defaults to ``()`` since a
            single trained model represents one distribution).
        name: Optional human-readable label, surfaced in ``__repr__``.

    Use the convenience method ``model.as_distribution(event_shape, ...)``
    on supported model classes (``DiffusionDenoiser``, ``MultinomialDiffusion``)
    instead of constructing this directly when the model has a built-in
    sampler.
    """

    def __init__(
        self,
        *,
        event_shape: Tuple[int, ...],
        sampler_fn: Callable[[RngKey, Tuple[int, ...]], Array],
        logpdf_fn: Optional[Callable[[ArrayLike], Array]] = None,
        batch_shape: Tuple[int, ...] = (),
        name: Optional[str] = None,
    ) -> None:
        self._event_shape = tuple(int(d) for d in event_shape)
        self._batch_shape = tuple(int(d) for d in batch_shape)
        self._sampler_fn = sampler_fn
        self._logpdf_fn = logpdf_fn
        self._name = name

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return self._event_shape

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return self._batch_shape

    def rvs(
        self,
        rng: RngKey,
        shape: Tuple[int, ...] = (),
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Array:
        return self._sampler_fn(rng, tuple(shape))

    def logpdf(self, x: ArrayLike) -> Array:
        if self._logpdf_fn is None:
            raise NotImplementedError(
                f"{self!r} doesn't expose a tractable logpdf. For diffusion / "
                "flow-matching models, log-density requires ODE-based "
                "change-of-variables integration (O(num_steps) cost). To enable, "
                "construct LearnedDistribution(..., logpdf_fn=fn) where ``fn(x)`` "
                "performs that integration explicitly."
            )
        return self._logpdf_fn(x)

    def __repr__(self) -> str:
        label = self._name or type(self).__name__
        return f"{label}(event_shape={self._event_shape}, batch_shape={self._batch_shape})"
