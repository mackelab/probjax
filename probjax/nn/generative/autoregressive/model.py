"""Autoregressive density models.

``p(x) = prod_i p(x_i | x_<i)``: a causal conditioner predicts raw parameters
for every dimension, and a :class:`~probjax.nn.generative.autoregressive.heads.Head`
turns them into a univariate distribution:

>>> model = Autoregressive(4, MixtureHead(10), rngs=nnx.Rngs(0))

Unlike a normalizing flow there is no bijection: the density is read straight
off the head, so ``logpdf`` costs one conditioner pass whatever the head.
Sampling is the sequential part -- one pass per dimension.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.generative.autoregressive.conditioners import (
    ARConditionerConfig,
    MLPARConditionerConfig,
)
from probjax.nn.generative.autoregressive.heads import Head, NormalHead
from probjax.nn.generative.base import GenerativeModel
from probjax.nn.generative.standardize import StandardizingMixin
from probjax.nn.inputs import context_only
from probjax.nn.training import is_stream
from probjax.utils.guards import is_tracing
from probjax.utils.jaxutils import leading_dim, where

__all__ = ["Autoregressive"]


class Autoregressive(StandardizingMixin, GenerativeModel):
    """Autoregressive density over ``input_dim`` variables.

    Args:
        input_dim: Number of variables in the joint.
        head: The conditional family (default :class:`NormalHead`, i.e. MADE).
        rngs: Parameter initialisation.
        conditioner: Config of the causal network (default: masked MLP).
        context_features: Width of an optional conditioning vector.
        standardize: Fit a per-dimension affine map to the training data
            (continuous heads only).
    """

    def __init__(
        self,
        input_dim: int,
        head: Optional[Head] = None,
        rngs: Optional[nnx.Rngs] = None,
        *,
        conditioner: Optional[ARConditionerConfig] = None,
        context_features: Optional[int] = None,
        standardize: bool = True,
        name: Optional[str] = None,
    ) -> None:
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive; got {input_dim}.")
        if rngs is None:
            raise ValueError("rngs is required.")
        head = NormalHead() if head is None else head
        if not isinstance(head, Head):
            raise TypeError(f"head must be a Head; got {type(head).__name__}.")
        conditioner = conditioner or MLPARConditionerConfig()
        if not isinstance(conditioner, ARConditionerConfig):
            raise TypeError("conditioner must implement ARConditionerConfig")

        self.input_dim = input_dim
        self.head = head
        self.context_features = context_features
        self.name = name
        self.conditioner = conditioner.build(
            input_dim,
            self._width,
            head.num_params,
            context_features=context_features,
            rngs=rngs,
        )
        # Integer labels have no meaningful mean/std to standardise with.
        self._init_standardization(input_dim, standardize and not self._discrete)
        super().__init__()

    # -- tokens --------------------------------------------------------------

    @property
    def _discrete(self) -> bool:
        return self.head.num_classes is not None

    @property
    def _width(self) -> int:
        """Token width: 1 per value, or a one-hot vector per class label."""
        return self.head.num_classes if self._discrete else 1

    @property
    def event_dtype(self):
        return jnp.int32 if self._discrete else jnp.float32

    def _tokens(self, x):
        """``(..., input_dim)`` values -> ``(..., input_dim, width)`` tokens."""
        x = jnp.asarray(x)
        if self._discrete:
            return jax.nn.one_hot(x.astype(jnp.int32), self.head.num_classes)
        return x[..., None].astype(jnp.float32)

    # -- density -------------------------------------------------------------

    def predict_params(self, x, context=None, *, rng=None):
        """Raw head parameters for every dimension, ``(..., input_dim, num_params)``."""
        return self.conditioner(self._tokens(x), context, rng=rng)

    def conditional_logpdfs(self, x, context=None):
        """``log p(x_i | x_<i)`` for every ``i``, shape ``(..., input_dim)``."""
        return self.head.logpdf(self.predict_params(x, context), jnp.asarray(x))

    def _logpdf(self, value, context=None):
        # Density of the original variable: standardise, then carry the
        # Jacobian of that map so this stays a density of x, not of z.
        z = self._standardize(value) if self.standardize else value
        inner = jnp.sum(self.conditional_logpdfs(z, context), axis=-1)
        return inner - self._log_scale_correction()

    def logpdf(self, x, inputs=None):
        """Joint log density of a batch ``x`` in data space."""
        context = context_only(inputs, "Autoregressive")
        if context is None:
            return jax.vmap(self._logpdf)(x)
        return jax.vmap(lambda item, c: self._logpdf(item, context=c))(x, context)

    def _check_support(self, data) -> None:
        """Reject data a bounded head cannot represent, instead of scoring -inf."""
        if self.head.support is None:
            return
        low, high = self.head.support
        # The head sees standardised values, so that is what must be in range.
        data = self._standardize(data) if self.standardize else data
        lo, hi = float(jnp.min(data)), float(jnp.max(data))
        if lo < low or hi > high:
            raise ValueError(
                f"{type(self.head).__name__} has bounded support [{low}, {high}] "
                f"but the data spans [{lo:.4g}, {hi:.4g}], so those points have "
                "zero density. Widen the bounds, or use a head with tails "
                "(e.g. HistogramHead(tails=True))."
            )

    def loss(self, rng, data, *args, context=None, **kwargs):
        """Negative mean joint log-likelihood."""
        del rng, args, kwargs
        if context is None:
            return -jnp.mean(self._logpdf(data))
        pair = jax.vmap(lambda x, c: self._logpdf(x, context=c))
        return -jnp.mean(pair(data, context))

    # -- sampling ------------------------------------------------------------

    def sample(
        self,
        rng,
        sample_shape=(),
        *,
        context=None,
        prefix=None,
        prefix_len=0,
        use_cache=True,
    ):
        """Ancestral sampling: one conditioner pass per dimension.

        ``prefix`` conditions the draw on known leading values: with
        ``prefix`` of shape ``(..., input_dim)`` and ``prefix_len=k`` the
        first ``k`` positions are clamped to ``prefix`` and only the rest
        are sampled -- image completion from a top half, for example.

        A conditioner with a KV cache (the transformer) decodes one position
        at a time when ``use_cache`` is set; otherwise every step re-reads the
        whole prefix. Both paths draw from the same per-dimension keys, so
        they agree up to floating-point dust.
        """
        sample_shape = tuple(sample_shape)
        if not 0 <= prefix_len <= self.input_dim:
            raise ValueError(
                f"prefix_len must lie in [0, {self.input_dim}]; got {prefix_len}."
            )
        if prefix is None:
            if prefix_len:
                raise ValueError("prefix_len requires prefix.")
            prefix_z = None
        else:
            prefix_z = (
                self._standardize(prefix) if self.standardize else jnp.asarray(prefix)
            )
            prefix_z = jnp.broadcast_to(prefix_z, sample_shape + (self.input_dim,))
        keys = jax.random.split(rng, self.input_dim)
        if use_cache and hasattr(self.conditioner, "predict_next"):
            x = self._sample_cached(keys, sample_shape, context, prefix_z, prefix_len)
        else:
            x = self._sample_naive(keys, sample_shape, context, prefix_z, prefix_len)
        return self._unstandardize(x) if self.standardize else x

    def _sample_naive(self, keys, sample_shape, context, prefix_z, prefix_len):
        x = jnp.zeros(sample_shape + (self.input_dim,), dtype=self.event_dtype)

        def step(x, inputs):
            i, key = inputs
            raw_i = jnp.take(self.predict_params(x, context), i, axis=-2)
            value = self.head.sample(key, raw_i).astype(x.dtype)
            if prefix_z is not None:
                value = where(i < prefix_len, prefix_z[..., i], value)
            # Only dimension i is written; the conditioner guarantees its
            # parameters depended on x_<i alone.
            return x.at[..., i].set(value), None

        x, _ = jax.lax.scan(step, x, (jnp.arange(self.input_dim), keys))
        return x

    def _sample_cached(self, keys, sample_shape, context, prefix_z, prefix_len):
        """KV-cached ancestral sampling, compiled to one :func:`flax.nnx.scan`.

        The attention caches are threaded through as scan carry (the model is
        passed explicitly so its ``Cache`` state is lifted) while the
        parameters stay put; every call starts from a fresh ``init_decode``.
        """
        head, dtype = self.head, self.event_dtype
        self.conditioner.init_decode(sample_shape, dtype=jnp.float32)  # type: ignore
        carry0 = (
            jnp.zeros(sample_shape + (self.input_dim,), dtype=dtype),
            jnp.zeros(sample_shape + (self._width,), jnp.float32),
        )

        @nnx.scan(
            in_axes=(
                nnx.Carry,
                0,
                # Parameters are shared, but each step must see the caches
                # written by the preceding step.
                nnx.StateAxes({nnx.Cache: nnx.Carry, ...: None}),
            ),
            out_axes=(nnx.Carry, 0),
            length=self.input_dim,
        )
        def step(carry, i, model):
            x, prev_token = carry
            raw_i = model.conditioner.predict_next(prev_token, i, context)
            value = head.sample(keys[i], raw_i).astype(dtype)
            if prefix_z is not None:
                value = where(i < prefix_len, prefix_z[..., i], value)
            token = model._tokens(value[..., None])[..., 0, :]
            return (x.at[..., i].set(value), token), None

        (x, _), _ = step(carry0, jnp.arange(self.input_dim), self)
        return x

    # -- generative-model contract --------------------------------------------

    def sample_base(self, rng, sample_shape, spec, **options):
        """Placeholder noise: only its batch size is used, the draws come from ``rng``."""
        del rng, spec, options
        return jnp.zeros(tuple(sample_shape) + (self.input_dim,))

    def sample_from_noise(self, noise, rng=None, inputs=None):
        """Ancestral samples, one per row of ``noise``; ``rng`` supplies the draws."""
        if rng is None:
            raise ValueError("rng is required: autoregressive sampling is stochastic.")
        context = context_only(inputs, "Autoregressive")
        return self.sample(rng, (leading_dim(noise),), context=context)

    def as_dist(self, event_spec=None, **options):
        if event_spec is None:
            event_spec = jax.ShapeDtypeStruct((self.input_dim,), self.event_dtype)
        return super().as_dist(event_spec, **options)

    def _before_fit(self, data):
        """Fit the standardising transform, and check the data fits the head."""
        data = self.fit_standardization(data)
        if not is_stream(data) and not is_tracing(jnp.asarray(data["data"])):
            self._check_support(jnp.asarray(data["data"]))
        # `loss` goes through `_logpdf`, which standardises internally.
        return super()._before_fit(data)

    def fit_defaults(self) -> dict:
        """Autoregressive models train better on a warm-started, decaying rate."""
        return {"schedule": "warmup_cosine"}
