"""Autoregressive density models.

``p(x) = prod_i p(x_i | x_<i)``, where each conditional is a univariate family
from :mod:`probjax.stats` whose parameters a masked network predicts.

Unlike a normalizing flow there is no bijection: the density is read straight
off the family, so evaluating ``logpdf`` costs a single masked forward pass no
matter which family is used. Sampling is the sequential part -- one pass per
dimension, drawing ``x_i`` from the family before moving on.
"""

import weakref
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.generative.autoregressive.config import (
    ARConditionerConfig,
    ARFamily,
    MLPARConditionerConfig,
)
from probjax.nn.generative.base import GenerativeModel
from probjax.nn.generative.sampling import _ExportedSampler
from probjax.nn.generative.standardize import StandardizingMixin

__all__ = [
    "Autoregressive",
    "MADE",
    "MixtureAutoregressive",
    "SplineAutoregressive",
    "HistogramAutoregressive",
    "CategoricalAutoregressive",
    "made",
]


class _EagerARSampler:
    """Sampler for conditioners that cannot go through ``jax.export``.

    Mirrors the escape hatch the discrete diffusion model uses: a weak
    reference to the model plus a direct call, accepted by ``as_dist`` without
    a compiled export.
    """

    def __init__(self, model, event_shape):
        self._model_ref = weakref.ref(model)
        self.event_shape = event_shape

    @property
    def invalidated(self) -> bool:
        return False

    def __call__(self, rng, shape=(), *, context=None):
        model = self._model_ref()
        if model is None:
            raise RuntimeError("The model backing this distribution no longer exists.")
        return model.sample(rng, tuple(shape), context=context)

    def from_noise(self, *args, **kwargs):
        raise NotImplementedError(
            "This autoregressive conditioner does not support sampling from "
            "supplied noise; its sampler is not exported."
        )


class _KeyNormalizingSampler:
    """Accept both typed and legacy PRNG keys.

    The exported sampler takes the rng as a real argument (sampling is
    stochastic, unlike a flow's), and an export is traced for one key
    representation. Normalising here keeps ``rvs`` callable with either
    ``jax.random.key`` or ``jax.random.PRNGKey``, as every other family is.
    """

    def __init__(self, inner):
        self._inner = inner

    @property
    def invalidated(self):
        return self._inner.invalidated

    @staticmethod
    def _as_typed(rng):
        if jnp.issubdtype(jnp.asarray(rng).dtype, jax.dtypes.prng_key):
            return rng
        return jax.random.wrap_key_data(jnp.asarray(rng, dtype=jnp.uint32))

    def __call__(self, rng, shape=(), *, context=None):
        return self._inner(self._as_typed(rng), shape, context=context)

    def from_noise(self, eps, *, rng=None, context=None):
        if rng is not None:
            rng = self._as_typed(rng)
        return self._inner.from_noise(eps, rng=rng, context=context)


class Autoregressive(StandardizingMixin, GenerativeModel):
    """Autoregressive density over ``input_dim`` variables.

    Parameters
    ----------
    input_dim :
        Number of variables in the joint.
    family :
        The univariate conditional, as an :class:`ARFamily`. A bare
        ``probjax.stats`` generator is accepted and wrapped.
    conditioner :
        Config for the masked network. Defaults to a MADE-style masked MLP.
    context_features :
        Width of an optional conditioning vector.
    """

    def __init__(
        self,
        input_dim: int,
        family: Any = None,
        rngs: nnx.Rngs = None,
        *,
        conditioner: Optional[ARConditionerConfig] = None,
        context_features: Optional[int] = None,
        name: Optional[str] = None,
        standardize: bool = True,
    ) -> None:
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive; got {input_dim}.")
        if rngs is None:
            raise ValueError("rngs is required.")

        if family is None:
            family = ARFamily.normal()
        elif not isinstance(family, ARFamily):
            # A bare stats generator: wrap it with default packing.
            family = ARFamily(family)
        conditioner = conditioner or MLPARConditionerConfig()
        if not isinstance(conditioner, ARConditionerConfig):
            raise TypeError("conditioner must implement ARConditionerConfig")

        self.input_dim = input_dim
        self.family = family
        self.conditioner_config = conditioner
        self.context_features = context_features
        self.name = name

        params_dim = family.params_dim()
        # A discrete head one-hots its input, widening the network's first layer
        # without changing the number of autoregressive variables.
        in_features = input_dim
        if family.discrete:
            in_features = input_dim * family._natural_sizes[family._predicted[0]]
        self.conditioner = conditioner.build(
            input_dim,
            params_dim,
            in_features=in_features,
            context_features=context_features,
            rngs=rngs,
        )
        # A discrete family has no meaningful mean/std, and shifting integer
        # labels would destroy them.
        self._init_standardization(input_dim, standardize and not family.discrete)
        super().__init__()

    # -- parameters and density ---------------------------------------------

    def predict_params(self, x, context=None, *, rng=None):
        """Per-dimension parameter blocks, shape ``(..., input_dim, params_dim)``."""
        encoded = self.family.encode(x)
        if self.family.discrete:
            encoded = encoded.reshape(encoded.shape[:-2] + (-1,))
        flat = self.conditioner(encoded, context, rng=rng)
        return flat.reshape(flat.shape[:-1] + (self.input_dim, self.family.params_dim()))

    def conditional_logpdfs(self, x, context=None):
        """``log p(x_i | x_<i)`` for every ``i``, shape ``(..., input_dim)``."""
        params = self.predict_params(x, context)
        natural = self.family.unpack(params)
        return self.family.logpdf(jnp.asarray(x), natural)

    def _logpdf(self, value, context=None):
        # Density of the original variable: standardise, then carry the
        # Jacobian of that map so this stays a density of x, not of z.
        z = self._standardize(value) if self.standardize else value
        inner = jnp.sum(self.conditional_logpdfs(z, context), axis=-1)
        return inner - self._log_scale_correction()

    def __call__(self, x, context=None, *, rng=None):
        return self._logpdf(x, context)

    def _check_support(self, data) -> None:
        """Reject data the family cannot represent, instead of returning -inf."""
        bounds = self.family.bounded_support()
        if bounds is None:
            return
        low, high = bounds
        # The family sees standardised values, so that is what must be in range.
        data = self._standardize(data) if self.standardize else data
        lo, hi = float(jnp.min(data)), float(jnp.max(data))
        if lo < low or hi > high:
            raise ValueError(
                f"{self.family.dist.name!r} has bounded support [{low}, {high}] "
                f"but the data spans [{lo:.4g}, {hi:.4g}], so those points have "
                "zero density. Widen the bounds, or use a family with tails "
                "(e.g. ARFamily.histogram(..., tails=True))."
            )

    def loss(self, rng, data, *args, context=None, **kwargs):
        """Negative mean joint log-likelihood."""
        del rng, args, kwargs
        if not isinstance(jnp.asarray(data), jax.core.Tracer):
            self._check_support(jnp.asarray(data))
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

        The naive loop scans over the *event* dimension, which is static,
        so it survives export with a symbolic batch size.

        ``prefix`` conditions the draw on known leading values: with
        ``prefix`` of shape ``(..., input_dim)`` and ``prefix_len=k`` the
        first ``k`` positions are clamped to ``prefix`` and only the rest
        are sampled -- image completion from a top half, for example.
        ``prefix_len`` may be any value in ``[0, input_dim]``.

        With a transformer conditioner, ``use_cache=True`` (the default)
        decodes with the attention KV cache instead of re-reading the whole
        prefix at every step; ``False`` selects the naive loop, which is
        also what non-transformer conditioners always use. Both paths draw
        from the same per-dimension keys, so they agree up to the
        floating-point dust between the two compiled scans.

        The cached loop is compiled to a single program and does
        asymptotically less attention work (one growing prefix row per step
        instead of a full matrix). Time both paths with
        ``block_until_ready`` -- JAX dispatches asynchronously, so bare
        ``time.time`` differences only measure enqueueing.
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
        if use_cache and hasattr(self.conditioner, "predict_next_params"):
            return self._sample_cached(
                keys, sample_shape, context, prefix_z, prefix_len
            )
        return self._sample_naive(keys, sample_shape, context, prefix_z, prefix_len)

    def _sample_naive(self, keys, sample_shape, context, prefix_z, prefix_len):
        dtype = self.family.event_dtype
        x = jnp.zeros(sample_shape + (self.input_dim,), dtype=dtype)

        def step(carry, inputs):
            i, key = inputs
            params = self.predict_params(carry, context)
            params_i = jnp.take(params, i, axis=-2)
            natural = self.family.unpack(params_i)
            draw = self.family.rvs(key, natural).astype(dtype)
            new = draw[..., None]
            if prefix_z is not None:
                given = prefix_z[..., i][..., None]
                new = jnp.where(i < prefix_len, given, new)
            # Only dimension i is written; the mask guarantees the parameters
            # used here depended on x_<i alone.
            onehot = jnp.arange(self.input_dim) == i
            return jnp.where(onehot, new, carry), None

        x, _ = jax.lax.scan(step, x, (jnp.arange(self.input_dim), keys))
        return self._unstandardize(x) if self.standardize else x

    def _sample_cached(self, keys, sample_shape, context, prefix_z, prefix_len):
        """KV-cached ancestral sampling for transformer conditioners.

        Compiled to a single program with :func:`flax.nnx.scan`: the
        attention caches are threaded through as scan carry (the model is
        passed explicitly so its ``Cache`` state is lifted) while the
        parameters stay put. Cache updates apply to this model in place,
        so every call starts with a fresh :meth:`init_decode`.
        """
        dtype = self.family.event_dtype
        family = self.family
        feature_width = self.conditioner.feature_width
        input_dim = self.input_dim
        self.conditioner.init_decode(sample_shape, dtype=jnp.float32)
        carry0 = (
            jnp.zeros(sample_shape + (input_dim,), dtype=dtype),
            # Encoder input space is floating point (one-hot for discrete).
            jnp.zeros(sample_shape + (feature_width,), jnp.float32),
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
            length=input_dim,
        )
        def step(carry, i, model):
            x, prev_feat = carry
            params_i = model.conditioner.predict_next_params(
                prev_feat, i, context
            )
            natural = family.unpack(params_i)
            draw = family.rvs(keys[i], natural).astype(dtype)
            value = draw
            if prefix_z is not None:
                value = jnp.where(i < prefix_len, prefix_z[..., i], value)
            x = x.at[..., i].set(value)
            encoded = family.encode(value)
            prev_feat = encoded[..., None] if feature_width == 1 else encoded
            return (x, prev_feat), draw

        (x, _), _ = step(carry0, jnp.arange(input_dim), self)
        return self._unstandardize(x) if self.standardize else x

    def _sample_base(self, rng, sample_shape, spec):
        """Unused noise; the per-dimension draws carry the randomness."""
        del spec
        return jnp.zeros(tuple(sample_shape) + (self.input_dim,))

    def _distribution_sampler(
        self,
        event_spec=None,
        *,
        dtype=None,
        context_spec=None,
    ):
        if event_spec is None:
            event_spec = (self.input_dim,)
        if dtype is None:
            dtype = self.family.event_dtype

        if not self.conditioner_config.exportable:
            return _EagerARSampler(self, (self.input_dim,))

        def make_sample_fn(graphdef, with_context):
            def sample_fn(state, rng, eps, *rest):
                model = nnx.merge(graphdef, state)
                context = rest[0] if with_context else None
                batch = eps.shape[0]
                return model.sample(rng, (batch,), context=context)

            return sample_fn

        return _KeyNormalizingSampler(
            self._build_exported_sampler(
                ("autoregressive-sampler",),
                event_spec,
                make_sample_fn,
                dtype=dtype,
                stochastic=True,
                context_spec=context_spec,
            )
        )

    def _distribution_logpdf(
        self,
        event_spec=None,
        *,
        dtype=jnp.float32,
        context_spec=None,
    ):
        if event_spec is None:
            event_spec = (self.input_dim,)

        def make_logpdf_fn(graphdef):
            if context_spec is None:

                def logpdf_fn(current_state, value):
                    model = nnx.merge(graphdef, current_state)
                    return jax.vmap(model._logpdf)(value)

            else:

                def logpdf_fn(current_state, value, context):
                    model = nnx.merge(graphdef, current_state)
                    return jax.vmap(
                        lambda item, cond: model._logpdf(item, context=cond)
                    )(value, context)

            return logpdf_fn

        return self._build_exported_logpdf(
            ("autoregressive-logpdf",),
            event_spec,
            make_logpdf_fn,
            dtype=dtype,
            context_spec=context_spec,
        )

    def as_dist(self, event_spec=None, **kwargs):
        if event_spec is None:
            event_spec = jax.ShapeDtypeStruct(
                (self.input_dim,), self.family.event_dtype
            )
        return super().as_dist(event_spec, **kwargs)

    def fit(self, rng, data, **kwargs):
        """Fit the standardising transform once, then train as usual."""
        self.fit_standardization(data)
        # `loss` goes through `_logpdf`, which standardises internally.
        return super().fit(rng, data, **kwargs)

    def _default_fit_kwargs(self) -> dict:
        return {"schedule": "warmup_cosine"}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class MADE(Autoregressive):
    """Gaussian conditionals over a masked MLP (Germain et al., 2015)."""

    def __init__(self, input_dim, rngs, **kwargs):
        super().__init__(input_dim, ARFamily.normal(), rngs, **kwargs)


class MixtureAutoregressive(Autoregressive):
    """Mixture-density conditionals: the KDE-like flexible head."""

    def __init__(self, input_dim, rngs, *, num_components: int = 10, **kwargs):
        super().__init__(
            input_dim, ARFamily.mixture(num_components), rngs, **kwargs
        )


class SplineAutoregressive(Autoregressive):
    """Spline-warped normal conditionals.

    Distinct from ``SplineAutoregressiveFlow``: that composes spline bijections,
    this predicts a spline-warped *distribution* per dimension.
    """

    def __init__(self, input_dim, rngs, *, num_bins: int = 8, bound: float = 5.0, **kwargs):
        super().__init__(
            input_dim, ARFamily.spline(num_bins, bound), rngs, **kwargs
        )


class HistogramAutoregressive(Autoregressive):
    """Piecewise-constant conditionals with exponential tails."""

    def __init__(
        self,
        input_dim,
        rngs,
        *,
        num_bins: int = 32,
        low: float = -5.0,
        high: float = 5.0,
        tails: bool = True,
        **kwargs,
    ):
        super().__init__(
            input_dim, ARFamily.histogram(num_bins, low, high, tails), rngs, **kwargs
        )


class CategoricalAutoregressive(Autoregressive):
    """Categorical conditionals over integer-valued data."""

    def __init__(self, input_dim, rngs, *, num_categories: int, **kwargs):
        super().__init__(
            input_dim, ARFamily.categorical(num_categories), rngs, **kwargs
        )


made = MADE
