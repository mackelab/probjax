"""Configuration for autoregressive density models.

Two axes:

* **family** — :class:`ARFamily` wraps a univariate distribution from
  :mod:`probjax.stats` and turns a conditioner's unconstrained parameter vector
  into that distribution's parameters. Everything else is derived from the
  distribution itself: ``dist.parameters`` gives the names, order and
  constraints, ``dist.param_sizes`` (when defined) gives the vector lengths, and
  ``logpdf`` / ``_rvs_impl`` provide the density and sampler. So any univariate
  family works, including the flexible ones added for this purpose
  (``mixture_kernel``, ``histogram``, ``tailed_histogram``, ``spline_normal``).
* **conditioner** — the masked network that emits those vectors given ``x_<i``.

The parameter packing mirrors
:mod:`probjax.nn.generative.nflows.config`, and deliberately reuses its
constrainers so that a zero-initialised head produces a sane distribution
(a standard normal, a flat histogram, an identity spline) rather than something
degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer
from jax import Array

from probjax.nn.generative.nflows.config import (
    WidthRule,
    default_hidden_dims,
    default_model_dim,
    resolve_hidden_dims,
)
from probjax.stats import constraints as _c
from probjax.stats.base import rv_generic
from probjax.stats.constraint_registry import biject_to

__all__ = [
    "ARConditionerConfig",
    "ARFamily",
    "MLPARConditionerConfig",
    "TransformerARConditionerConfig",
]


# =============================================================================
# Parameter constrainers
# =============================================================================


@dataclass(frozen=True)
class _Constrainer:
    """Maps an unconstrained block onto one natural parameter.

    ``raw_size`` exists because the two need not agree: spline knot positions
    are ``n`` values spanning a fixed interval with exact endpoints, which has
    only ``n - 1`` degrees of freedom.
    """

    fn: Callable[[Array], Array]
    raw_size: Callable[[int], int] = lambda n: n


def _default_constrainer(name: str, constraint) -> _Constrainer:
    """Pick a stable, identity-at-zero constrainer for a parameter.

    Neither registry default is right for a network head:
    ``transform_to(strict_positive)`` is ``max(abs, eps)``, which is
    non-smooth and maps 0 to 0, and ``biject_to(strict_positive)`` is ``exp``,
    unbounded above -- the failure mode removed from the affine flow scale.
    These reuse the bounded, identity-at-zero constrainers from the flow config.
    """
    from probjax.nn.generative.nflows.config import _positive

    if isinstance(constraint, _c.Simplex):
        # The constraint says "probability vector", so emit probabilities --
        # families that want log-weights declare them as `real` instead.
        return _Constrainer(lambda raw: jax.nn.softmax(raw, axis=-1))
    if isinstance(constraint, _c.UnitInterval):
        return _Constrainer(jax.nn.sigmoid)
    if isinstance(constraint, (_c.StrictPositive, _c.Positive)):
        return _Constrainer(lambda raw: _positive(raw, 1e-3, "tanh", 10.0))
    if isinstance(constraint, _c.Integer):
        raise TypeError(
            f"Parameter {name!r} has an integer constraint "
            f"({type(constraint).__name__}), which cannot be predicted by a "
            "network: there is no differentiable map onto it. Pin it with "
            f"fixed={{{name!r}: ...}}."
        )
    if isinstance(constraint, _c.Real):
        return _Constrainer(lambda raw: raw)

    try:
        transform = biject_to(constraint)
    except NotImplementedError as exc:  # pragma: no cover - depends on registry
        raise TypeError(
            f"No differentiable constrainer for parameter {name!r} with "
            f"constraint {type(constraint).__name__}. Pin it with fixed=, or "
            "pass an explicit constrain= entry."
        ) from exc
    return _Constrainer(transform)


def _spline_knot_constrainer(lo: float, hi: float, min_bin_size: float = 1e-4):
    """Knot positions spanning ``[lo, hi]`` exactly, from ``n - 1`` raw widths."""
    from probjax.nn.generative.nflows.config import _bin_positions

    return _Constrainer(
        lambda raw: _bin_positions(raw, lo, hi, min_bin_size),
        raw_size=lambda n: n - 1,
    )


def _spline_slope_constrainer(max_slope: float = 10.0):
    from probjax.nn.generative.nflows.config import _knot_slopes

    return _Constrainer(lambda raw: _knot_slopes(raw, 1e-3, max_slope))


# =============================================================================
# The family adapter
# =============================================================================


@dataclass
class ARFamily:
    """A univariate ``probjax.stats`` family used as a conditional head.

    Parameters
    ----------
    dist :
        The distribution generator, e.g. ``norm``, ``laplace``, ``histogram``.
    hyper :
        Hyperparameters fixing the parameter vector lengths, forwarded to
        ``dist.param_sizes`` (e.g. ``num_components``, ``num_bins``).
    fixed :
        Parameters that are *not* predicted, supplied as constants instead.
        Integer-constrained parameters must go here.
    constrain :
        Per-parameter overrides of the default constrainer.
    discrete :
        Whether the data are integer-valued; drives the event dtype and the
        one-hot encoding of the conditioner's input.
    """

    dist: rv_generic
    hyper: Mapping[str, Any] = field(default_factory=dict)
    fixed: Mapping[str, Any] = field(default_factory=dict)
    constrain: Mapping[str, _Constrainer] = field(default_factory=dict)
    sizes: Mapping[str, int] = field(default_factory=dict)
    discrete: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.fixed) - set(self.dist.parameters)
        if unknown:
            raise ValueError(
                f"fixed names {sorted(unknown)} are not parameters of "
                f"{self.dist.name!r}; expected some of "
                f"{list(self.dist.parameters)}."
            )
        if not self._predicted:
            raise ValueError(
                f"Every parameter of {self.dist.name!r} is fixed, leaving "
                "nothing for the conditioner to predict."
            )

    # -- parameter layout ---------------------------------------------------

    @property
    def _predicted(self) -> list[str]:
        """Parameter names the network must emit, in declaration order."""
        return [n for n in self.dist.parameters if n not in self.fixed]

    @property
    def _natural_sizes(self) -> dict[str, int]:
        """Natural (constrained) size of every parameter."""
        sizes_fn = getattr(self.dist, "param_sizes", None)
        derived = (
            {n: 1 for n in self.dist.parameters}
            if sizes_fn is None
            else dict(sizes_fn(**self.hyper))
        )
        # `sizes` covers families that predate the `param_sizes` convention but
        # still have vector parameters (categorical's `probs`, for instance).
        derived.update(self.sizes)
        return derived

    @property
    def _constrainers(self) -> dict[str, _Constrainer]:
        out = {}
        for name in self._predicted:
            if name in self.constrain:
                out[name] = self.constrain[name]
            else:
                out[name] = _default_constrainer(name, self.dist.parameters[name])
        return out

    def raw_sizes(self) -> dict[str, int]:
        """Unconstrained block length the conditioner must emit per parameter."""
        natural = self._natural_sizes
        return {
            name: c.raw_size(natural.get(name, 1))
            for name, c in self._constrainers.items()
        }

    def params_dim(self) -> int:
        """Total parameters the conditioner emits per data dimension."""
        return sum(self.raw_sizes().values())

    def unpack(self, params: Array) -> dict[str, Array]:
        """Split a ``(..., params_dim)`` vector into constrained parameters."""
        params = jnp.asarray(params)
        if params.shape[-1] != self.params_dim():
            raise ValueError(
                f"{self.dist.name!r} head expects params with trailing "
                f"dimension {self.params_dim()}, got {params.shape[-1]}."
            )
        sizes = self.raw_sizes()
        constrainers = self._constrainers

        natural: dict[str, Any] = {}
        start = 0
        for name in self._predicted:
            width = sizes[name]
            block = params[..., start : start + width]
            if width == 1 and self._natural_sizes.get(name, 1) == 1:
                block = block[..., 0]  # scalar parameter
            natural[name] = constrainers[name].fn(block)
            start += width
        natural.update(self.fixed)
        return natural

    def params_init(self) -> Initializer:
        """Zeros: every constrainer is neutral there."""
        return nnx.initializers.zeros

    # -- density and sampling, straight from the distribution ---------------

    def logpdf(self, x: Array, natural: Mapping[str, Array]) -> Array:
        return self.dist.logpdf(x, **natural)

    def rvs(self, rng, natural: Mapping[str, Array]) -> Array:
        """One draw per batch element, using the family's own sampler."""
        return self.dist._rvs_impl(rng, **natural, shape=())

    # -- data plumbing ------------------------------------------------------

    @property
    def event_dtype(self):
        return jnp.int32 if self.discrete else jnp.float32

    def encode(self, x: Array) -> Array:
        """Conditioner input encoding.

        Integer labels carry no usable metric, so a discrete family one-hots
        them; continuous data passes through.
        """
        if not self.discrete:
            return jnp.asarray(x)
        num_classes = self._natural_sizes[self._predicted[0]]
        return jax.nn.one_hot(jnp.asarray(x).astype(jnp.int32), num_classes)

    def bounded_support(self) -> Optional[tuple[float, float]]:
        """``(low, high)`` when the family's support is a bounded interval.

        Used to reject out-of-range training data loudly instead of silently
        returning ``-inf``.
        """
        try:
            support = self.dist.support(**self.fixed)
        except Exception:  # pragma: no cover - families with required params
            return None
        lower = getattr(support, "lower", None)
        upper = getattr(support, "upper", None)
        if lower is None or upper is None:
            return None
        if not (jnp.isfinite(jnp.asarray(lower)) and jnp.isfinite(jnp.asarray(upper))):
            return None
        return float(lower), float(upper)

    # -- convenience constructors -------------------------------------------

    @classmethod
    def normal(cls) -> "ARFamily":
        from probjax.stats import norm

        return cls(norm)

    @classmethod
    def mixture(
        cls,
        num_components: int = 10,
        kernel: str = "norm",
        spread: float = 2.0,
    ) -> "ARFamily":
        """Mixture head, with the component locations pulled apart at init.

        The offset is not cosmetic. A zero-initialised conditioner emits the
        same vector for every component, so all of them share a location, a
        scale *and* a gradient -- the mixture is exactly one kernel and stays
        that way, since nothing breaks the symmetry. Spreading the locations
        deterministically makes each component see a different gradient from
        the first step, at no cost to the neutral density being sensible.
        """
        from probjax.stats import logistic_mixture_kernel, mixture_kernel

        dist = mixture_kernel if kernel == "norm" else logistic_mixture_kernel
        offsets = jnp.linspace(-spread, spread, num_components)
        return cls(
            dist,
            hyper={"num_components": num_components},
            constrain={"locs": _Constrainer(lambda raw: raw + offsets)},
        )

    @classmethod
    def histogram(
        cls, num_bins: int = 32, low: float = -5.0, high: float = 5.0, tails: bool = True
    ) -> "ARFamily":
        """Piecewise-constant head; ``tails`` keeps the support unbounded."""
        from probjax.stats import histogram as _hist
        from probjax.stats import tailed_histogram as _thist

        dist = _thist if tails else _hist
        constrain = {}
        if tails:
            # A raw 0 would otherwise put sigmoid(0) = half the mass in the
            # tails; offset the logit so the neutral state is a near-flat
            # histogram with only ~5% tail mass.
            constrain["tail_logit"] = _Constrainer(lambda raw: raw - 3.0)
        return cls(
            dist,
            hyper={"num_bins": num_bins},
            fixed={"low": low, "high": high},
            constrain=constrain,
        )

    @classmethod
    def spline(
        cls, num_bins: int = 8, bound: float = 5.0, latent_bound: float = 5.0
    ) -> "ARFamily":
        """Spline-warped normal head; knots span the given ranges exactly.

        ``latent_bound`` defaults to ``bound`` so that zero parameters give
        matching knots with unit slopes -- an identity spline, i.e. exactly a
        standard normal.
        """
        from probjax.stats import spline_normal

        return cls(
            spline_normal,
            hyper={"num_bins": num_bins},
            constrain={
                "x_pos": _spline_knot_constrainer(-latent_bound, latent_bound),
                "y_pos": _spline_knot_constrainer(-bound, bound),
                "knot_slopes": _spline_slope_constrainer(),
            },
        )

    @classmethod
    def categorical(cls, num_categories: int) -> "ARFamily":
        from probjax.stats import categorical

        return cls(
            categorical,
            sizes={"probs": num_categories},
            discrete=True,
        )


# =============================================================================
# Conditioner configs
# =============================================================================


@runtime_checkable
class ARConditionerConfig(Protocol):
    """Builds the masked network that emits per-dimension parameters."""

    def build(
        self,
        input_dim: int,
        params_dim: int,
        *,
        in_features: int,
        context_features: Optional[int],
        rngs: nnx.Rngs,
    ) -> nnx.Module: ...

    @property
    def exportable(self) -> bool:
        """Whether the resulting sampler can go through ``jax.export``."""
        ...


def _autoregressive_masks(dims: Sequence[int], num_vars: int) -> list:
    """MADE masks where the input width may exceed the variable count.

    ``get_autoregressive_masks`` derives the degrees from ``dims[0]``, which
    assumes one input unit per autoregressive variable. That breaks for a
    one-hot encoded discrete model, where each variable occupies ``num_classes``
    adjacent units: every slot would get its own degree and dimension ``i``
    would end up conditioned on part of its own value.
    """
    in_features = dims[0]
    if in_features % num_vars != 0:
        raise ValueError(
            f"input width {in_features} is not a multiple of the "
            f"{num_vars} autoregressive variables."
        )
    width = in_features // num_vars
    # slot j belongs to variable j // width
    in_degrees = jnp.repeat(jnp.arange(num_vars), width) + 1

    masks = []
    for layer_idx in range(len(dims) - 1):
        x1 = (
            in_degrees.reshape(-1, 1)
            if layer_idx == 0
            else jnp.arange(dims[layer_idx]).reshape(-1, 1) % num_vars + 1
        )
        out_dim = dims[layer_idx + 1]
        if layer_idx == len(dims) - 2:
            if out_dim % num_vars != 0:
                raise ValueError(
                    "final layer size must be a multiple of the variable count."
                )
            x2 = jnp.repeat(jnp.arange(num_vars), out_dim // num_vars).reshape(1, -1) + 1
        else:
            x2 = jnp.arange(out_dim).reshape(1, -1) % num_vars + 1
        # strict at the input so dimension i never sees x_i
        masks.append(x2 > x1 if layer_idx == 0 else x2 >= x1)
    return masks


class _MaskedConditioner(nnx.Module):
    """MADE-style masked MLP returning ``(..., input_dim * params_dim)``."""

    def __init__(self, net: nnx.Module, input_dim: int, params_dim: int):
        self.net = net
        self.input_dim = input_dim
        self.params_dim = params_dim

    def __call__(self, x, context=None, *, rng=None):
        return self.net(x, context, rng=rng)


@dataclass
class MLPARConditionerConfig(ARConditionerConfig):
    """Masked MLP conditioner (MADE).

    Uses ``get_autoregressive_masks`` and ``MaskedMLP`` directly rather than
    ``AutoregressiveMLP``, whose constructor requires an invertible bijector and
    eagerly builds its inverse -- machinery an AR density head has no use for.

    Only the ``"grouped"`` mask order is used, which puts dimension ``i``'s
    parameters in a contiguous block at ``[i * p : (i + 1) * p]``.
    """

    hidden_dims: "Sequence[int] | WidthRule" = default_hidden_dims
    activation: Callable = jax.nn.gelu
    norm_cls: Optional[type] = None
    init_last_layer_to_zero: bool = True
    #: Training-set size, when known; the width rule scales with it.
    num_examples: Optional[int] = None

    @property
    def exportable(self) -> bool:
        return True

    def build(self, input_dim, params_dim, *, in_features, context_features, rngs):
        from probjax.nn.nets.simple import MaskedMLP

        out_features = input_dim * params_dim
        hidden = resolve_hidden_dims(
            self.hidden_dims, in_features, out_features, self.num_examples
        )
        dims = [in_features] + list(hidden) + [out_features]
        masks = _autoregressive_masks(dims, input_dim)
        net = MaskedMLP(
            dims,
            masks,
            rngs=rngs,
            context_features=context_features,
            norm_cls=self.norm_cls,
            activation=self.activation,
        )
        if self.init_last_layer_to_zero:
            last_kernel = net.layers[-1].kernel
            last_kernel[...] = jnp.zeros_like(last_kernel[...])
        return _MaskedConditioner(net, input_dim, params_dim)


class _TokenConditioner(nnx.Module):
    """Causal transformer over one token per dimension."""

    def __init__(self, inner: nnx.Module, input_dim: int, params_dim: int):
        self.inner = inner
        self.input_dim = input_dim
        self.params_dim = params_dim

    def __call__(self, x, context=None, *, rng=None):
        tokens = jnp.asarray(x)[..., :, None]
        out = self.inner.predict_bij_params(tokens, context, rng=rng)
        return out.reshape(out.shape[:-2] + (self.input_dim * self.params_dim,))


@dataclass
class TransformerARConditionerConfig(ARConditionerConfig):
    """Causal-transformer conditioner.

    Wraps :class:`AutoregressiveTransformer` purely for its
    ``predict_bij_params`` path; the bijector it is constructed with is never
    invoked, so a trivial one is supplied.

    Not exportable: attention's sharded primitive needs concrete shapes, so
    ``as_dist`` falls back to an eager sampler (the same limitation the flow
    transformer conditioner carries).
    """

    model_dim: "int | WidthRule" = default_model_dim
    num_examples: Optional[int] = None
    num_heads: int = 4
    num_layers: int = 4
    attn_size: int = 8
    widening_factor: int = 2

    @property
    def exportable(self) -> bool:
        return False

    def _resolved_model_dim(self, params_dim: int) -> int:
        """Token width, kept divisible by the head count."""
        if callable(self.model_dim):
            width = int(self.model_dim(1, params_dim, self.num_examples)[0])
        else:
            width = int(self.model_dim)
        return max(self.num_heads, width - width % self.num_heads)

    def build(self, input_dim, params_dim, *, in_features, context_features, rngs):
        del in_features  # tokens are per-dimension scalars
        from probjax.nn.generative.nflows.autoregressive import (
            AutoregressiveTransformer,
        )

        inner = AutoregressiveTransformer(
            1,  # one scalar feature per token
            params_dim,
            lambda params, x: x,  # never called; only predict_bij_params is used
            rngs,
            context_dim=context_features,
            model_dim=self._resolved_model_dim(params_dim),
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            attn_size=self.attn_size,
            widening_factor=self.widening_factor,
        )
        return _TokenConditioner(inner, input_dim, params_dim)
