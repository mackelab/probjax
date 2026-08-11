"""Configuration classes for normalizing flows.

Three orthogonal axes, mirroring the schedule / preconditioning / training /
solver split used by :mod:`probjax.nn.generative.diffusion.config` and
:mod:`probjax.nn.generative.flow_matching.config`:

* **bijector** — maps a well-behaved unconstrained parameter vector onto the
  *natural* parameters of a bijection in :mod:`probjax.stats.bijective`
  (knot positions, slopes, mixture log-weights, ...). This is where every
  ``softmax`` / ``softplus`` / ``cumsum`` lives; the stats layer sees only
  already-constrained quantities.
* **conditioner** — the neural architecture that emits those parameter vectors
  (masked MLP or causal transformer, coupling or autoregressive).
* **mixing** — the fixed or learnable layer interleaved between transforms.

:class:`NFlowConfig` and its three subclasses bundle the axes together with the
flow's shape. Every bijector config is *itself* the ``bijector`` callable the
conditioners expect, so ``config.bijector`` can be passed straight through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Callable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer
from jax import Array
from jax.typing import ArrayLike

from probjax.stats.bijective import (
    affine,
    bernstein,
    deep_sigmoid,
    mixture_cdf,
    monotone_hermite_cubic_spline,
    piecewise_affine_spline,
    rational_linear_spline,
    rational_quadratic_spline,
    shift,
    sos_polynomial,
    unconstrained_monotone,
)

__all__ = [
    # protocols
    "BijectorConfigProtocol",
    "ConditionerConfigProtocol",
    "MixingConfigProtocol",
    # bijector configs
    "AffineBijectorConfig",
    "BernsteinBijectorConfig",
    "DeepSigmoidBijectorConfig",
    "MixtureCDFBijectorConfig",
    "MonotoneHermiteCubicSplineConfig",
    "PiecewiseAffineSplineConfig",
    "RationalLinearSplineConfig",
    "RationalQuadraticSplineConfig",
    "ShiftBijectorConfig",
    "SumOfSquaresBijectorConfig",
    "UMNNBijectorConfig",
    # conditioner configs
    "MLPConditionerConfig",
    "TransformerConditionerConfig",
    # mixing configs
    "FlipMixingConfig",
    "NoMixingConfig",
    "PermuteMixingConfig",
    "RotationMixingConfig",
    # flow configs
    "AutoregressiveNFlowConfig",
    "CouplingNFlowConfig",
    "ElementwiseNFlowConfig",
    "NFlowConfig",
]

_EPS = 1e-6


# =============================================================================
# Parameter packers
# =============================================================================


def _softplus_identity_offset(minimum: float) -> float:
    """Pre-activation value at which ``softplus(.) + minimum`` equals ``1.0``."""
    if minimum >= 1.0:
        raise ValueError(f"minimum must be less than 1; got {minimum}.")
    return math.log(math.expm1(1.0 - minimum))


def _positive(
    raw: Array,
    minimum: float,
    transform: str = "softplus",
    max_scale: float = 10.0,
) -> Array:
    """Map ``raw`` to a positive value, with ``raw == 0`` giving exactly ``1.0``.

    The identity at zero is what makes zero-initialised conditioners emit the
    identity bijection, so every family here starts from a well-conditioned map.

    ``"tanh"`` additionally bounds the result into ``[1/max_scale, max_scale]``.
    Both other transforms are unbounded above and bottom out at ``minimum``,
    which lets a conditioner driven far off the data manifold produce a scale
    extreme enough to overflow the composed inverse.
    """
    if transform == "tanh":
        if max_scale <= 1.0:
            raise ValueError(f"max_scale must exceed 1; got {max_scale}.")
        return jnp.exp(jnp.tanh(raw) * math.log(max_scale))
    if transform == "softplus":
        return jax.nn.softplus(raw + _softplus_identity_offset(minimum)) + minimum
    if transform == "exp":
        if minimum >= 1.0:
            raise ValueError(f"minimum must be less than 1; got {minimum}.")
        return jnp.exp(raw) * (1.0 - minimum) + minimum
    raise ValueError(f"Unknown positivity transform {transform!r}.")


def _bin_positions(raw_widths: Array, lo: float, hi: float, min_bin_size: float) -> Array:
    """``(..., num_bins)`` raw widths -> ``(..., num_bins + 1)`` boundaries.

    The endpoints land on ``lo`` / ``hi`` exactly, and every bin is at least
    ``min_bin_size`` of the total width.
    """
    num_bins = raw_widths.shape[-1]
    widths = min_bin_size + (1.0 - min_bin_size * num_bins) * jax.nn.softmax(
        raw_widths, axis=-1
    )
    interior = jnp.cumsum(widths, axis=-1)[..., :-1]
    leading = raw_widths.shape[:-1]
    lo_ = jnp.full(leading + (1,), lo, raw_widths.dtype)
    hi_ = jnp.full(leading + (1,), hi, raw_widths.dtype)
    return jnp.concatenate([lo_, lo + (hi - lo) * interior, hi_], axis=-1)


def _knot_slopes(raw: Array, min_knot_slope: float, max_knot_slope: float) -> Array:
    """``(..., num_bins + 1)`` positive slopes in ``[1/max, max]``, 1 at ``raw == 0``.

    Bounding matters most at the two ends: the spline's tails extrapolate
    linearly with the boundary slopes, so an unbounded slope compounds to
    ``s**L`` across ``L`` stacked transforms.
    """
    del min_knot_slope  # the tanh parameterization is symmetric in log space
    return _positive(raw, 0.0, "tanh", max_knot_slope)


def _log_simplex(raw: Array) -> Array:
    """Point on the log-simplex; kept in log space for the stats layer."""
    return jax.nn.log_softmax(raw, axis=-1)


# =============================================================================
# Bijector configs
# =============================================================================


@runtime_checkable
class BijectorConfigProtocol(Protocol):
    """Maps an unconstrained parameter vector to a natural-parameter bijection."""

    def params_dim(self) -> int:
        """Number of parameters consumed per scalar element."""
        ...

    def unpack(self, params: Array) -> Tuple[Array, ...]:
        """Split and constrain ``params`` into natural parameters."""
        ...

    def apply(self, x: Array, *natural: Array) -> Array:
        """Call the underlying :mod:`probjax.stats.bijective` function."""
        ...

    def params_init(self) -> Initializer:
        """Initializer for a free (non-conditioned) parameter table."""
        ...

    def __call__(self, params: Array, x: Array) -> Array: ...


@dataclass
class BaseBijectorConfig(BijectorConfigProtocol):
    """Shared plumbing: batching and the identity-at-zero initializer.

    Subclasses implement :meth:`params_dim`, :meth:`unpack` and :meth:`apply`.
    """

    def params_init(self) -> Initializer:
        # Every family below is constructed so that zero params give the
        # identity map, so zeros is the right free-parameter initializer.
        return nnx.initializers.zeros

    def unpack(self, params: Array) -> Tuple[Array, ...]:
        raise NotImplementedError

    def apply(self, x: Array, *natural: Array) -> Array:
        raise NotImplementedError

    def params_dim(self) -> int:
        raise NotImplementedError

    @cached_property
    def _vectorized(self) -> Callable[[Array, Array], Array]:
        """Lift the scalar core to arbitrary batch shapes.

        The stats layer takes a scalar ``x`` with 1-D natural params;
        ``jnp.vectorize`` with an explicit signature is the single batching
        path for every family (vmap composes with ``custom_inverse_call_p``
        through its registered batching rule).
        """

        def fn(params, x):
            return self.apply(x, *self.unpack(params))

        return jnp.vectorize(fn, signature=f"({self.params_dim()}),()->()")

    def __call__(self, params: ArrayLike, x: ArrayLike, **kwargs) -> Array:
        del kwargs  # conditioners may forward bijector kwargs; none are used
        params = jnp.asarray(params)
        # jnp.vectorize does not check a literal core dimension against the
        # signature, so a mis-shaped params vector would silently reuse one
        # element's block for every element. Catch it here instead.
        if params.shape[-1] != self.params_dim():
            raise ValueError(
                f"{type(self).__name__} expects params with trailing dimension "
                f"{self.params_dim()}, got {params.shape[-1]}. A flat "
                "multi-element parameter vector must be reshaped to "
                "(..., num_elements, params_dim) first."
            )
        return self._vectorized(params, jnp.asarray(x))


# ---- elementwise affine -----------------------------------------------------


@dataclass
class ShiftBijectorConfig(BaseBijectorConfig):
    """``y = x + loc`` (NICE-style additive coupling)."""

    def params_dim(self) -> int:
        return 1

    def unpack(self, params: Array) -> Tuple[Array, ...]:
        return (params[..., 0],)

    def apply(self, x: Array, loc: Array) -> Array:
        return shift(x, loc)


@dataclass
class AffineBijectorConfig(BaseBijectorConfig):
    """``y = loc + scale * x``.

    The default bounds the scale into ``[1/max_scale, max_scale]``. An
    unbounded scale is the classic affine-coupling failure mode: a conditioner
    evaluated far from the data can emit a scale extreme enough that composing
    the inverse over several layers overflows, leaving the log-density
    non-finite off-support (on a 2-D checkerboard, a quarter of the plane).
    ``"softplus"`` and ``"exp"`` keep the unbounded behaviour.
    """

    min_scale: float = 1e-3
    max_scale: float = 10.0
    scale_transform: Literal["tanh", "softplus", "exp"] = "tanh"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_scale < 1.0:
            raise ValueError(f"min_scale must be in [0, 1); got {self.min_scale}.")
        if self.max_scale <= 1.0:
            raise ValueError(f"max_scale must exceed 1; got {self.max_scale}.")

    def params_dim(self) -> int:
        return 2

    def unpack(self, params: Array) -> Tuple[Array, ...]:
        loc, raw_scale = params[..., 0], params[..., 1]
        return loc, _positive(
            raw_scale, self.min_scale, self.scale_transform, self.max_scale
        )

    def apply(self, x: Array, loc: Array, scale: Array) -> Array:
        return affine(x, loc, scale)


# ---- splines ----------------------------------------------------------------


@dataclass
class BaseSplineConfig(BaseBijectorConfig):
    """Shared knot packing for the four spline families.

    The parameter vector is ``num_bins`` raw x-widths, ``num_bins`` raw
    y-widths, and (for the slope-carrying families) ``num_bins + 1`` raw
    slopes. Knots span ``[x_min, x_max]`` exactly; outside that interval the
    map extrapolates linearly with the boundary knot slopes, so it stays a
    bijection on the whole real line.

    The bounds should bracket the data: knots outside its support are wasted
    capacity, and data outside the knots falls in the linear tails. The default
    suits roughly standardised data (a standard-normal base), and measurably
    beats a wider domain -- on a 2-D checkerboard, +-5 reaches 3.59 nats
    against 3.64 at +-10, and on a spiral 2.44 against 2.58.

    Setting ``bounded`` replaces the linear tails with a clamp onto
    ``[y_min, y_max]``, for modelling data on a genuinely compact domain. The
    result is a bijection **only on** ``[x_min, x_max]``: outside it the map is
    constant and the density is zero. Every value reaching the layer must
    therefore lie inside the domain -- base samples and the outputs of all
    preceding layers included -- so pair it with a bounded base distribution.
    It is wrong with the default standard-normal base, whose samples are
    unbounded, and it is off by default for that reason.
    """

    num_bins: int = 16
    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -5.0
    y_max: float = 5.0
    min_bin_size: float = 1e-4
    min_knot_slope: float = 1e-4
    max_knot_slope: float = 10.0
    bounded: bool = False

    _has_slopes: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_bins < 1:
            raise ValueError(f"num_bins must be positive; got {self.num_bins}.")
        if self.min_bin_size * self.num_bins >= 1.0:
            raise ValueError(
                f"min_bin_size * num_bins must be < 1; got "
                f"{self.min_bin_size} * {self.num_bins}."
            )
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Spline bounds must satisfy x_min < x_max and y_min < y_max.")

    @property
    def _bounds(self) -> dict:
        """Domain bounds forwarded to the core spline only when clamping."""
        if not self.bounded:
            return {}
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }

    def params_dim(self) -> int:
        return 3 * self.num_bins + 1 if self._has_slopes else 2 * self.num_bins

    def unpack(self, params: Array) -> Tuple[Array, ...]:
        n = self.num_bins
        raw_x, raw_y = params[..., :n], params[..., n : 2 * n]
        x_pos = _bin_positions(raw_x, self.x_min, self.x_max, self.min_bin_size)
        y_pos = _bin_positions(raw_y, self.y_min, self.y_max, self.min_bin_size)
        if not self._has_slopes:
            return x_pos, y_pos
        slopes = _knot_slopes(
            params[..., 2 * n :], self.min_knot_slope, self.max_knot_slope
        )
        return x_pos, y_pos, slopes



@dataclass
class RationalQuadraticSplineConfig(BaseSplineConfig):
    """Durkan et al. (2019) rational-quadratic spline."""

    def apply(self, x, x_pos, y_pos, knot_slopes):
        return rational_quadratic_spline(x, x_pos, y_pos, knot_slopes, **self._bounds)


@dataclass
class RationalLinearSplineConfig(BaseSplineConfig):
    """Degree-1/1 rational-linear spline (analytic inverse, no square root)."""

    def apply(self, x, x_pos, y_pos, knot_slopes):
        return rational_linear_spline(x, x_pos, y_pos, knot_slopes, **self._bounds)


@dataclass
class MonotoneHermiteCubicSplineConfig(BaseSplineConfig):
    """Fritsch-Carlson monotone cubic Hermite spline."""

    def apply(self, x, x_pos, y_pos, knot_slopes):
        return monotone_hermite_cubic_spline(
            x, x_pos, y_pos, knot_slopes, **self._bounds
        )


@dataclass
class PiecewiseAffineSplineConfig(BaseSplineConfig):
    """Piecewise-linear spline; no knot slopes."""

    _has_slopes: bool = field(default=False, init=False, repr=False)

    def apply(self, x, x_pos, y_pos):
        return piecewise_affine_spline(x, x_pos, y_pos, **self._bounds)


# ---- monotone networks ------------------------------------------------------


@dataclass
class DeepSigmoidBijectorConfig(BaseBijectorConfig):
    """NAF deep sigmoidal transform (Huang et al., 2018)."""

    num_components: int = 8
    min_slope: float = 1e-3
    max_slope: float = 10.0

    def __post_init__(self) -> None:
        if self.num_components < 1:
            raise ValueError("num_components must be positive.")
        if self.max_slope <= 1.0:
            raise ValueError(f"max_slope must exceed 1; got {self.max_slope}.")

    def params_dim(self) -> int:
        return 3 * self.num_components

    def unpack(self, params: Array) -> Tuple[Array, ...]:
        k = self.num_components
        raw_w, raw_a, b = params[..., :k], params[..., k : 2 * k], params[..., 2 * k :]
        # Unit slopes at zero params make the layer start as the identity:
        # logit(mean_k sigmoid(t)) == t. The map's tail slope is min_k a_k, and
        # log(a) enters the log-det directly, so bound a on both sides.
        return (
            _log_simplex(raw_w),
            _positive(raw_a, self.min_slope, "tanh", self.max_slope),
            b,
        )

    def apply(self, x, log_weights, slopes, biases):
        return deep_sigmoid(x, log_weights, slopes, biases)


@dataclass
class UMNNBijectorConfig(BaseBijectorConfig):
    """UMNN monotone-integrand transform (Wehenkel & Louppe, 2019).

    The integrand's weights are genuinely unconstrained — positivity comes from
    the softplus head inside the bijection — so ``unpack`` only reshapes, apart
    from an offset on the output bias that puts the integrand at exactly 1 for
    zero params, i.e. starts the layer as the identity.
    """

    num_hidden: int = 8
    min_integrand: float = 1e-2

    def __post_init__(self) -> None:
        if self.num_hidden < 1:
            raise ValueError("num_hidden must be positive.")

    def params_dim(self) -> int:
        return 3 * self.num_hidden + 2

    def unpack(self, params: Array) -> Tuple[Array, ...]:
        k = self.num_hidden
        return (
            params[..., :k],  # hidden_weights
            params[..., k : 2 * k],  # hidden_biases
            params[..., 2 * k : 3 * k],  # out_weights
            params[..., 3 * k] + _softplus_identity_offset(self.min_integrand),
            params[..., 3 * k + 1],  # offset
        )

    def apply(self, x, hidden_weights, hidden_biases, out_weights, out_bias, offset):
        return unconstrained_monotone(
            x,
            hidden_weights,
            hidden_biases,
            out_weights,
            out_bias,
            offset,
            min_integrand=self.min_integrand,
        )


@dataclass
class SumOfSquaresBijectorConfig(BaseBijectorConfig):
    """Sum-of-squares polynomial flow (Jaini et al., 2019).

    The derivative is ``eps + sum_k poly_k(x / bound)^2``, so it is positive for
    any coefficients. Two packing steps keep the map well-conditioned: a
    constant added to the degree-0 coefficients makes zero params give slope 1,
    and dividing by that same slope keeps it exactly 1 as the params move away.

    ``bound`` is what makes the family usable at the default learning rate. The
    polynomial acts on ``x / bound`` and continues linearly outside
    ``|x| <= bound``, so the map grows linearly rather than as
    ``x**(2*degree+1)``. Evaluated on raw ``x`` the degree-7 default overflows
    float32 above ``|x| ~ 340`` and yields an infinite log-determinant, which
    used to make this the one family that diverged out of the box. Set ``bound``
    to cover the data, as for the spline families.
    """

    num_polys: int = 2
    degree: int = 3
    bound: float = 5.0

    def __post_init__(self) -> None:
        if self.num_polys < 1 or self.degree < 0:
            raise ValueError("num_polys must be positive and degree non-negative.")
        if self.bound <= 0.0:
            raise ValueError(f"bound must be positive; got {self.bound}.")

    def params_dim(self) -> int:
        return self.num_polys * (self.degree + 1) + 1

    def unpack(self, params: Array) -> Tuple[Array, ...]:
        raw, constant = params[..., :-1], params[..., -1]
        coeffs = jnp.reshape(
            raw, raw.shape[:-1] + (self.num_polys, self.degree + 1)
        )
        coeffs = coeffs.at[..., 0].add(math.sqrt(1.0 / self.num_polys))
        norm = jnp.sqrt(_EPS + jnp.sum(coeffs[..., 0] ** 2, axis=-1))
        return coeffs / norm[..., None, None], constant

    def apply(self, x, coefficients, constant):
        return sos_polynomial(x, coefficients, constant, bound=self.bound)


@dataclass
class BernsteinBijectorConfig(BaseBijectorConfig):
    """Monotone Bernstein polynomial with identity tails (Sick et al., 2021)."""

    degree: int = 16
    bound: float = 5.0

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be positive.")
        if self.bound <= 0.0:
            raise ValueError("bound must be positive.")

    def params_dim(self) -> int:
        return self.degree

    def unpack(self, params: Array) -> Tuple[Array, ...]:
        weights = jax.nn.softmax(params, axis=-1)
        zero = jnp.zeros(params.shape[:-1] + (1,), params.dtype)
        # theta: increasing from 0 to 1, length degree + 1
        return (jnp.concatenate([zero, jnp.cumsum(weights, axis=-1)], axis=-1),)

    def apply(self, x, theta):
        return bernstein(x, theta, bound=self.bound)


@dataclass
class MixtureCDFBijectorConfig(BaseBijectorConfig):
    """Gaussianization kernel layer: logistic-mixture CDF then logistic quantile."""

    num_components: int = 16
    min_scale: float = 1e-3
    max_scale: float = 10.0

    def __post_init__(self) -> None:
        if self.num_components < 1:
            raise ValueError("num_components must be positive.")
        if self.max_scale <= 1.0:
            raise ValueError(f"max_scale must exceed 1; got {self.max_scale}.")

    def params_init(self) -> Initializer:
        """Zeros except the location block, which is spread randomly.

        Identical locations would make every mixture component redundant, so
        unlike the other families this one needs a non-trivial free init.
        """
        k = self.num_components

        def init(key, shape, dtype=None):
            dtype = dtype or jnp.float32
            params = jnp.zeros(shape, dtype)
            locs = jax.random.normal(key, shape[:-1] + (k,), dtype)
            return params.at[..., k : 2 * k].set(locs)

        return init

    def params_dim(self) -> int:
        return 3 * self.num_components

    def unpack(self, params: Array) -> Tuple[Array, ...]:
        k = self.num_components
        raw_w, mu, raw_s = (
            params[..., :k],
            params[..., k : 2 * k],
            params[..., 2 * k :],
        )
        # Unit scales at zero params give logit(mean_k sigmoid(t)) == t. The
        # log-det carries -log(s) and the map divides by s, so bound it.
        return (
            _log_simplex(raw_w),
            mu,
            _positive(raw_s, self.min_scale, "tanh", self.max_scale),
        )

    def apply(self, x, log_weights, locs, scales):
        return mixture_cdf(x, log_weights, locs, scales)


# =============================================================================
# Conditioner configs
# =============================================================================


@runtime_checkable
class ConditionerConfigProtocol(Protocol):
    """Builds the network that emits bijector parameters."""

    def build_coupling(
        self,
        split_index: int,
        params_dim: int,
        bijector: Callable,
        *,
        context_features: Optional[int],
        rngs: nnx.Rngs,
    ) -> nnx.Module: ...

    def build_autoregressive(
        self,
        in_out_features: int,
        params_dim: int,
        bijector: Callable,
        *,
        context_features: Optional[int],
        output_order: str,
        rngs: nnx.Rngs,
    ) -> nnx.Module: ...


@dataclass
class MLPConditionerConfig(ConditionerConfigProtocol):
    """Dense conditioner: a plain MLP for coupling, a MADE-masked one otherwise."""

    hidden_dims: Sequence[int] = (128, 128)
    activation: Callable = jax.nn.gelu
    norm_cls: Optional[type] = None
    init_last_layer_to_zero: bool = True

    def build_coupling(
        self, split_index, params_dim, bijector, *, context_features, rngs
    ):
        from probjax.nn.generative.nflows.coupling import CouplingMLP

        return CouplingMLP(
            split_index,
            params_dim,
            bijector,
            rngs,
            context_dim=context_features,
            hidden_dims=tuple(self.hidden_dims),
            activation=self.activation,
            init_last_layer_to_zero=self.init_last_layer_to_zero,
        )

    def build_autoregressive(
        self, in_out_features, params_dim, bijector, *, context_features, output_order, rngs
    ):
        from probjax.nn.generative.nflows.autoregressive import AutoregressiveMLP

        return AutoregressiveMLP(
            in_out_features,
            params_dim,
            bijector,
            rngs,
            context_features=context_features,
            hidden_dims=tuple(self.hidden_dims),
            activation=self.activation,
            norm_cls=self.norm_cls,
            init_last_layer_to_zero=self.init_last_layer_to_zero,
            output_order=output_order,
        )


def _scalar_token_bijector(bijector: Callable) -> Callable:
    """Strip the length-1 token axis a sequence model carries around ``x``."""

    def fn(params, x, **kwargs):
        return bijector(params, jnp.asarray(x)[..., 0], **kwargs)[..., None]

    return fn


class _ScalarTokenConditioner(nnx.Module):
    """Present a flat feature vector to a sequence model as one token per scalar.

    :class:`AutoregressiveTransformer` attends over axis ``-2`` and treats the
    last axis as token features, which is the right contract for sequence data
    but not for the flat vectors a flow transforms. Adding and removing a
    trailing singleton axis makes the two meet; the reshapes are exactly
    invertible, so they compose with the inverse interpreter.
    """

    def __init__(self, inner: nnx.Module):
        self.inner = inner

    def __call__(self, x, context=None, *, rng=None, **kwargs):
        y = self.inner(jnp.asarray(x)[..., None], context=context, rng=rng, **kwargs)
        return y[..., 0]


@dataclass
class TransformerConditionerConfig(ConditionerConfigProtocol):
    """Attention conditioner: causal for autoregressive, full for coupling.

    Known limitation: the autoregressive variant trains and evaluates ``logpdf``
    fine, but cannot yet be sampled through :meth:`~probjax.nn.generative.base.
    GenerativeModel.as_dist` — exporting that sampler batches the layer's
    ``custom_inverse`` under symbolic shapes, and attention's sharded primitive
    requires concrete ones. Use :class:`MLPConditionerConfig`, or a coupling
    flow, when you need the exported sampler.
    """

    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    attn_size: int = 8
    widening_factor: int = 2

    def build_coupling(
        self, split_index, params_dim, bijector, *, context_features, rngs
    ):
        from probjax.nn.generative.nflows.coupling import CouplingTransformer

        return CouplingTransformer(
            split_index,
            params_dim,
            bijector,
            rngs,
            context_dim=context_features,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            attn_size=self.attn_size,
            widening_factor=self.widening_factor,
        )

    def build_autoregressive(
        self, in_out_features, params_dim, bijector, *, context_features, output_order, rngs
    ):
        del in_out_features  # sequence length is implicit in the input shape
        del output_order  # causal attention makes the output order implicit
        from probjax.nn.generative.nflows.autoregressive import AutoregressiveTransformer

        return _ScalarTokenConditioner(
            AutoregressiveTransformer(
                1,  # one scalar feature per token
                params_dim,
                _scalar_token_bijector(bijector),
                rngs,
                context_dim=context_features,
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                attn_size=self.attn_size,
                widening_factor=self.widening_factor,
            )
        )


# =============================================================================
# Mixing configs
# =============================================================================


@runtime_checkable
class MixingConfigProtocol(Protocol):
    """Builds the layer interleaved between transforms, or ``None`` for no mixing."""

    def build(self, input_dim: int, *, rngs: nnx.Rngs) -> Optional[nnx.Module]: ...


@dataclass
class NoMixingConfig(MixingConfigProtocol):
    """No mixing layer between transforms."""

    def build(self, input_dim, *, rngs):
        return None


@dataclass
class FlipMixingConfig(MixingConfigProtocol):
    """Reverse the feature axis (the conventional coupling-flow default)."""

    axis: int = -1

    def build(self, input_dim, *, rngs):
        from probjax.nn.generative.nflows.bijective import Flip

        return Flip(axis=self.axis, rngs=rngs)


@dataclass
class PermuteMixingConfig(MixingConfigProtocol):
    """Fixed permutation of the features, either reversed or drawn from ``rngs``."""

    mode: Literal["random", "reverse"] = "random"

    def build(self, input_dim, *, rngs):
        from probjax.nn.generative.nflows.bijective import Permute

        if self.mode == "reverse":
            permutation = jnp.arange(input_dim)[::-1]
        elif self.mode == "random":
            permutation = jax.random.permutation(rngs.params(), input_dim)
        else:
            raise ValueError(f"Unknown permutation mode {self.mode!r}.")
        return Permute(permutation, rngs=rngs)


@dataclass
class RotationMixingConfig(MixingConfigProtocol):
    """Orthogonal rotation; learnable via a skew-symmetric matrix exponential."""

    learnable: bool = False

    def build(self, input_dim, *, rngs):
        from probjax.nn.generative.nflows.bijective import Rotate

        return Rotate(input_dim, learnable=self.learnable, rngs=rngs)


# =============================================================================
# Flow configs
# =============================================================================


@dataclass
class NFlowConfig:
    """Shape of a normalizing flow plus its three configuration axes."""

    input_dim: int
    num_transforms: int = 8
    context_features: Optional[int] = None
    bijector: BijectorConfigProtocol = field(default_factory=AffineBijectorConfig)
    conditioner: ConditionerConfigProtocol = field(default_factory=MLPConditionerConfig)
    mixing: MixingConfigProtocol = field(default_factory=FlipMixingConfig)

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be positive; got {self.input_dim}.")
        if self.num_transforms < 1:
            raise ValueError(
                f"num_transforms must be positive; got {self.num_transforms}."
            )
        if self.context_features is not None and self.context_features < 1:
            raise ValueError("context_features must be positive when given.")


@dataclass
class CouplingNFlowConfig(NFlowConfig):
    """Coupling flow: ``x[:split_index]`` conditions the rest."""

    split_index: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.split_index is None:
            self.split_index = self.input_dim // 2
        if not 0 < self.split_index < self.input_dim:
            raise ValueError(
                f"split_index must be in (0, input_dim); got {self.split_index} "
                f"for input_dim={self.input_dim}."
            )


@dataclass
class AutoregressiveNFlowConfig(NFlowConfig):
    """Autoregressive flow: dimension ``i`` is conditioned on ``x[:i]``."""

    output_order: Literal["interleaved", "grouped"] = "grouped"


@dataclass
class ElementwiseNFlowConfig(NFlowConfig):
    """Elementwise flow: a free parameter table per dimension, no conditioner.

    Expressiveness comes from the mixing layers, so the default is a learnable
    rotation rather than a flip (this is the Gaussianization-flow shape).
    """

    mixing: MixingConfigProtocol = field(
        default_factory=lambda: RotationMixingConfig(learnable=True)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.context_features is not None:
            raise ValueError(
                "Elementwise flows have no conditioner and cannot use context."
            )
