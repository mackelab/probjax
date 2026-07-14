from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import ArrayLike

from probjax.nn.generative.flows.autoregressive import AutoregressiveMLP
from probjax.nn.generative.flows.bijective import ElementwiseMonotone, Flip, Rotate
from probjax.nn.generative.flows.coupling import CouplingMLP
from probjax.nn.nets.simple import Sequential

from probjax.stats.base import DistributionAPI, rv_frozen
from probjax.stats.fit import FitMixin
from probjax.stats.bijective import additive_bijector, affine_bijector
from probjax.stats.bijective.monotone import (
    bernstein_bijector,
    deep_sigmoid_bijector,
    mixture_cdf_bijector,
    sos_polynomial_bijector,
    unconstrained_monotone_bijector,
)
from probjax.stats.bijective.monotone_hermite_cubic import (
    monotone_hermite_cubic_spline as _monotone_hermite_cubic_spline,
)
from probjax.stats.bijective.piecewise_affine import (
    piecewise_affine_spline as _piecewise_affine_spline,
)
from probjax.stats.bijective.rational_linear import (
    rational_linear_spline as _rational_linear_spline,
)
from probjax.stats.bijective.rational_quadratic import (
    rational_quadratic_spline as _rational_quadratic_spline,
)
from probjax.stats.bijective.rational_quadratic import (
    rational_quadratic_spline_and_logdets as _rational_quadratic_spline_and_logdets,
)
from probjax.stats.continuous import norm
from probjax.stats.indep import indep
from probjax.stats.transformed import transformed


# ---------------------------------------------------------------------------
# Shared parameter normalization utilities
# ---------------------------------------------------------------------------


def _normalize_knot_slopes(
    unnormalized_knot_slopes: Array, min_knot_slope: float
) -> Array:
    if min_knot_slope >= 1.0:
        raise ValueError(
            f"The minimum knot slope must be less than 1; got {min_knot_slope}."
        )
    min_slope = jnp.array(min_knot_slope, dtype=unnormalized_knot_slopes.dtype)
    offset = jnp.log(jnp.exp(1.0 - min_slope) - 1.0)
    return jax.nn.softplus(unnormalized_knot_slopes + offset) + min_slope


def _normalize_bin_positions(
    raw: Array,
    lo: float,
    hi: float,
    min_bin_size: float,
) -> Array:
    """Convert raw unconstrained values into monotonically increasing bin positions."""
    return (jnp.cumsum(jax.nn.softmax(raw), -1) + min_bin_size) * (
        (hi - min_bin_size) - lo
    ) + lo


def _normalize_spline_knots(
    params: ArrayLike,
    *,
    has_slopes: bool,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    min_bin_size: float,
    min_knot_slope: float = 1e-4,
    center_positions: bool = False,
):
    """Split raw params and normalise into knot positions (and optionally slopes).

    Returns ``(x_pos, y_pos)`` when ``has_slopes=False``, or
    ``(x_pos, y_pos, knot_slopes)`` when ``has_slopes=True``.
    """
    n_parts = 3 if has_slopes else 2
    parts = jnp.split(params, n_parts, axis=-1)

    raw_x, raw_y = parts[0], parts[1]

    if center_positions:
        raw_x = raw_x - jnp.mean(raw_x)
        raw_y = raw_y - jnp.mean(raw_y)

    x_pos = _normalize_bin_positions(raw_x, x_min, x_max, min_bin_size)
    y_pos = _normalize_bin_positions(raw_y, y_min, y_max, min_bin_size)

    if has_slopes:
        knot_slopes = _normalize_knot_slopes(parts[2], min_knot_slope)
        return x_pos, y_pos, knot_slopes

    return x_pos, y_pos


# ---------------------------------------------------------------------------
# Parametrised spline wrappers
#
# These take a single *params* vector (output of a neural network) and an
# input *x*, normalise the params into knot positions / slopes, then delegate
# to the corresponding function in ``probjax.stats.bijective``.
#
# Kwarg names match the stats layer: x_min, x_max, y_min, y_max.
# ---------------------------------------------------------------------------


def rational_quadratic_spline_and_logdets(
    params: ArrayLike,
    x: ArrayLike,
    x_min: float = -10.0,
    x_max: float = 10.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos, knot_slopes = _normalize_spline_knots(
        params,
        has_slopes=True,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        min_bin_size=min_bin_size,
        min_knot_slope=min_knot_slope,
    )
    if not bounded:
        return _rational_quadratic_spline_and_logdets(x, x_pos, y_pos, knot_slopes)
    return _rational_quadratic_spline_and_logdets(
        x,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def rational_quadratic_spline(
    params: ArrayLike,
    x: ArrayLike,
    x_min: float = -10.0,
    x_max: float = 10.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos, knot_slopes = _normalize_spline_knots(
        params,
        has_slopes=True,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        min_bin_size=min_bin_size,
        min_knot_slope=min_knot_slope,
    )
    if not bounded:
        return _rational_quadratic_spline(x, x_pos, y_pos, knot_slopes)
    return _rational_quadratic_spline(
        x,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def rational_linear_spline(
    params: ArrayLike,
    x: ArrayLike,
    x_min: float = -10.0,
    x_max: float = 10.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos, knot_slopes = _normalize_spline_knots(
        params,
        has_slopes=True,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        min_bin_size=min_bin_size,
        min_knot_slope=min_knot_slope,
    )
    if not bounded:
        return _rational_linear_spline(x, x_pos, y_pos, knot_slopes)
    return _rational_linear_spline(
        x,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def piecewise_affine_spline(
    params: ArrayLike,
    x: ArrayLike,
    x_min: float = -10.0,
    x_max: float = 10.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
    min_bin_size: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos = _normalize_spline_knots(
        params,
        has_slopes=False,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        min_bin_size=min_bin_size,
        center_positions=True,
    )
    if not bounded:
        return _piecewise_affine_spline(x, x_pos, y_pos)
    return _piecewise_affine_spline(
        x,
        x_pos,
        y_pos,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def monotone_hermite_cubic_spline(
    params: ArrayLike,
    x: ArrayLike,
    x_min: float = -10.0,
    x_max: float = 10.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos, knot_slopes = _normalize_spline_knots(
        params,
        has_slopes=True,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        min_bin_size=min_bin_size,
        min_knot_slope=min_knot_slope,
    )
    if not bounded:
        return _monotone_hermite_cubic_spline(x, x_pos, y_pos, knot_slopes)
    return _monotone_hermite_cubic_spline(
        x,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


# ---------------------------------------------------------------------------
# NormalizingFlow base class
# ---------------------------------------------------------------------------


class NormalizingFlow(nnx.Module, DistributionAPI, FitMixin):
    def __init__(
        self,
        base_dist,
        transformation: Callable[..., Any],
        name: Optional[str] = None,
    ):
        self.base_dist = base_dist
        self.transformation = transformation
        self.name = name
        super().__init__()

    # -- scipy-like stats API via transformed distribution --

    @property
    def dist(self):
        """Frozen ``transformed`` distribution for full scipy-like API access."""
        return transformed(base_dist=self.base_dist, bijector=self.transformation)

    def conditional_dist(self, context):
        """Frozen transformed distribution conditioned on context."""

        def _bijector(x):
            return self.transformation(x, context)

        return transformed(base_dist=self.base_dist, bijector=_bijector)

    def _dist_for_context(self, context=None):
        return self.dist if context is None else self.conditional_dist(context)

    @property
    def batch_shape(self):
        return self.dist.batch_shape

    @property
    def event_shape(self):
        return self.dist.event_shape

    def transform(self, x, context=None, *, rng: jax.Array | None = None):
        if context is None:
            return self.transformation(x, rng=rng)
        return self.transformation(x, context, rng=rng)

    def __call__(self, x, context=None, *, rng: jax.Array | None = None):
        return self.transform(x, context=context, rng=rng)

    def sample(self, rng, shape=(), context=None):
        """Sample from the flow distribution."""
        return self.rvs(rng, shape=shape, context=context)

    def rvs(self, rng, shape=(), name: Optional[str] = None, context=None, **kwargs):
        return self._dist_for_context(context).rvs(
            rng, shape=shape, name=name, **kwargs
        )

    def logpdf(self, x, context=None):
        return self._dist_for_context(context).logpdf(x)

    def pdf(self, x, context=None):
        return self._dist_for_context(context).pdf(x)

    def cdf(self, x, context=None):
        return self._dist_for_context(context).cdf(x)

    def ppf(self, q, context=None):
        return self._dist_for_context(context).ppf(q)

    def logcdf(self, x, context=None):
        return self._dist_for_context(context).logcdf(x)

    def sf(self, x, context=None):
        return self._dist_for_context(context).sf(x)

    def logsf(self, x, context=None):
        return self._dist_for_context(context).logsf(x)

    def isf(self, q, context=None):
        return self._dist_for_context(context).isf(q)

    def mean(self):
        return self.dist.mean()

    def mode(self):
        return self.dist.mode()

    def var(self):
        return self.dist.var()

    def std(self):
        return self.dist.std()

    def entropy(self):
        return self.dist.entropy()

    def median(self):
        return self.dist.median()

    def interval(self, confidence=None, context=None):
        return self._dist_for_context(context).interval(confidence)

    def moment(self, order: Optional[int] = None, context=None):
        return self._dist_for_context(context).moment(order)

    def stats(self, moments: str = "mv", context=None):
        return self._dist_for_context(context).stats(moments=moments)

    def support(self):
        return self.dist.support()

    def loss(self, rng, data, *args, context=None, **kwargs):
        """Negative mean log-likelihood training loss.

        With ``context``, each data row is scored against its own context row
        (per-pair conditional likelihood).
        """
        del rng, args, kwargs
        if context is None:
            return -jnp.mean(self.logpdf(data))
        pair_logpdf = jax.vmap(lambda x, c: self.logpdf(x, context=c))
        return -jnp.mean(pair_logpdf(data, context))

    def as_distribution(self, event_shape=None, *, context=None):
        """View this flow as a :class:`~probjax.stats.base.DistributionAPI`.

        The flow already is one, so this returns ``self`` (or the frozen
        conditional distribution when ``context`` is given). ``event_shape``
        is accepted for protocol compatibility; it is fixed by the flow.
        """
        del event_shape
        if context is None:
            return self
        return self.conditional_dist(context)

    # -- Helper for building the standard normal base distribution --

    @staticmethod
    def _standard_normal_base(input_dim: int):
        """Create an independent standard normal base distribution."""
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        return indep(norm(mu0, std0))


# ---------------------------------------------------------------------------
# Helper to build a sequence of transforms with interleaved mixing layers
# ---------------------------------------------------------------------------


def _build_transform_sequence(
    layer_fn,
    num_transforms: int,
    rngs,
    mixing_class,
    last_transform,
):
    """Build a ``Sequential`` of alternating transform + mixing layers."""
    transforms = []
    for i in range(num_transforms):
        transforms.append(layer_fn(rngs=rngs))
        if i < num_transforms - 1:
            transforms.append(mixing_class(rngs=rngs))
    if last_transform is not None:
        transforms.append(last_transform)
    return Sequential(*transforms)


# ---------------------------------------------------------------------------
# Coupling-based flows
# ---------------------------------------------------------------------------


class AdditiveCouplingFlow(NormalizingFlow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        context_features: Optional[int] = None,
        last_transform: Optional[Callable] = None,
        coupling_class: nnx.Module = CouplingMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = input_dim - split_dim
        coupling_net = partial(
            coupling_class,
            split_dim,
            params_dim,
            additive_bijector,
            context_dim=context_features,
        )

        transform = _build_transform_sequence(
            coupling_net,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name)


class AffineCouplingFlow(NormalizingFlow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        context_features: Optional[int] = None,
        last_transform: Optional[Callable] = None,
        coupling_class: nnx.Module = CouplingMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = (input_dim - split_dim) * 2
        coupling_net = partial(
            coupling_class,
            split_dim,
            params_dim,
            affine_bijector,
            context_dim=context_features,
        )

        transform = _build_transform_sequence(
            coupling_net,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name)


class SplineCouplingFlow(NormalizingFlow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        context_features: Optional[int] = None,
        num_bins: int = 10,
        last_transform: Optional[Callable] = None,
        coupling_class: nnx.Module = CouplingMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        dims = input_dim - split_dim
        params_dim = (num_bins * 3) * dims
        spline = partial(
            rational_quadratic_spline,
            x_min=-10.0,
            x_max=10.0,
            y_min=-10.0,
            y_max=10.0,
        )

        # Apply spline independently per dimension, handling arbitrary batch shapes.
        # The core spline takes scalar x and 1D params of size num_bins*3.
        # jnp.vectorize broadcasts over any leading batch dimensions.
        _vectorized_spline = jnp.vectorize(spline, signature=f"({num_bins * 3}),()->()")

        def spline_fn(params, x):
            # params: (..., dims * num_bins * 3) -> (..., dims, num_bins * 3)
            batch_shape = params.shape[:-1]
            params = jnp.reshape(params, batch_shape + (dims, num_bins * 3))
            return _vectorized_spline(params, x)

        coupling_net = partial(
            coupling_class,
            split_dim,
            params_dim,
            spline_fn,
            context_dim=context_features,
        )

        transform = _build_transform_sequence(
            coupling_net,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name)


# ---------------------------------------------------------------------------
# Autoregressive flows
# ---------------------------------------------------------------------------


class AdditiveAutoregressiveFlow(NormalizingFlow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        context_features: Optional[int] = None,
        last_transform: Optional[Callable] = None,
        autoregressive_class: nnx.Module = AutoregressiveMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 1
        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            additive_bijector,
            context_features=context_features,
        )

        transform = _build_transform_sequence(
            autoregressive,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name)


class AffineAutoregressiveFlow(NormalizingFlow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        context_features: Optional[int] = None,
        last_transform: Optional[Callable] = None,
        autoregressive_class: nnx.Module = AutoregressiveMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 2
        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            affine_bijector,
            context_features=context_features,
        )

        transform = _build_transform_sequence(
            autoregressive,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name)


class SplineAutoregressiveFlow(NormalizingFlow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        context_features: Optional[int] = None,
        num_bins: int = 10,
        last_transform: Optional[Callable] = None,
        autoregressive_class: nnx.Module = AutoregressiveMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 3 * num_bins * input_dim
        spline = partial(
            rational_quadratic_spline,
            x_min=-10.0,
            x_max=10.0,
            y_min=-10.0,
            y_max=10.0,
        )

        def spline_fn(params, x):
            return spline(params, x)

        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            spline_fn,
            context_features=context_features,
        )

        transform = _build_transform_sequence(
            autoregressive,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name)


class NeuralSplineFlow(SplineAutoregressiveFlow):
    """Neural Spline Flow (NSF) style wrapper."""


class _MonotoneAutoregressiveFlow(NormalizingFlow):
    """Shared constructor for autoregressive flows with monotone-net bijectors.

    The bijector's analytic direction is data -> base, so ``logpdf`` (and
    training) is closed-form and differentiable; sampling solves the monotone
    map elementwise by bisection.
    """

    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        bijector,
        bijector_dim: int,
        *,
        context_features: Optional[int] = None,
        last_transform: Optional[Callable] = None,
        autoregressive_class: nnx.Module = AutoregressiveMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        autoregressive = partial(
            autoregressive_class,
            input_dim,
            bijector_dim,
            bijector,
            context_features=context_features,
        )
        transform = _build_transform_sequence(
            autoregressive,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name)


class NeuralAutoregressiveFlow(_MonotoneAutoregressiveFlow):
    """Neural Autoregressive Flow with deep sigmoidal transforms (Huang et al., 2018)."""

    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        num_components: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim,
            num_transforms,
            rngs,
            deep_sigmoid_bijector,
            3 * num_components,
            **kwargs,
        )


class UnconstrainedNeuralAutoregressiveFlow(_MonotoneAutoregressiveFlow):
    """UMNN-style flow: monotone neural integrand integrated by quadrature
    (Wehenkel & Louppe, 2019)."""

    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        num_hidden: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim,
            num_transforms,
            rngs,
            unconstrained_monotone_bijector,
            3 * num_hidden + 2,
            **kwargs,
        )


class SumOfSquaresPolynomialFlow(_MonotoneAutoregressiveFlow):
    """Sum-of-squares polynomial flow (Jaini et al., 2019)."""

    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        num_polys: int = 2,
        **kwargs,
    ) -> None:
        from probjax.stats.bijective.monotone import _SOS_DEGREE

        super().__init__(
            input_dim,
            num_transforms,
            rngs,
            sos_polynomial_bijector,
            num_polys * (_SOS_DEGREE + 1) + 1,
            **kwargs,
        )


class BernsteinPolynomialFlow(_MonotoneAutoregressiveFlow):
    """Monotone Bernstein polynomial flow with identity tails (Sick et al., 2021)."""

    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        degree: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim,
            num_transforms,
            rngs,
            bernstein_bijector,
            degree,
            **kwargs,
        )


def _gf_params_init(key, shape, dtype=None):
    """Zero-init mixture params except the location block, spread randomly."""
    dtype = dtype or jnp.float32
    num_components = shape[-1] // 3
    params = jnp.zeros(shape, dtype)
    locs = jax.random.normal(key, shape[:-1] + (num_components,), dtype)
    return params.at[..., num_components : 2 * num_components].set(locs)


class GaussianizationFlow(NormalizingFlow):
    """Gaussianization Flow: learnable logistic-mixture-CDF layers alternated
    with learnable rotations (Meng et al., 2020). Unconditional only."""

    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        num_components: int = 8,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        layer_fn = partial(
            ElementwiseMonotone,
            input_dim,
            3 * num_components,
            mixture_cdf_bijector,
            params_init=_gf_params_init,
        )
        mixing = partial(Rotate, input_dim, learnable=True)
        transform = _build_transform_sequence(
            layer_fn,
            num_transforms,
            rngs,
            mixing,
            None,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name)


rv_frozen.register(NormalizingFlow)


# Lowercase aliases mirroring common flow naming conventions.
nice = AdditiveCouplingFlow
realnvp = AffineCouplingFlow
maf = AffineAutoregressiveFlow
nsf = SplineAutoregressiveFlow
naf = NeuralAutoregressiveFlow
unaf = UnconstrainedNeuralAutoregressiveFlow
sospf = SumOfSquaresPolynomialFlow
bpf = BernsteinPolynomialFlow
gf = GaussianizationFlow


__all__ = [
    "NormalizingFlow",
    "AdditiveCouplingFlow",
    "AffineCouplingFlow",
    "SplineCouplingFlow",
    "AdditiveAutoregressiveFlow",
    "AffineAutoregressiveFlow",
    "SplineAutoregressiveFlow",
    "NeuralSplineFlow",
    "NeuralAutoregressiveFlow",
    "UnconstrainedNeuralAutoregressiveFlow",
    "SumOfSquaresPolynomialFlow",
    "BernsteinPolynomialFlow",
    "GaussianizationFlow",
    "nice",
    "realnvp",
    "maf",
    "nsf",
    "naf",
    "unaf",
    "sospf",
    "bpf",
    "gf",
]
