from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import ArrayLike

from probjax.nn.layers.bijective import Flip
from probjax.nn.nets.autoregressive import AutoregressiveMLP
from probjax.nn.nets.coupling import CouplingMLP
from probjax.nn.nets.simple import Sequential
from probjax.nn.sharding import ShardingCfg, resolve_sharding_mesh
from probjax.stats.bijective import additive_bijector, affine_bijector
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
from probjax.stats.independent import independent
from probjax.stats.transformed import transformed
from probjax.utils.solver import root_scalar


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
# Learnable mixture CDF bijector
# ---------------------------------------------------------------------------


def learnable_mixture_cdf(
    params: ArrayLike,
    y: ArrayLike,
    min_value: float = -10.0,
    max_value: float = 10.0,
    **kwargs,
):
    def f(x):
        return _inv_learnable_mixture_cdf(params, x) - y

    x = root_scalar(
        f,
        bracket=(min_value * jnp.ones_like(y), max_value * jnp.ones_like(y)),
        **kwargs,
    )
    return x


def _inv_learnable_mixture_cdf(
    params: ArrayLike,
    x: ArrayLike,
    **kwargs,
):
    """Inverse of the learnable mixture CDF (forward pass of the bijector).

    Computes a mixture of logistic CDFs, then maps through the normal PPF.
    """
    del kwargs
    x = jnp.asarray(x)
    loc, scale = jnp.split(params, 2, axis=-1)
    scale = jnp.exp(scale)
    x_ks = (x[..., None] - loc) / scale
    # Logistic CDF mixture -> normal quantile
    cdf = jnp.mean(jax.nn.sigmoid(x_ks), -1)
    return norm.ppf(cdf)


def _inv_and_logdet_learnable_mixture_cdf(params, x, **kwargs):
    del kwargs
    _f = jax.vmap(jax.value_and_grad(_inv_learnable_mixture_cdf, argnums=1))
    value, grad = _f(params, x)
    return value, jnp.log(jnp.abs(grad))


# ---------------------------------------------------------------------------
# NormalizingFlow base class
# ---------------------------------------------------------------------------


class NormalizingFlow(nnx.Module):
    def __init__(
        self,
        base_dist,
        transformation: Callable[..., Any],
        name: Optional[str] = None,
        *,
        sharding_cfg: ShardingCfg | None = None,
    ):
        self.base_dist = base_dist
        self.transformation = transformation
        self._mesh = resolve_sharding_mesh(sharding_cfg)
        super().__init__()

    # -- scipy-like stats API via transformed distribution --

    @property
    def dist(self):
        """Frozen ``transformed`` distribution for full scipy-like API access."""
        return transformed(base_dist=self.base_dist, bijector=self.transformation)

    def transform(self, x, *, rng: jax.Array | None = None):
        return self.transformation(x, rng=rng)

    def __call__(self, x, *, rng: jax.Array | None = None):
        return self.transform(x, rng=rng)

    def sample(self, rng, shape=()):
        """Sample from the flow distribution."""
        return self.dist.rvs(rng, shape=shape)

    def logpdf(self, x):
        return self.dist.logpdf(x)

    # -- Helper for building the standard normal base distribution --

    @staticmethod
    def _standard_normal_base(input_dim: int):
        """Create an independent standard normal base distribution."""
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        return independent(norm(mu0, std0))


# ---------------------------------------------------------------------------
# Helper to build a sequence of transforms with interleaved mixing layers
# ---------------------------------------------------------------------------


def _build_transform_sequence(
    layer_fn,
    num_transforms: int,
    rngs,
    mixing_class,
    last_transform,
    sharding_cfg,
):
    """Build a ``Sequential`` of alternating transform + mixing layers."""
    transforms = []
    for i in range(num_transforms):
        transforms.append(layer_fn(rngs=rngs))
        if i < num_transforms - 1:
            transforms.append(mixing_class(rngs=rngs, sharding_cfg=sharding_cfg))
    if last_transform is not None:
        transforms.append(last_transform)
    return Sequential(*transforms, sharding_cfg=sharding_cfg)


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
        last_transform: Optional[Callable] = None,
        coupling_class: nnx.Module = CouplingMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
        sharding_cfg: ShardingCfg | None = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = input_dim - split_dim
        coupling_net = partial(
            coupling_class,
            split_dim,
            params_dim,
            additive_bijector,
            sharding_cfg=sharding_cfg,
        )

        transform = _build_transform_sequence(
            coupling_net,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
            sharding_cfg,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name, sharding_cfg=sharding_cfg)


class AffineCouplingFlow(NormalizingFlow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        last_transform: Optional[Callable] = None,
        coupling_class: nnx.Module = CouplingMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
        sharding_cfg: ShardingCfg | None = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = (input_dim - split_dim) * 2
        coupling_net = partial(
            coupling_class,
            split_dim,
            params_dim,
            affine_bijector,
            sharding_cfg=sharding_cfg,
        )

        transform = _build_transform_sequence(
            coupling_net,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
            sharding_cfg,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name, sharding_cfg=sharding_cfg)


class SplineCouplingFlow(NormalizingFlow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        num_bins: int = 10,
        last_transform: Optional[Callable] = None,
        coupling_class: nnx.Module = CouplingMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
        sharding_cfg: ShardingCfg | None = None,
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
            sharding_cfg=sharding_cfg,
        )

        transform = _build_transform_sequence(
            coupling_net,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
            sharding_cfg,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name, sharding_cfg=sharding_cfg)


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
        last_transform: Optional[Callable] = None,
        autoregressive_class: nnx.Module = AutoregressiveMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
        sharding_cfg: ShardingCfg | None = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 1
        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            additive_bijector,
            sharding_cfg=sharding_cfg,
        )

        transform = _build_transform_sequence(
            autoregressive,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
            sharding_cfg,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name, sharding_cfg=sharding_cfg)


class AffineAutoregressiveFlow(NormalizingFlow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        last_transform: Optional[Callable] = None,
        autoregressive_class: nnx.Module = AutoregressiveMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
        sharding_cfg: ShardingCfg | None = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 2
        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            affine_bijector,
            sharding_cfg=sharding_cfg,
        )

        transform = _build_transform_sequence(
            autoregressive,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
            sharding_cfg,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name, sharding_cfg=sharding_cfg)


class SplineAutoregressiveFlow(NormalizingFlow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        num_bins: int = 10,
        last_transform: Optional[Callable] = None,
        autoregressive_class: nnx.Module = AutoregressiveMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
        sharding_cfg: ShardingCfg | None = None,
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
            sharding_cfg=sharding_cfg,
        )

        transform = _build_transform_sequence(
            autoregressive,
            num_transforms,
            rngs,
            mixing_class,
            last_transform,
            sharding_cfg,
        )
        q0 = self._standard_normal_base(input_dim)
        super().__init__(q0, transform, name=name, sharding_cfg=sharding_cfg)
