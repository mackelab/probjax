from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import ArrayLike

from probjax.core import inverse, inverse_and_logabsdet
from probjax.nn.layers.bijective import Flip
from probjax.nn.nets.autoregressive import AutoregressiveMLP
from probjax.nn.nets.coupling import CouplingMLP
from probjax.nn.nets.simple import Sequential
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


def rational_quadratic_spline_and_logdets(
    params: ArrayLike,
    x: ArrayLike,
    range_min_x: float = -1.0,
    range_max_x: float = 1.0,
    range_min_y: float = -1.0,
    range_max_y: float = 1.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        ((range_max_x - min_bin_size) - range_min_x) + (range_min_x)
    )
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (range_max_y - min_bin_size) - range_min_y
    ) + range_min_y

    if not bounded:
        return _rational_quadratic_spline_and_logdets(x, x_pos, y_pos, knot_slopes)
    return _rational_quadratic_spline_and_logdets(
        x,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=range_min_x,
        x_max=range_max_x,
        y_min=range_min_y,
        y_max=range_max_y,
    )


def rational_quadratic_spline(
    params: ArrayLike,
    x: ArrayLike,
    range_min_x: float = -10.0,
    range_max_x: float = 10.0,
    range_min_y: float = -10.0,
    range_max_y: float = 10.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        (range_max_x - min_bin_size) - range_min_x
    ) + range_min_x
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (range_max_y - min_bin_size) - range_min_y
    ) + range_min_y

    if not bounded:
        return _rational_quadratic_spline(x, x_pos, y_pos, knot_slopes)
    return _rational_quadratic_spline(
        x,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=range_min_x,
        x_max=range_max_x,
        y_min=range_min_y,
        y_max=range_max_y,
    )


def rational_linear_spline(
    params,
    x,
    x_min=-10.0,
    x_max=10.0,
    y_min=-10.0,
    y_max=10.0,
    min_bin_size=1e-4,
    min_knot_slope=1e-4,
    bounded=False,
):
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        (x_max - min_bin_size) - x_min
    ) + x_min
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (y_max - min_bin_size) - y_min
    ) + y_min

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
    range_min_x: float = -10.0,
    range_max_x: float = 10.0,
    range_min_y: float = -10.0,
    range_max_y: float = 10.0,
    min_bin_size: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos = jnp.split(params, 2, axis=-1)
    x_pos = x_pos - jnp.mean(x_pos)
    y_pos = y_pos - jnp.mean(y_pos)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        ((range_max_x - min_bin_size) - range_min_x) + (range_min_x)
    )
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (range_max_y - min_bin_size) - range_min_y
    ) + range_min_y

    if not bounded:
        return _piecewise_affine_spline(x, x_pos, y_pos)
    return _piecewise_affine_spline(
        x,
        x_pos,
        y_pos,
        x_min=range_min_x,
        x_max=range_max_x,
        y_min=range_min_y,
        y_max=range_max_y,
    )


def monotone_hermite_cubic_spline(
    params,
    x,
    x_min=-10.0,
    x_max=10.0,
    y_min=-10.0,
    y_max=10.0,
    min_bin_size=1e-4,
    min_knot_slope=1e-4,
    bounded=False,
):
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        (x_max - min_bin_size) - x_min
    ) + x_min
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (y_max - min_bin_size) - y_min
    ) + y_min

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


def learnable_mixture_cdf(
    params: ArrayLike,
    y: ArrayLike,
    min_value=-10.0,
    max_value=10.0,
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
    del kwargs
    x = jnp.asarray(x)
    loc, scale = jnp.split(params, 2, axis=-1)
    scale = jnp.exp(scale)
    x_ks = (x[..., None] - loc) / scale
    cdf = jnp.mean(jax.nn.sigmoid(x_ks), -1)
    out = jax.scipy.stats.norm.ppf(cdf)
    return out


def _inv_and_logdet_learnable_mixture_cdf(params, x, **kwargs):
    del kwargs
    _f = jax.vmap(jax.value_and_grad(_inv_learnable_mixture_cdf, argnums=1))
    value, grad = _f(params, x)
    return value, jnp.log(jnp.abs(grad))


rational_quadratic_spline.inv = inverse(rational_quadratic_spline, invertible_arg=1)
rational_quadratic_spline.inv_and_logdet = inverse_and_logabsdet(
    rational_quadratic_spline,
    invertible_arg=1,
)

piecewise_affine_spline.inv = inverse(piecewise_affine_spline, invertible_arg=1)
piecewise_affine_spline.inv_and_logdet = inverse_and_logabsdet(
    piecewise_affine_spline,
    invertible_arg=1,
)

rational_linear_spline.inv = inverse(rational_linear_spline, invertible_arg=1)
rational_linear_spline.inv_and_logdet = inverse_and_logabsdet(
    rational_linear_spline,
    invertible_arg=1,
)

monotone_hermite_cubic_spline.inv = inverse(
    monotone_hermite_cubic_spline,
    invertible_arg=1,
)
monotone_hermite_cubic_spline.inv_and_logdet = inverse_and_logabsdet(
    monotone_hermite_cubic_spline,
    invertible_arg=1,
)


class NormalizingFlow(nnx.Module):
    def __init__(
        self,
        base_dist,
        transformation: Callable[..., Any],
        name: Optional[str] = None,
        *,
        sharding: jax.sharding.Mesh | None = None,
    ):
        self.base_dist = base_dist
        self.transformation = transformation
        self._mesh = sharding
        super().__init__()

    def transform(self, x, *, rng: jax.Array | None = None):
        return self.transformation(x, rng=rng)

    def __call__(self, x, *, rng: jax.Array | None = None):
        return self.transform(x, rng=rng)

    def sample(self, rng, shape=()):
        """Sample from the flow distribution."""
        return transformed.rvs(
            rng, shape=shape, base_dist=self.base_dist, bijector=self.transformation
        )

    def logpdf(self, x):
        return transformed.logpdf(
            x, base_dist=self.base_dist, bijector=self.transformation
        )


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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = input_dim - split_dim
        coupling_net = partial(
            coupling_class, split_dim, params_dim, additive_bijector, sharding=sharding
        )

        # Build the flow transformation
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)


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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = (input_dim - split_dim) * 2
        coupling_net = partial(
            coupling_class, split_dim, params_dim, affine_bijector, sharding=sharding
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)


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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        dims = input_dim - split_dim
        params_dim = (num_bins * 3) * dims
        spline = partial(
            rational_quadratic_spline,
            range_min_x=-10.0,
            range_max_x=10.0,
            range_min_y=-10.0,
            range_max_y=10.0,
        )

        def spline_fn(params, x):
            params = jnp.reshape(params, (dims, num_bins * 3))
            return jax.vmap(spline)(params, x)

        coupling_net = partial(
            coupling_class, split_dim, params_dim, spline_fn, sharding=sharding
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)


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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 1
        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            additive_bijector,
            sharding=sharding,
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [autoregressive(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)


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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 2
        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            affine_bijector,
            sharding=sharding,
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [autoregressive(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)


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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 3 * num_bins * input_dim
        spline = partial(
            rational_quadratic_spline,
            range_min_x=-10.0,
            range_max_x=10.0,
            range_min_y=-10.0,
            range_max_y=10.0,
        )

        def spline_fn(params, x):
            return spline(params, x)

        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            spline_fn,
            sharding=sharding,
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [autoregressive(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)
