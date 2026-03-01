import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

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
    rational_quadratic_spline_and_logdets as _rational_quadratic_spline_and_logdets,
)
from probjax.utils.solver import root_scalar


def _normalize_knot_slopes(
    unnormalized_knot_slopes: Array, min_knot_slope: float
) -> Array:
    """Make knot slopes be no less than `min_knot_slope`."""
    if min_knot_slope >= 1.0:
        raise ValueError(
            f"The minimum knot slope must be less than 1; got {min_knot_slope}."
        )
    min_knot_slope = jnp.array(min_knot_slope, dtype=unnormalized_knot_slopes.dtype)
    offset = jnp.log(jnp.exp(1.0 - min_knot_slope) - 1.0)
    return jax.nn.softplus(unnormalized_knot_slopes + offset) + min_knot_slope


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
    x = jnp.asarray(x)
    loc, scale = jnp.split(params, 2, axis=-1)
    scale = jnp.exp(scale)
    x_ks = (x[..., None] - loc) / scale
    cdf = jnp.mean(jax.nn.sigmoid(x_ks), -1)
    out = jax.scipy.stats.norm.ppf(cdf)
    return out


def _inv_and_logdet_learnable_mixture_cdf(params, x, **kwargs):
    _f = jax.vmap(jax.value_and_grad(_inv_learnable_mixture_cdf, argnums=1))
    value, grad = _f(params, x)
    return value, jnp.log(jnp.abs(grad))


def affine_bijector(params: ArrayLike, x: ArrayLike, min_scale=1e-3, **kwargs):
    x = jnp.asarray(x)
    loc, scale = jnp.split(params, 2, axis=-1)
    loc, scale = loc.reshape(x.shape), scale.reshape(x.shape)
    scale = jnp.exp(scale) + min_scale

    return loc + scale * x


def additive_bijector(params: ArrayLike, x: ArrayLike, **kwargs):
    x = jnp.asarray(x)
    params = jnp.asarray(params)
    params = params.reshape(x.shape)
    return x + params
