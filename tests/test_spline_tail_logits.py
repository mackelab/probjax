"""Tail-logit damping preserves learnability and spline density accounting."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probjax.core import inverse_and_logabsdet
from probjax.nn import (
    MonotoneHermiteCubicSplineConfig,
    RationalLinearSplineConfig,
    RationalQuadraticSplineConfig,
    SplineCouplingFlow,
    nsf,
)


@pytest.mark.parametrize(
    "config_cls",
    [
        RationalQuadraticSplineConfig,
        RationalLinearSplineConfig,
        MonotoneHermiteCubicSplineConfig,
    ],
)
def test_only_boundary_slopes_are_damped_and_remain_learnable(config_cls):
    original = config_cls(num_bins=4, tail_logit_scale=1.0)
    damped = config_cls(num_bins=4)
    params = jnp.linspace(-0.5, 0.5, original.params_dim())
    x_pos, y_pos, slopes = original.unpack(params)
    damped_x, damped_y, damped_slopes = damped.unpack(params)
    np.testing.assert_array_equal(damped_x, x_pos)
    np.testing.assert_array_equal(damped_y, y_pos)
    np.testing.assert_array_equal(damped_slopes[1:-1], slopes[1:-1])
    endpoints = jnp.array([0, -1])
    assert jnp.all(
        jnp.abs(jnp.log(damped_slopes[endpoints])) < jnp.abs(jnp.log(slopes[endpoints]))
    )

    zeros = jnp.zeros(original.params_dim())
    original_derivative = jax.jacfwd(lambda p: original.unpack(p)[2])(zeros)
    damped_derivative = jax.jacfwd(lambda p: damped.unpack(p)[2])(zeros)
    np.testing.assert_allclose(
        damped_derivative[endpoints],
        0.1 * original_derivative[endpoints],
    )
    assert jnp.linalg.norm(damped_derivative[endpoints]) > 0


def test_wider_damped_spline_starts_at_identity_and_inverts_tails():
    config = RationalQuadraticSplineConfig(
        num_bins=8, x_min=-6, x_max=6, y_min=-6, y_max=6, tail_logit_scale=0.1
    )
    x = jnp.array([-12.0, -7.0, 0.2, 7.0, 12.0])
    zeros = jnp.zeros((len(x), config.params_dim()))
    np.testing.assert_allclose(config(zeros, x), x, atol=1e-6)
    params = 0.3 * jax.random.normal(jax.random.key(0), zeros.shape)
    y = config(params, x)
    recovered, logdet = inverse_and_logabsdet(config, invertible_arg=1)(params, y)
    np.testing.assert_allclose(recovered, x, atol=2e-6)
    derivatives = jax.vmap(jax.grad(lambda p, value: config(p, value), argnums=1))(
        params, x
    )
    np.testing.assert_allclose(logdet, -jnp.log(derivatives).sum(), atol=2e-6)


@pytest.mark.parametrize("scale", [0, -0.1, float("nan"), float("inf")])
def test_tail_logit_scale_requires_positive_finite_value(scale):
    with pytest.raises(ValueError, match="tail_logit_scale"):
        RationalQuadraticSplineConfig(tail_logit_scale=scale)


@pytest.mark.parametrize("flow_cls", [nsf, SplineCouplingFlow])
def test_flow_presets_default_to_damped_tails_and_allow_override(flow_cls):
    flow = flow_cls(4, 2, rngs=nnx.Rngs(0))
    config = flow.config.bijector
    assert config.tail_logit_scale == 0.1
    assert (config.x_min, config.x_max, config.y_min, config.y_max) == (-5, 5, -5, 5)
    assert type(flow.transformation.layers[1]).__name__ == "Flip"
    x = jnp.array([-12.0, -0.5, 0.3, 12.0])
    y = flow.transformation(x)
    recovered, logdet = inverse_and_logabsdet(flow.transformation)(y)
    np.testing.assert_allclose(recovered, x, atol=1e-6)
    np.testing.assert_allclose(logdet, 0, atol=1e-6)

    undamped = flow_cls(4, 2, rngs=nnx.Rngs(0), tail_logit_scale=1.0)
    assert undamped.config.bijector.tail_logit_scale == 1.0
