import jax
import pytest

from probjax.nn.loss_fn import (
    build_denoising_loss,
    build_denoising_score_matching_loss,
    build_flow_matching_loss,
    build_score_matching_loss,
    build_sliced_score_matching_loss,
    build_target_score_matching_loss,
)
from probjax.nn.nets.flow_matching_configs import (
    AutodiffInterpolationSchedule,
    LinearInterpolationSchedule,
)


def model_fn(x):
    return x


@pytest.mark.parametrize(
    "builder, kwargs",
    [
        (build_denoising_loss, {"std": 1.0, "scale": 1.0}),
        (build_denoising_score_matching_loss, {"std": 1.0}),
        (build_score_matching_loss, {}),
        (build_sliced_score_matching_loss, {"num_slices": 1}),
        (build_target_score_matching_loss, {"score_fn": lambda x: x, "std": 1.0}),
    ],
)
def test_loss_fn(builder, kwargs):
    loss_fn = builder(model_fn, **kwargs)
    assert callable(loss_fn), "loss_fn is not callable"

    batch_vectors = jax.random.normal(jax.random.key(0), (100, 10))
    loss = loss_fn(batch_vectors, rng=jax.random.key(0))
    assert loss.ndim == 0, "loss is not a scalar"
    assert jax.numpy.isfinite(loss), "loss is not finite"


def test_flow_matching_schedule_zero_loss():
    schedule = LinearInterpolationSchedule()

    def model_fn(t, xt, x0, x1):
        return x1 - x0

    loss_fn = build_flow_matching_loss(model_fn, schedule=schedule)
    x0 = jax.numpy.ones((4, 3))
    x1 = 2.0 * jax.numpy.ones((4, 3))
    t = 0.3 * jax.numpy.ones((4, 1))
    loss = loss_fn(t, x0, x1, x0, x1, rng=jax.random.key(0))

    assert jax.numpy.allclose(loss, 0.0, atol=1e-5), "loss should be ~0"


def test_flow_matching_schedule_noise_zero_loss():
    schedule = LinearInterpolationSchedule(
        noise_fn=lambda t, x0, x1: 0.1 * jax.numpy.ones_like(x0),
        noise_velocity_fn=lambda t, x0, x1: jax.numpy.zeros_like(x0),
    )

    def model_fn(t, xt, x0, x1):
        return x1 - x0

    loss_fn = build_flow_matching_loss(model_fn, schedule=schedule)
    x0 = jax.numpy.ones((2, 5))
    x1 = 3.0 * jax.numpy.ones((2, 5))
    t = 0.6 * jax.numpy.ones((2, 1))
    loss = loss_fn(t, x0, x1, x0, x1, rng=jax.random.key(1))

    assert jax.numpy.allclose(loss, 0.0, atol=1e-5), "loss should be ~0"


def test_flow_matching_autodiff_schedule():
    def interp_fn(t, x0, x1):
        tt = t**2
        return (1.0 - tt) * x0 + tt * x1

    def interp_velocity_fn(t, x0, x1):
        return (-2.0 * t) * x0 + (2.0 * t) * x1

    schedule = AutodiffInterpolationSchedule(
        interp_fn=interp_fn, interp_velocity_fn=interp_velocity_fn
    )

    def model_fn(t, xt, x0, x1):
        return interp_velocity_fn(t, x0, x1)

    loss_fn = build_flow_matching_loss(model_fn, schedule=schedule)
    x0 = jax.numpy.ones((3, 2))
    x1 = -jax.numpy.ones((3, 2))
    t = 0.4 * jax.numpy.ones((3, 1))
    loss = loss_fn(t, x0, x1, x0, x1, rng=jax.random.key(2))

    assert jax.numpy.allclose(loss, 0.0, atol=1e-5), "loss should be ~0"
