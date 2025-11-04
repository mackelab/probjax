import jax
import pytest

from probjax.nn.loss_fn import (
    build_denoising_loss,
    build_denoising_score_matching_loss,
    build_score_matching_loss,
    build_sliced_score_matching_loss,
    build_target_score_matching_loss,
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
