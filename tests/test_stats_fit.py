import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn import LinearFlow, maf
from probjax.stats import fit


def test_fit_functional_dict_batch():
    def loss_fn(params, rng, batch):
        del rng
        return jnp.mean((batch["data"] - params["w"]) ** 2)

    data = jax.random.normal(jax.random.key(0), (256, 2)) + jnp.array([2.0, -1.0])
    params, losses = fit(
        loss_fn,
        {"w": jnp.zeros(2)},
        jax.random.key(1),
        {"data": data},
        num_steps=300,
        learning_rate=1e-2,
    )
    assert losses.shape == (300,)
    assert jnp.allclose(params["w"], jnp.array([2.0, -1.0]), atol=0.2)


def test_flow_fit_trains_in_place_and_stays_normalized():
    flow = maf(2, 3, rngs=nnx.Rngs(0))
    x_test = jnp.ones((4, 2))
    lp_before = flow.logpdf(x_test)

    data = jax.random.normal(jax.random.key(0), (512, 2)) * jnp.array(
        [1.5, 0.5]
    ) + jnp.array([2.0, -1.0])
    losses = flow.fit(jax.random.key(1), data, num_steps=150, batch_size=128)

    assert losses[-1] < losses[0]
    assert not jnp.allclose(lp_before, flow.logpdf(x_test))

    # logpdf must agree with brute-force change of variables at the trained
    # weights (regression for the nested-jit logdet accumulation bug).
    zs = jax.random.normal(jax.random.key(7), (8, 2))
    fwd = lambda z: flow.transformation(z)  # noqa: E731
    ys = jax.vmap(fwd)(zs)
    jacs = jax.vmap(jax.jacobian(fwd))(zs)
    reference = jnp.sum(jax.scipy.stats.norm.logpdf(zs), axis=-1) - jnp.log(
        jnp.abs(jnp.linalg.det(jacs))
    )
    assert jnp.allclose(flow.logpdf(ys), reference, atol=5e-3)


def test_flow_fit_conditional():
    flow = maf(2, 2, rngs=nnx.Rngs(1), context_features=1)
    context = jax.random.normal(jax.random.key(3), (256, 1))
    data = jax.random.normal(jax.random.key(4), (256, 2)) + context
    losses = flow.fit(
        jax.random.key(5), data, context=context, num_steps=60, batch_size=64
    )
    assert losses[-1] < losses[0]
    assert jnp.all(jnp.isfinite(losses))


def test_flow_matcher_fit():
    class TinyNet(nnx.Module):
        def __init__(self, rngs):
            self.proj = nnx.Linear(3, 3, rngs=rngs)

        def __call__(self, t, x, **kwargs):
            return self.proj(x)

    model = LinearFlow(TinyNet(nnx.Rngs(0)))
    data = jax.random.normal(jax.random.key(1), (256, 3))
    losses = model.fit(jax.random.key(2), data, num_steps=30)
    assert losses.shape == (30,)
    assert jnp.all(jnp.isfinite(losses))
