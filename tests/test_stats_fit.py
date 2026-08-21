import jax
import jax.numpy as jnp
import optax
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
    distribution = flow.as_dist()
    x_test = jnp.ones((4, 2))
    lp_before = distribution.logpdf(x_test)

    data = jax.random.normal(jax.random.key(0), (512, 2)) * jnp.array([
        1.5,
        0.5,
    ]) + jnp.array([2.0, -1.0])
    losses = flow.fit(jax.random.key(1), data, num_steps=150, batch_size=128)

    # `fit` standardises the data, so on this near-Gaussian target the flow
    # starts at the optimum and the loss has nowhere to fall; requiring a
    # strict decrease would be testing noise rather than training.
    assert losses[-1] <= losses[0] + 0.05
    assert not jnp.allclose(lp_before, distribution.logpdf(x_test))

    # logpdf must agree with brute-force change of variables at the trained
    # weights (regression for the nested-jit logdet accumulation bug).
    zs = jax.random.normal(jax.random.key(7), (8, 2))
    # `transform` (not `transformation`) is the data-space map, i.e. the one
    # whose Jacobian belongs in the change of variables that logpdf reports.
    fwd = lambda z: flow.transform(z)  # noqa: E731
    ys = jax.vmap(fwd)(zs)
    jacs = jax.vmap(jax.jacobian(fwd))(zs)
    reference = jnp.sum(jax.scipy.stats.norm.logpdf(zs), axis=-1) - jnp.log(
        jnp.abs(jnp.linalg.det(jacs))
    )
    assert jnp.allclose(distribution.logpdf(ys), reference, atol=5e-3)


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


def _quadratic_loss(params, rng, batch):
    del rng
    return jnp.mean((batch["data"] - params["w"]) ** 2)


def test_fit_does_not_unroll_over_num_steps():
    """The program `fit` builds must not grow with num_steps.

    This is the whole reason the loop is a scan rather than Python: a long run
    should cost one compile, not one per step. Lowering the scan `fit` builds
    (rather than `fit` itself, which finishes with a host-side finiteness
    check) is what pins the property down.
    """
    from probjax.stats.fit import _array_fetch, _build_optimizer, _make_step

    data = jnp.zeros((128, 2))
    params = {"w": jnp.zeros(2)}

    def program_size(num_steps):
        tx = _build_optimizer(1e-2, num_steps, "constant", 10.0)
        body = _make_step(_quadratic_loss, tx, _array_fetch({"data": data}, 32, 128))
        init = (params, tx.init(params), jax.random.key(0))
        run = jax.jit(lambda c: jax.lax.scan(body, c, None, length=num_steps))
        return len(run.lower(init).as_text())

    small, large = program_size(100), program_size(10_000)
    # Not exactly equal: the trip count appears as a literal in the HLO, so a
    # few characters differ. Unrolling would multiply the size, not shift it.
    assert abs(large - small) < 0.01 * small


def test_fit_reproduces_the_reference_update_sequence():
    """Pins the per-step randomness: three splits per step, batch key first.

    Any change to how keys are drawn silently changes every trained model, so
    the sequence is spelled out here rather than left to the implementation.
    """
    from probjax.stats.fit import _build_optimizer

    data = jax.random.normal(jax.random.key(0), (256, 2)) + jnp.array([2.0, -1.0])
    key, params = jax.random.key(1), {"w": jnp.zeros(2)}
    tx = _build_optimizer(1e-2, 300, "constant", 10.0)

    # The step is jitted, as the Python loop this replaced had it: an eager
    # step fuses differently and drifts by an ulp, which would make the
    # comparison about XLA fusion rather than about the update sequence.
    @jax.jit
    def step(params, opt_state, rng, minibatch):
        loss, grads = jax.value_and_grad(_quadratic_loss)(params, rng, minibatch)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    opt_state, rng, expected = tx.init(params), key, []
    reference = dict(params)
    for _ in range(300):
        rng, rng_batch, rng_loss = jax.random.split(rng, 3)
        idx = jax.random.randint(rng_batch, (64,), 0, 256)
        reference, opt_state, loss = step(
            reference, opt_state, rng_loss, {"data": data[idx]}
        )
        expected.append(loss)

    got_params, got_losses = fit(
        _quadratic_loss,
        params,
        key,
        {"data": data},
        num_steps=300,
        batch_size=64,
        learning_rate=1e-2,
    )
    assert jnp.array_equal(jnp.stack(expected), got_losses)
    assert jnp.array_equal(reference["w"], got_params["w"])
