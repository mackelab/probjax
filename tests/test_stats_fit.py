import jax
import jax.numpy as jnp
import optax
import pytest
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
        body = _make_step(
            _quadratic_loss, tx, _array_fetch({"data": data}, 32, 128), None, 1
        )
        init = (
            params,
            tx.init(params),
            jax.random.key(0),
            jnp.asarray(False),
            jnp.asarray(0, jnp.int32),
        )
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


# ---------------------------------------------------------------------------
# Iterable data sources
# ---------------------------------------------------------------------------


def _reference_batches(data, key, num_steps, batch_size):
    """The exact minibatch sequence `fit` draws from an array, as a list.

    Lets an array run and an iterable run be compared on identical data, which
    is the only way the two paths can be held to the same numbers.
    """
    batches, rng = [], key
    for _ in range(num_steps):
        rng, rng_batch, _ = jax.random.split(rng, 3)
        idx = jax.random.randint(rng_batch, (batch_size,), 0, data.shape[0])
        batches.append({"data": data[idx]})
    return batches


def test_fit_array_and_iterable_agree_on_identical_batches():
    data = jax.random.normal(jax.random.key(0), (256, 2)) + jnp.array([2.0, -1.0])
    key, p0 = jax.random.key(1), {"w": jnp.zeros(2)}
    batches = _reference_batches(data, key, 120, 64)

    array_params, array_losses = fit(
        _quadratic_loss,
        p0,
        key,
        {"data": data},
        num_steps=120,
        batch_size=64,
        learning_rate=1e-2,
    )
    stream_params, stream_losses = fit(
        _quadratic_loss,
        p0,
        key,
        batches,
        num_steps=120,
        learning_rate=1e-2,
    )
    # Same batches, same seed, same update rule -- these must not merely be
    # close, and a drift here means one path is consuming randomness the other
    # is not.
    assert jnp.array_equal(array_losses, stream_losses)
    assert jnp.array_equal(array_params["w"], stream_params["w"])


def test_fit_restarts_a_short_iterable():
    data = jax.random.normal(jax.random.key(0), (128, 2))
    batches = [{"data": data[i : i + 32]} for i in range(0, 128, 32)]  # 4 batches
    _, losses = fit(
        _quadratic_loss,
        {"w": jnp.zeros(2)},
        jax.random.key(1),
        batches,
        num_steps=20,
        learning_rate=1e-2,
    )
    assert losses.shape == (20,)
    assert jnp.all(jnp.isfinite(losses))


def test_fit_accepts_an_endless_generator():
    data = jax.random.normal(jax.random.key(0), (128, 2))

    def stream():
        key = jax.random.key(7)
        while True:
            key, sub = jax.random.split(key)
            yield {"data": data[jax.random.randint(sub, (32,), 0, 128)]}

    _, losses = fit(
        _quadratic_loss,
        {"w": jnp.zeros(2)},
        jax.random.key(1),
        stream(),
        num_steps=30,
        learning_rate=1e-2,
    )
    assert losses.shape == (30,)
    assert jnp.all(jnp.isfinite(losses))


def test_fit_rejects_batch_size_for_an_iterable():
    batches = [{"data": jnp.zeros((8, 2))}] * 4
    with pytest.raises(ValueError, match="batch_size cannot be set"):
        fit(
            _quadratic_loss,
            {"w": jnp.zeros(2)},
            jax.random.key(0),
            batches,
            num_steps=4,
            batch_size=4,
        )


def test_fit_reports_a_ragged_batch_clearly():
    batches = [{"data": jnp.zeros((8, 2))}, {"data": jnp.zeros((3, 2))}]
    # The failure happens inside an io_callback, where JAX would otherwise
    # bury it under a JaxRuntimeError; fit is expected to surface the original.
    with pytest.raises(ValueError, match="different shape or dtype"):
        fit(
            _quadratic_loss,
            {"w": jnp.zeros(2)},
            jax.random.key(0),
            batches,
            num_steps=2,
        )


def test_fit_reports_an_exhausted_iterator_clearly():
    with pytest.raises(RuntimeError, match="could not be restarted"):
        fit(
            _quadratic_loss,
            {"w": jnp.zeros(2)},
            jax.random.key(0),
            iter([{"data": jnp.zeros((8, 2))}]),
            num_steps=5,
        )


def test_batch_stream_classification_is_fixed_not_inferred():
    """A list of arrays streams; a dict of arrays does not.

    Both readings are pytrees of arrays, so this cannot be inferred from
    structure. Guessing it silently trained a flow on a stacked
    ``(num_batches, batch, features)`` array once, which is why the rule is
    pinned here rather than left to shape heuristics.
    """
    from probjax.stats.fit import is_batch_stream

    arr = jnp.zeros((4, 2))
    assert not is_batch_stream(arr)
    assert not is_batch_stream({"data": arr, "context": arr})
    assert is_batch_stream([arr, arr])
    assert is_batch_stream((arr, arr))
    assert is_batch_stream([{"data": arr}])
    assert is_batch_stream(iter([arr]))


def test_flow_fit_from_an_iterable_matches_an_array_run():
    raw = jax.random.normal(jax.random.key(0), (512, 2)) * 20.0 + 100.0
    batches = [raw[i : i + 128] for i in range(0, 512, 128)]

    from_array = maf(2, 3, rngs=nnx.Rngs(0))
    losses_array = from_array.fit(jax.random.key(1), raw, num_steps=100, batch_size=128)
    from_stream = maf(2, 3, rngs=nnx.Rngs(0))
    losses_stream = from_stream.fit(jax.random.key(1), batches, num_steps=100)

    # The same 512 examples either way, so the standardising transform fitted
    # from a handful of batches must be the one fitted from the whole array.
    shift_a, scale_a = from_array.standardization
    shift_s, scale_s = from_stream.standardization
    assert jnp.allclose(shift_a, shift_s)
    assert jnp.allclose(scale_a, scale_s)
    assert jnp.all(jnp.isfinite(losses_stream))
    assert abs(float(losses_stream[-1]) - float(losses_array[-1])) < 0.5


def test_flow_fit_from_an_iterable_of_context_pairs():
    context = jax.random.normal(jax.random.key(3), (256, 1))
    data = jax.random.normal(jax.random.key(4), (256, 2)) + context
    pairs = [(data[i : i + 64], context[i : i + 64]) for i in range(0, 256, 64)]

    flow = maf(2, 2, rngs=nnx.Rngs(1), context_features=1)
    losses = flow.fit(jax.random.key(5), pairs, num_steps=80)
    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < losses[0]

    with pytest.raises(ValueError, match="context cannot be passed"):
        flow.fit(jax.random.key(5), pairs, context=context, num_steps=5)


def test_flow_fit_streams_a_list_of_arrays_rather_than_stacking_it():
    raw = jax.random.normal(jax.random.key(0), (256, 2))
    batches = [raw[i : i + 64] for i in range(0, 256, 64)]

    flow = maf(2, 3, rngs=nnx.Rngs(0))
    losses = flow.fit(jax.random.key(1), batches, num_steps=30)

    # Were the list stacked into (4, 64, 2), the loss would be computed over a
    # 3-D "batch" and the standardisation would still look plausible -- so
    # check the model actually sees 2-D examples by evaluating it on one.
    assert jnp.all(jnp.isfinite(losses))
    assert jnp.isfinite(flow.as_dist().logpdf(jnp.zeros(2)))


def test_fit_on_step_callback_fires_and_stops():
    data = jax.random.normal(jax.random.key(0), (128, 2))
    seen = []

    def on_step(step, loss):
        seen.append((step, loss))
        return False if step >= 30 else None

    _, losses = fit(
        _quadratic_loss,
        {"w": jnp.zeros(2)},
        jax.random.key(1),
        {"data": data},
        num_steps=200,
        batch_size=32,
        learning_rate=1e-2,
        on_step=on_step,
        log_every=10,
    )
    assert [s for s, _ in seen] == [0, 10, 20, 30]
    assert all(isinstance(loss, float) for _, loss in seen)
    # Steps after the stop are no-ops and must not be reported as losses.
    assert losses.shape == (31,)
    assert jnp.all(jnp.isfinite(losses))


def test_fit_on_step_runs_to_the_end_when_it_never_stops():
    data = jax.random.normal(jax.random.key(0), (128, 2))
    seen = []
    _, losses = fit(
        _quadratic_loss,
        {"w": jnp.zeros(2)},
        jax.random.key(1),
        {"data": data},
        num_steps=25,
        batch_size=32,
        learning_rate=1e-2,
        on_step=lambda s, _: seen.append(s),
        log_every=5,
    )
    assert seen == [0, 5, 10, 15, 20]
    assert losses.shape == (25,)
