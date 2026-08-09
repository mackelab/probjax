import jax
import jax.numpy as jnp
import pytest

from probjax.core import inverse, inverse_and_logabsdet
from probjax.utils.odeint import odeint


def test_inverse_1d(invertible_function_1d):
    x = jnp.linspace(0, 1, 100)
    y = invertible_function_1d(x)

    mask = jnp.isfinite(y)

    inv_fun = inverse(invertible_function_1d)
    inv_y = jnp.asarray(inv_fun(y))

    assert x.shape == inv_y.shape, "Inverse function shape is not correct."
    assert jnp.allclose(x[mask], inv_y[mask], atol=1e-3, rtol=1e-3), (
        "Inverse function value is not correct."
    )


def test_inverse_and_logabsdet_1d(invertible_function_1d):
    x = jnp.linspace(0, 10, 100).reshape(-1, 1)
    y = jax.vmap(invertible_function_1d)(x)
    mask = jnp.isfinite(y).all(-1)

    inv_and_det_fn = inverse_and_logabsdet(invertible_function_1d)
    inv_y, logabsdet = jax.vmap(inv_and_det_fn)(y)

    print(x.shape, inv_y.shape, logabsdet.shape)

    assert x.shape == inv_y.shape, "Inverse function shape is not correct."
    assert jnp.allclose(x[mask], inv_y[mask], atol=1e-3, rtol=1e-3), (
        "Inverse function value is not correct."
    )

    J = jax.grad(lambda x: invertible_function_1d(x).sum())(x)
    logabsdet_true = -jnp.log(jnp.abs(J)).sum(-1)

    assert jnp.allclose(logabsdet[mask], logabsdet_true[mask], atol=1e-3, rtol=1e-3), (
        "Logabsdet is not correct."
    )


def test_inverse_and_logabsdet_wrapper_is_stateless():
    def f(x):
        return jnp.exp(x) + 1.0

    inv_and_logdet = inverse_and_logabsdet(f)

    y0 = f(jnp.array(0.2))
    x0, logdet0 = inv_and_logdet(y0)
    assert jnp.allclose(x0, 0.2, atol=1e-6, rtol=1e-6)

    y1 = f(jnp.array(1.3))
    x1, logdet1 = inv_and_logdet(y1)
    assert jnp.allclose(x1, 1.3, atol=1e-6, rtol=1e-6)

    expected0 = -jnp.log(jnp.exp(jnp.array(0.2)))
    expected1 = -jnp.log(jnp.exp(jnp.array(1.3)))
    assert jnp.allclose(logdet0, expected0, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet1, expected1, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_nested_jit():
    def f(x):
        return jax.jit(lambda z: jnp.exp(z) + 1.0)(x)

    x0 = jnp.array(0.7)
    y0 = f(x0)

    inv_and_logdet = inverse_and_logabsdet(f)
    x_rec, log_det = inv_and_logdet(y0)

    expected_log_det = -jnp.log(jnp.exp(x0))
    assert jnp.allclose(x_rec, x0, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(log_det, expected_log_det, atol=1e-6, rtol=1e-6)


# =============================================================================
# ODE integration tests
# =============================================================================


def test_inverse_odeint_linear_system():
    ts = jnp.linspace(0.0, 1.0, 1000)

    def drift(t, x):
        return -x

    def forward(x):
        return odeint(drift, x, ts, collect_trace=False)

    x0 = jnp.array([1.0, 2.0, 3.0])
    y = forward(x0)

    inv_forward = inverse(forward)
    x_inv = inv_forward(y)

    assert jnp.allclose(x0, x_inv, atol=1e-3, rtol=1e-3), (
        "Inverse function failed for ODE-based transform."
    )


def test_inverse_odeint_linear_system_with_drift_args():
    """Drift parameters forwarded as ``*args`` still support inversion."""
    ts = jnp.linspace(0.0, 1.0, 1000)
    rate = jnp.array(0.8)

    def drift(t, x, rate):
        del t
        return -rate * x

    def forward(x):
        return odeint(drift, x, ts, rate, collect_trace=False)

    x0 = jnp.array([1.0, -2.0, 0.5])
    y = forward(x0)

    inv_forward = inverse(forward)
    x_inv = inv_forward(y)

    assert jnp.allclose(x0, x_inv, atol=1e-3, rtol=1e-3), (
        "Inverse function failed for ODE-based transform with drift args."
    )


def test_inverse_odeint_with_traced_drift_args():
    """Traced positional drift args must survive through the inverse."""
    ts = jnp.linspace(0.0, 1.0, 400)

    def drift(t, x, rate, bias):
        del t
        return -rate * x + bias

    def forward(x, rate, bias):
        return odeint(
            drift,
            x,
            ts,
            rate,
            bias,
            collect_trace=False,
            method="rk4",
        )

    x0 = jnp.array([1.0, -2.0, 0.5])
    rate = jnp.array(0.7)
    bias = jnp.array(0.1)
    y = forward(x0, rate, bias)

    inv_forward = inverse(forward, invertible_arg=0)
    x_inv = inv_forward(y, rate, bias)

    assert jnp.allclose(x0, x_inv, atol=1e-3, rtol=1e-3), (
        "Inverse failed when odeint drift args are traced runtime values."
    )


def test_inverse_and_logabsdet_odeint_with_traced_drift_args():
    ts = jnp.linspace(0.0, 1.0, 400)

    def drift(t, x, rate, bias):
        del t
        return -rate * x + bias

    def forward(x, rate, bias):
        return odeint(
            drift,
            x,
            ts,
            rate,
            bias,
            collect_trace=False,
            method="rk4",
        )

    x0 = jnp.array([1.2, -0.3, 0.8])
    rate = jnp.array(0.4)
    bias = jnp.array(-0.2)
    y = forward(x0, rate, bias)

    inv_and_det = inverse_and_logabsdet(forward, invertible_arg=0)
    x_inv, logabsdet = inv_and_det(y, rate, bias)

    duration = ts[-1] - ts[0]
    expected_logabsdet = x0.shape[0] * rate * duration

    assert jnp.allclose(x0, x_inv, atol=1e-3, rtol=1e-3), (
        "Inverse+logabsdet failed when odeint drift args are traced values."
    )
    assert jnp.allclose(logabsdet, expected_logabsdet, atol=5e-2, rtol=5e-2), (
        "Inverse logabsdet for linear ODE drift is inconsistent with expectation."
    )


def test_inverse_odeint_with_partial_bound_drift():
    """Static drift configuration bound via ``functools.partial`` survives inversion."""
    from functools import partial

    ts = jnp.linspace(0.0, 1.0, 400)

    def drift(t, x, rate, bias, mode):
        del t
        if mode == "affine":
            return -rate * x + bias
        return -rate * x

    def forward(x, rate):
        bias = 0.1 * jnp.ones_like(rate)
        bound_drift = partial(drift, mode="affine")
        return odeint(
            bound_drift,
            x,
            ts,
            rate,
            bias,
            collect_trace=False,
            method="rk4",
        )

    x0 = jnp.array([1.0, -2.0, 0.5])
    rate = jnp.array(0.7)
    y = forward(x0, rate)

    inv_forward = inverse(forward, invertible_arg=0)
    x_inv = inv_forward(y, rate)

    assert jnp.allclose(x0, x_inv, atol=1e-3, rtol=1e-3), (
        "Inverse failed when drift configuration was bound via partial."
    )


def test_inverse_odeint_with_static_bool_control_flow_via_closure():
    """Static boolean control-flow fixed by closure still works under inversion."""
    ts = jnp.linspace(0.0, 1.0, 400)

    def make_drift(use_bias):
        def drift(t, x, rate, bias):
            del t
            if use_bias:
                return -rate * x + bias
            return -rate * x

        return drift

    def forward_true(x, rate):
        bias = 0.1 * jnp.ones_like(rate)
        return odeint(
            make_drift(True),
            x,
            ts,
            rate,
            bias,
            collect_trace=False,
            method="rk4",
        )

    def forward_false(x, rate):
        bias = 0.1 * jnp.ones_like(rate)
        return odeint(
            make_drift(False),
            x,
            ts,
            rate,
            bias,
            collect_trace=False,
            method="rk4",
        )

    x0 = jnp.array([1.0, -2.0, 0.5])
    rate = jnp.array(0.7)

    y_true = forward_true(x0, rate)
    inv_forward_true = inverse(forward_true, invertible_arg=0)
    x_inv_true = inv_forward_true(y_true, rate)
    assert jnp.allclose(x0, x_inv_true, atol=1e-3, rtol=1e-3), (
        "Inverse failed with static boolean control-flow set to True."
    )

    y_false = forward_false(x0, rate)
    inv_forward_false = inverse(forward_false, invertible_arg=0)
    x_inv_false = inv_forward_false(y_false, rate)
    assert jnp.allclose(x0, x_inv_false, atol=1e-3, rtol=1e-3), (
        "Inverse failed with static boolean control-flow set to False."
    )


def test_inverse_and_logabsdet_odeint_hutchinson_matches_exact_in_expectation():
    """Hutchinson trace estimator matches exact log-det in expectation."""
    ts = jnp.linspace(0.0, 1.0, 200)
    A = jnp.array(
        [
            [-0.5, 0.1, 0.0],
            [0.0, -0.3, 0.2],
            [0.1, 0.0, -0.4],
        ]
    )

    def drift(t, x, A):
        del t
        return A @ x

    def forward_exact(x, A):
        return odeint(
            drift,
            x,
            ts,
            A,
            collect_trace=False,
            method="rk4",
            trace_estimator="exact",
        )

    def forward_hutch(x, A, rng):
        return odeint(
            drift,
            x,
            ts,
            A,
            collect_trace=False,
            method="rk4",
            trace_estimator="hutchinson",
            num_samples=1,
            logdet_rng=rng,
        )

    x0 = jnp.array([1.0, -0.5, 0.3])

    y_exact = forward_exact(x0, A)
    inv_exact = inverse_and_logabsdet(forward_exact, invertible_arg=0)
    _, logdet_exact = inv_exact(y_exact, A)

    seeds = jax.random.split(jax.random.PRNGKey(0), 256)

    def single_hutch(rng):
        y = forward_hutch(x0, A, rng)
        inv_hutch = inverse_and_logabsdet(forward_hutch, invertible_arg=0)
        _, logdet = inv_hutch(y, A, rng)
        return jnp.squeeze(logdet)

    logdets_hutch = jax.vmap(single_hutch)(seeds)
    mean_hutch = jnp.mean(logdets_hutch)

    duration = ts[-1] - ts[0]
    expected = -jnp.trace(A) * duration

    assert jnp.allclose(jnp.squeeze(logdet_exact), expected, atol=1e-2, rtol=1e-2), (
        "Exact log-det does not match analytical -tr(A)*T."
    )
    assert jnp.allclose(mean_hutch, expected, atol=5e-2, rtol=5e-2), (
        "Hutchinson MC mean over seeds does not match exact log-det."
    )


def test_inverse_and_logabsdet_odeint_hutchinson_num_samples_reduces_variance():
    """Increasing ``num_samples`` reduces per-trajectory variance."""
    ts = jnp.linspace(0.0, 1.0, 100)
    A = jnp.array(
        [
            [-0.3, 0.2, 0.0, 0.1],
            [0.1, -0.4, 0.3, 0.0],
            [0.0, 0.1, -0.2, 0.2],
            [0.2, 0.0, 0.1, -0.5],
        ]
    )

    def drift(t, x, A):
        del t
        return A @ x

    def forward_k1(x, A, rng):
        return odeint(
            drift,
            x,
            ts,
            A,
            collect_trace=False,
            method="rk4",
            trace_estimator="hutchinson",
            num_samples=1,
            logdet_rng=rng,
        )

    def forward_k16(x, A, rng):
        return odeint(
            drift,
            x,
            ts,
            A,
            collect_trace=False,
            method="rk4",
            trace_estimator="hutchinson",
            num_samples=16,
            logdet_rng=rng,
        )

    x0 = jnp.array([1.0, -0.5, 0.3, 0.8])
    seeds = jax.random.split(jax.random.PRNGKey(1), 128)

    def single(forward, rng):
        y = forward(x0, A, rng)
        inv = inverse_and_logabsdet(forward, invertible_arg=0)
        _, logdet = inv(y, A, rng)
        return jnp.squeeze(logdet)

    logdets_k1 = jax.vmap(lambda r: single(forward_k1, r))(seeds)
    logdets_k16 = jax.vmap(lambda r: single(forward_k16, r))(seeds)

    var_k1 = jnp.var(logdets_k1)
    var_k16 = jnp.var(logdets_k16)

    assert var_k16 < var_k1, (
        f"num_samples=16 variance ({var_k16:.4e}) should be below num_samples=1 "
        f"variance ({var_k1:.4e})."
    )
    assert var_k16 < 0.5 * var_k1, (
        f"num_samples=16 variance ({var_k16:.4e}) should be at least ~2x below "
        f"num_samples=1 variance ({var_k1:.4e})."
    )


def test_inverse_and_logabsdet_odeint_hutchinson_normal_probes():
    """The ``sample_dist='normal'`` path is unbiased in expectation too."""
    ts = jnp.linspace(0.0, 1.0, 100)
    A = jnp.array(
        [
            [-0.3, 0.1, 0.0],
            [0.0, -0.2, 0.2],
            [0.1, 0.0, -0.4],
        ]
    )

    def drift(t, x, A):
        del t
        return A @ x

    def forward(x, A, rng):
        return odeint(
            drift,
            x,
            ts,
            A,
            collect_trace=False,
            method="rk4",
            trace_estimator="hutchinson",
            num_samples=4,
            sample_dist="normal",
            logdet_rng=rng,
        )

    x0 = jnp.array([1.0, -0.5, 0.3])
    seeds = jax.random.split(jax.random.PRNGKey(2), 256)

    def single(rng):
        y = forward(x0, A, rng)
        inv = inverse_and_logabsdet(forward, invertible_arg=0)
        _, logdet = inv(y, A, rng)
        return jnp.squeeze(logdet)

    logdets = jax.vmap(single)(seeds)
    mean_logdet = jnp.mean(logdets)

    duration = ts[-1] - ts[0]
    expected = -jnp.trace(A) * duration

    assert jnp.allclose(mean_logdet, expected, atol=0.1, rtol=0.1), (
        "Normal-probe Hutchinson MC mean does not match analytical -tr(A)*T."
    )


def test_inverse_and_logabsdet_odeint_custom_trace_fn():
    """User-supplied ``trace_fn`` callable is honoured for analytic traces."""
    ts = jnp.linspace(0.0, 1.0, 400)

    def drift(t, x, rate):
        del t
        return -rate * x

    def trace_fn(drift_flat, t, x_flat, args):
        del drift_flat, t
        (rate,) = args
        return -rate * x_flat.shape[0]

    def forward(x, rate):
        return odeint(
            drift,
            x,
            ts,
            rate,
            collect_trace=False,
            method="rk4",
            trace_estimator=trace_fn,
        )

    x0 = jnp.array([1.0, -0.5, 0.3])
    rate = jnp.array(0.7)
    y = forward(x0, rate)

    inv_and_det = inverse_and_logabsdet(forward, invertible_arg=0)
    x_inv, logdet = inv_and_det(y, rate)

    duration = ts[-1] - ts[0]
    expected = x0.shape[0] * rate * duration

    assert jnp.allclose(x0, x_inv, atol=1e-3, rtol=1e-3), (
        "Custom trace_fn: inverse failed to recover x0."
    )
    assert jnp.allclose(jnp.squeeze(logdet), expected, atol=1e-3, rtol=1e-3), (
        "Custom trace_fn: log-det does not match analytical expectation."
    )


def test_inverse_and_logabsdet_odeint_hutchinson_requires_logdet_rng():
    """Asking for Hutchinson without an RNG must raise."""
    ts = jnp.linspace(0.0, 1.0, 50)

    def drift(t, x):
        return -x

    def forward(x):
        return odeint(
            drift,
            x,
            ts,
            collect_trace=False,
            trace_estimator="hutchinson",
        )

    x0 = jnp.array([1.0, 2.0])
    y = forward(x0)
    inv_and_det = inverse_and_logabsdet(forward)
    with pytest.raises(ValueError, match="logdet_rng"):
        inv_and_det(y)


def test_inverse_and_logabsdet_odeint_unknown_trace_estimator_raises():
    """Unknown string estimators are rejected up-front."""
    ts = jnp.linspace(0.0, 1.0, 50)

    def drift(t, x):
        return -x

    def forward(x):
        return odeint(
            drift,
            x,
            ts,
            collect_trace=False,
            trace_estimator="not_a_real_estimator",
        )

    x0 = jnp.array([1.0, 2.0])
    y = forward(x0)
    inv_and_det = inverse_and_logabsdet(forward)
    with pytest.raises(ValueError, match="Unknown trace_estimator"):
        inv_and_det(y)
