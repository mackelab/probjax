import jax
import jax.numpy as jnp
import pytest

from probjax.core import custom_inverse, inverse, inverse_and_logabsdet
from probjax.core.registry import REGISTRY, Context
from probjax.utils.functions import split_drift
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


def test_inverse_of_custom_inverse_returns_custom_inverse():
    @custom_inverse
    def f(x):
        return 3.0 * x + 1.0

    f.definv(lambda y: (y - 1.0) / 3.0)
    f.definv_and_logdet(lambda y: ((y - 1.0) / 3.0, -jnp.asarray(2.5)))

    inv_f = inverse(f)
    assert isinstance(inv_f, custom_inverse)

    x0 = jnp.asarray(0.7)
    y0 = f(x0)
    assert jnp.allclose(inv_f(y0), x0, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(inverse(inv_f)(x0), y0, atol=1e-6, rtol=1e-6)


def test_inverse_of_custom_inverse_logdet_priority_and_fallback():
    @custom_inverse
    def g(x):
        return x + 1.0

    g.definv_and_logdet(lambda y: (y - 1.0, -jnp.asarray(1.0)))
    g.defvalue_and_logdet(lambda x: (x + 1.0, jnp.asarray(9.0)))
    inv_g = inverse(g)
    y_g, logdet_g = inverse_and_logabsdet(inv_g)(jnp.asarray(2.0))
    assert jnp.allclose(y_g, jnp.asarray(3.0), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet_g, jnp.asarray(9.0), atol=1e-6, rtol=1e-6)

    @custom_inverse
    def h(x):
        return 4.0 * x - 2.0

    h.definv_and_logdet(lambda y: ((y + 2.0) / 4.0, -jnp.asarray(7.0)))
    inv_h = inverse(h)
    y_h, logdet_h = inverse_and_logabsdet(inv_h)(jnp.asarray(0.5))
    assert jnp.allclose(y_h, jnp.asarray(0.0), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet_h, jnp.asarray(7.0), atol=1e-6, rtol=1e-6)


def test_inverse_of_custom_inverse_respects_configuration_guards():
    @custom_inverse
    def f(x):
        return x + 1.0

    f.definv_and_logdet(lambda y: (y - 1.0, jnp.asarray(0.0)))

    with pytest.raises(ValueError):
        inverse(f, invertible_arg=1)

    with pytest.raises(ValueError):
        inverse(f, static_argnums=(0,))


def test_inverse_of_custom_inverse_requires_registered_inverse():
    @custom_inverse
    def f(x):
        return x + 1.0

    with pytest.raises(AttributeError):
        inverse(f)


def test_inverse_split():
    x = jnp.ones((10, 2))

    def f(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x2, x1], axis=-1)

    inv_f = inverse(f)
    inv_x = inv_f(x)

    assert jnp.allclose(x, inv_x, atol=1e-3, rtol=1e-3), (
        "Inverse function value is not correct."
    )


@pytest.mark.parametrize(
    "fun,x",
    [
        (lambda x: jnp.sin(x), jnp.linspace(-1.0, 1.0, 7)),
        (lambda x: jnp.arcsin(x), jnp.linspace(-0.9, 0.9, 7)),
        (lambda x: jnp.cos(x), jnp.linspace(0.2, 2.9, 7)),
        (lambda x: jnp.arccos(x), jnp.linspace(-0.9, 0.9, 7)),
        (lambda x: jnp.tan(x), jnp.linspace(-1.0, 1.0, 7)),
        (lambda x: jnp.arctan(x), jnp.linspace(-3.0, 3.0, 7)),
        (lambda x: jnp.tanh(x), jnp.linspace(-0.5, 0.5, 7)),
        (lambda x: jnp.sinh(x), jnp.linspace(-0.4, 0.4, 7)),
        (lambda x: jnp.exp(x), jnp.linspace(-1.0, 1.0, 7)),
        (lambda x: jnp.sqrt(x), jnp.linspace(0.25, 2.0, 7)),
        (lambda x: jnp.cbrt(x), jnp.linspace(-8.0, 8.0, 7)),
        (lambda x: jnp.copy(x), jnp.linspace(-0.5, 0.5, 7)),
        (lambda x: jnp.log1p(x), jnp.linspace(0.05, 0.95, 7)),
    ],
    ids=[
        "sin",
        "asin",
        "cos",
        "acos",
        "tan",
        "atan",
        "tanh",
        "sinh",
        "exp",
        "sqrt",
        "cbrt",
        "copy",
        "log1p",
    ],
)
def test_inverse_univariate_registry(fun, x):
    inv_fun = inverse(fun)
    y = fun(x)
    x_rec = jnp.asarray(inv_fun(y))
    assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "fun,x",
    [
        (lambda x: x * 3.0, jnp.linspace(0.1, 1.0, 7)),
        (lambda x: x / 2.5, jnp.linspace(0.5, 2.0, 7)),
        (lambda x: x + 1.25, jnp.linspace(-0.5, 0.5, 7)),
        (lambda x: x - 0.75, jnp.linspace(-1.5, 1.5, 7)),
        (lambda x: jnp.power(x, 3.0), jnp.linspace(0.5, 1.5, 7)),
    ],
    ids=["mul", "div", "add", "sub", "pow"],
)
def test_inverse_bivariate_registry(fun, x):
    inv_fun = inverse(fun)
    y = fun(x)
    x_rec = jnp.asarray(inv_fun(y))
    assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_scatter():
    x0 = jnp.array([1.0, 2.0, 3.0])

    def f(x):
        y = jnp.ones((10, 3))
        y = y.at[-1].set(x * 2.0)
        return y[-1]

    inv_f = inverse(f)
    x_inv = inv_f(f(x0))

    assert jnp.allclose(x0, x_inv, atol=1e-6, rtol=1e-6), (
        "Inverse function failed for scatter updates."
    )


def test_inverse_gather_permutation():
    indices = jnp.array([2, 0, 1], dtype=jnp.int32)

    def f(x):
        return x[indices]

    x0 = jnp.array([0.3, -1.2, 2.5])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_squeeze_broadcast():
    def f(x):
        y = jnp.expand_dims(x, axis=0)
        y = jnp.broadcast_to(y, (1,) + x.shape)
        return jnp.squeeze(y, axis=0)

    x0 = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec)


def test_inverse_rev_reshape():
    def f(x):
        y = jnp.reshape(x, (2, 2))
        y = jnp.flip(y, axis=0)
        return jnp.reshape(y, (-1,))

    x0 = jnp.array([1.0, 2.0, 3.0, 4.0])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec)


def test_inverse_transpose():
    def f(x):
        return jnp.transpose(x, (2, 0, 1))

    x0 = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec)


def test_inverse_slice_dynamic_slice():
    def f(x):
        head = jax.lax.dynamic_slice(x, (0,), (2,))
        tail = jax.lax.dynamic_slice(x, (2,), (x.shape[0] - 2,))
        prefix = head[:1]
        suffix = jnp.concatenate([head[1:], tail], axis=0)
        return jnp.concatenate([prefix, suffix], axis=0)

    x0 = jnp.array([5.0, 6.0, 7.0, 8.0])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec)


def test_inverse_convert_element_type():
    def f(x):
        return x.astype(jnp.float64)

    x0 = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float32)
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert x_rec.dtype == x0.dtype
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_bitcast_convert_type():
    def f(x):
        return jax.lax.bitcast_convert_type(x, jnp.uint32)

    x0 = jnp.array([0.0, 1.5, -2.25, 3.75], dtype=jnp.float32)
    eqn = jax.make_jaxpr(f)(x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.bitcast_convert_type_p, Context.INVERSE)
    result = rule(
        eqn,
        [None],
        [f(x0)],
    )
    x_rec = result.resolved_vals[0]
    assert x_rec.dtype == x0.dtype
    assert jnp.array_equal(
        jax.lax.bitcast_convert_type(x_rec, jnp.uint32),
        jax.lax.bitcast_convert_type(x0, jnp.uint32),
    )


def test_inverse_select_n():
    def f(x):
        cond = jnp.ones_like(x, dtype=bool)
        return jnp.select([cond], [x], default=-x)

    x0 = jnp.array([0.5, 1.2, 3.4])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


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
    # With 16x more probe vectors per trajectory, variance should drop
    # substantially. Leave a generous margin because vmap over paths and
    # Rademacher finite-sample noise introduce their own variability.
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


def test_inverse_and_logabsdet_tan():
    def f(x):
        return jnp.tan(x)

    x = jnp.linspace(-0.8, 0.8, 21)
    y = f(x)

    inv_and_det_fn = inverse_and_logabsdet(f)
    x_rec, logabsdet = jax.vmap(inv_and_det_fn)(y)

    assert jnp.allclose(x_rec, x, atol=1e-6, rtol=1e-6)

    jac = jax.grad(lambda t: f(t).sum())(x)
    expected = -jnp.log(jnp.abs(jac))
    assert jnp.allclose(logabsdet, expected, atol=1e-6, rtol=1e-6)


def test_inverse_dot_general_lhs():
    w = jnp.array([
        [2.0, 0.2, -0.1],
        [0.0, 1.5, 0.3],
        [0.0, 0.0, 1.1],
    ])

    def f(x):
        return x @ w

    x0 = jnp.array([0.3, -1.2, 2.1])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_dot_general_rhs():
    x = jnp.array([
        [1.2, 0.1, -0.2],
        [0.0, 1.1, 0.3],
        [-0.4, 0.2, 0.9],
    ])

    def f(w):
        return x @ w

    w0 = jnp.array([
        [0.4, -1.0],
        [1.2, 0.5],
        [-0.3, 2.0],
    ])
    inv_f = inverse(f)
    w_rec = inv_f(f(w0))
    assert jnp.allclose(w0, w_rec, atol=1e-6, rtol=1e-6)


def test_inverse_dot_general_non_square_raises():
    w = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    def f(x):
        return x @ w

    x0 = jnp.array([0.2, -0.3, 1.4])
    y0 = f(x0)
    eqn = jax.make_jaxpr(f)(x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.dot_general_p, Context.INVERSE)

    # Calling the rule with non-square matrix should raise
    with pytest.raises(NotImplementedError):
        rule(eqn, [None, w], [y0])


def test_inverse_and_logabsdet_dot_general_vector():
    w = jnp.array([
        [1.8, 0.2, -0.1],
        [0.0, 1.3, 0.4],
        [0.0, 0.0, 0.9],
    ])

    def f(x):
        return x @ w

    x0 = jnp.array([0.2, -0.7, 1.3])
    inv_and_det = inverse_and_logabsdet(f)
    x_rec, log_det = inv_and_det(f(x0))

    _, logabsdet_w = jnp.linalg.slogdet(w)
    expected = -logabsdet_w

    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(log_det, expected, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_dot_general_rhs():
    x = jnp.array([
        [1.6, 0.1, -0.2],
        [0.0, 1.2, 0.3],
        [0.0, 0.0, 1.1],
    ])

    def f(w):
        return x @ w

    w0 = jnp.array([
        [0.4, -1.0],
        [1.2, 0.5],
        [-0.3, 2.0],
    ])
    inv_and_det = inverse_and_logabsdet(f)
    w_rec, log_det = inv_and_det(f(w0))

    _, logabsdet_x = jnp.linalg.slogdet(x)
    expected = -w0.shape[-1] * logabsdet_x

    assert jnp.allclose(w0, w_rec, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(log_det, expected, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_dot_general_non_square_raises():
    w = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    def f(x):
        return x @ w

    x0 = jnp.array([0.2, -0.3, 1.4])
    y0 = f(x0)
    eqn = jax.make_jaxpr(f)(x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.dot_general_p, Context.INVERSE_LOGDET)

    with pytest.raises(NotImplementedError):
        rule(eqn, [None, w], [y0], context=None)


def test_inverse_cond_with_known_branch():
    def f(pred, x):
        return jax.lax.cond(
            pred,
            lambda t: jnp.exp(t),
            lambda t: t + 2.5,
            x,
        )

    inv_f = inverse(f, invertible_arg=1)

    x_true = jnp.array(0.7)
    y_true = f(True, x_true)
    x_true_rec = inv_f(True, y_true)

    x_false = jnp.array(-1.3)
    y_false = f(False, x_false)
    x_false_rec = inv_f(False, y_false)

    assert jnp.allclose(x_true, x_true_rec, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(x_false, x_false_rec, atol=1e-6, rtol=1e-6)


def test_inverse_switch_with_known_branch_index():
    def f(index, x):
        return jax.lax.switch(
            index,
            [
                lambda t: t + 1.0,
                lambda t: jnp.exp(t),
                lambda t: 2.0 * t - 1.0,
            ],
            x,
        )

    inv_f = inverse(f, invertible_arg=1)

    x0 = jnp.array(0.2)
    y0 = f(jnp.array(0, dtype=jnp.int32), x0)
    x0_rec = inv_f(jnp.array(0, dtype=jnp.int32), y0)

    x1 = jnp.array(-0.4)
    y1 = f(jnp.array(1, dtype=jnp.int32), x1)
    x1_rec = inv_f(jnp.array(1, dtype=jnp.int32), y1)

    x2 = jnp.array(1.5)
    y2 = f(jnp.array(2, dtype=jnp.int32), x2)
    x2_rec = inv_f(jnp.array(2, dtype=jnp.int32), y2)

    assert jnp.allclose(x0, x0_rec, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(x1, x1_rec, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(x2, x2_rec, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_cond_with_known_branch():
    def f(pred, x):
        return jax.lax.cond(
            pred,
            lambda t: jnp.exp(t),
            lambda t: 3.0 * t - 1.0,
            x,
        )

    inv_and_det = inverse_and_logabsdet(f, invertible_arg=1)

    x_true = jnp.array(0.6)
    y_true = f(True, x_true)
    x_true_rec, logdet_true = inv_and_det(True, y_true)
    expected_true = -jnp.log(jnp.exp(x_true))

    x_false = jnp.array(-0.2)
    y_false = f(False, x_false)
    x_false_rec, logdet_false = inv_and_det(False, y_false)
    expected_false = -jnp.log(3.0)

    assert jnp.allclose(x_true, x_true_rec, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet_true, expected_true, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(x_false, x_false_rec, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet_false, expected_false, atol=1e-6, rtol=1e-6)


def test_inverse_scan_carry_only():
    def f(c0, xs):
        def body(carry, x):
            return carry + x, None

        carry_final, _ = jax.lax.scan(body, c0, xs)
        return carry_final

    c0 = jnp.array(0.7)
    xs = jnp.array([0.2, -0.1, 0.4, 1.0])
    y = f(c0, xs)

    inv_f = inverse(f, invertible_arg=0)
    c0_rec = inv_f(y, xs)
    assert jnp.allclose(c0, c0_rec, atol=1e-6, rtol=1e-6)


def test_inverse_scan_with_outputs_rule_level():
    def f(c0, xs):
        def body(carry, x):
            carry_new = carry + x
            y = 2.0 * carry_new
            return carry_new, y

        return jax.lax.scan(body, c0, xs)

    c0 = jnp.array(1.2)
    xs = jnp.array([0.1, 0.3, -0.2, 0.8])
    carry_final, ys = f(c0, xs)

    eqn = jax.make_jaxpr(f)(c0, xs).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.scan_p, Context.INVERSE)
    result = rule(
        eqn,
        [None, xs],
        [carry_final, ys],
    )

    assert result.resolved_vars == [eqn.invars[0]]
    assert jnp.allclose(result.resolved_vals[0], c0, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_scan_carry_only():
    scale = 1.7

    def f(c0, xs):
        def body(carry, x):
            return scale * carry + x, None

        carry_final, _ = jax.lax.scan(body, c0, xs)
        return carry_final

    c0 = jnp.array(-0.4)
    xs = jnp.array([0.3, -0.2, 0.5])
    y = f(c0, xs)

    inv_and_det = inverse_and_logabsdet(f, invertible_arg=0)
    c0_rec, logdet = inv_and_det(y, xs)
    expected = -xs.shape[0] * jnp.log(jnp.abs(scale))

    assert jnp.allclose(c0, c0_rec, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, expected, atol=1e-6, rtol=1e-6)


def test_inverse_while_rule_level():
    def f(i0, x0):
        def cond(state):
            i, _ = state
            return i < 4

        def body(state):
            i, x = state
            return i + 1, x + 2.0

        return jax.lax.while_loop(cond, body, (i0, x0))

    i0 = jnp.array(0, dtype=jnp.int32)
    x0 = jnp.array(1.1)
    out_i, out_x = f(i0, x0)

    eqn = jax.make_jaxpr(f)(i0, x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.while_p, Context.INVERSE)
    result = rule(
        eqn,
        [i0, None],
        [out_i, out_x],
    )

    assert result.resolved_vars == [eqn.invars[1]]
    assert jnp.allclose(result.resolved_vals[0], x0, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_while_rule_level():
    scale = 1.5

    def f(i0, x0):
        def cond(state):
            i, _ = state
            return i < 4

        def body(state):
            i, x = state
            return i + 1, scale * x

        return jax.lax.while_loop(cond, body, (i0, x0))

    i0 = jnp.array(0, dtype=jnp.int32)
    x0 = jnp.array(0.8)
    out_i, out_x = f(i0, x0)

    eqn = jax.make_jaxpr(f)(i0, x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.while_p, Context.INVERSE_LOGDET)
    result = rule(
        eqn,
        [i0, None],
        [out_i, out_x],
        context=None,
    )

    assert result.resolved_vars == [eqn.invars[1]]
    assert jnp.allclose(result.resolved_vals[0], x0, atol=1e-6, rtol=1e-6)

    expected = -(out_i - i0).astype(jnp.float32) * jnp.log(jnp.abs(scale))
    assert eqn.invars[1] in result.state
    assert jnp.allclose(result.state[eqn.invars[1]], expected, atol=1e-6, rtol=1e-6)
