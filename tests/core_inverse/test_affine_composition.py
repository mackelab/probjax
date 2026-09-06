"""Affine sections must compose as ordinary, compilable JAX programs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.core import inverse, inverse_and_logabsdet
from probjax.core.interpreters.inverse import recovery
from probjax.core.interpreters.inverse.affine import is_affine_in


def _shared(x):
    z = jnp.exp(x)
    return 3 * z - z


def _alternating(x):
    z = jnp.exp(3 * x - x)
    z = jnp.log(4 * z - z)
    return jnp.exp(2 * z + z)


def _internal_shapes(x):
    z = jnp.exp(x)
    tiled = jnp.broadcast_to(z, (2, *z.shape))
    return jnp.sum(tiled, axis=0) - z / 2


COMPOSITIONS = [
    lambda x: jnp.exp(3 * x - x),
    _shared,
    lambda x: 3 * jnp.exp(x) - jnp.exp(x),
    _alternating,
    _internal_shapes,
    lambda x: jnp.exp(jax.jit(lambda z: 3 * z - z)(x)),
    lambda x: jax.jit(_alternating)(x),
    lambda x: jnp.exp(jnp.sum(x) * jnp.ones_like(x) + x),
]


@pytest.mark.parametrize("fn", COMPOSITIONS)
@pytest.mark.parametrize("transform", [inverse, inverse_and_logabsdet])
def test_composition_round_trip_and_explicit_compile(fn, transform):
    x = jnp.array([0.1, 0.2])
    y = fn(x)
    inv = transform(fn)
    compiled = jax.jit(inv).lower(y).compile()
    for result in (inv(y), compiled(y)):
        recovered = result[0] if transform is inverse_and_logabsdet else result
        np.testing.assert_allclose(recovered, x, atol=2e-6)
        if transform is inverse_and_logabsdet:
            reference = -jnp.linalg.slogdet(jax.jacfwd(fn)(x))[1]
            np.testing.assert_allclose(result[1], reference, atol=3e-6)


@pytest.mark.parametrize("transform", [inverse, inverse_and_logabsdet])
def test_dynamic_coefficients_vmap_and_differentiation(transform):
    def forward(x, matrix, bias):
        z = jnp.exp(x)
        return jnp.log(matrix @ z + z + bias)

    matrix = jnp.array([[2.0, 0.2], [0.1, 3.0]])
    bias = jnp.array([0.4, 0.5])
    xs = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    inv = transform(forward)
    compiled = jax.jit(inv).lower(forward(xs[0], matrix, bias), matrix, bias).compile()
    # The executable receives new coefficients, not those captured during trace.
    for scale in (1.0, 1.5):
        m = scale * matrix
        ys = jax.vmap(forward, in_axes=(0, None, None))(xs, m, bias)
        one = compiled(ys[0], m, bias)
        batch = jax.jit(jax.vmap(inv, in_axes=(0, None, None)))(ys, m, bias)
        if transform is inverse_and_logabsdet:
            np.testing.assert_allclose(one[0], xs[0], atol=1e-6)
            np.testing.assert_allclose(batch[0], xs, atol=1e-6)
            refs = jax.vmap(
                lambda x, m=m: -jnp.linalg.slogdet(jax.jacfwd(forward)(x, m, bias))[1]
            )(xs)
            np.testing.assert_allclose(batch[1], refs, atol=1e-6)
        else:
            np.testing.assert_allclose(one, xs[0], atol=1e-6)
            np.testing.assert_allclose(batch, xs, atol=1e-6)

    def recovered(y, m):
        result = inv(y, m, bias)
        return result[0] if transform is inverse_and_logabsdet else result

    y = forward(xs[0], matrix, bias)
    reference = jnp.linalg.inv(jax.jacfwd(forward)(xs[0], matrix, bias))
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(recovered))(y, matrix), reference, atol=2e-6
    )
    reference_inverse = lambda y, m: jnp.log(
        jnp.linalg.solve(m + jnp.eye(2), jnp.exp(y) - bias)
    )
    np.testing.assert_allclose(
        jax.grad(lambda m: recovered(y, m).sum())(matrix),
        jax.grad(lambda m: reference_inverse(y, m).sum())(matrix),
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x + jnp.tanh(x),
        lambda x: jnp.exp(x) * jnp.exp(x),
        lambda x: 1 / (x + 1) + 2 * x,
        lambda x: x * x,
    ],
)
def test_nonlinear_residuals_do_not_enter_affine_solver(fn):
    x = jnp.array([0.1, 0.2])
    for transform in (inverse, inverse_and_logabsdet):
        result = jax.jit(transform(fn))(fn(x))
        for leaf in jax.tree_util.tree_leaves(result):
            assert jnp.all(jnp.isnan(leaf))


@pytest.mark.parametrize(
    "fn,dtype",
    [
        (lambda x: 1 / x, jnp.float32),
        (lambda x: jax.lax.dynamic_slice(jnp.arange(4.0), (x,), (1,)), jnp.int32),
        (lambda x: jax.lax.select_n(x, 1.0, 2.0), jnp.int32),
        (lambda x: x.astype(jnp.int32), jnp.float32),
    ],
)
def test_affinity_requires_independent_control_operands(fn, dtype):
    closed = jax.make_jaxpr(fn)(jnp.array(1, dtype=dtype))
    assert not is_affine_in(closed.jaxpr, closed.jaxpr.invars)


def test_known_nonlinear_coefficients_remain_affine():
    fn = lambda x, c: jnp.exp(c) * x + x / (c + 2)
    closed = jax.make_jaxpr(fn)(jnp.ones(2), jnp.array(0.2))
    assert is_affine_in(closed.jaxpr, closed.jaxpr.invars[:1])
    x = jnp.array([0.1, 0.2])
    np.testing.assert_allclose(inverse(fn)(fn(x, 0.2), 0.2), x, atol=1e-6)


@pytest.mark.parametrize("transform", [inverse, inverse_and_logabsdet])
@pytest.mark.parametrize(
    "fn,value",
    [
        (lambda x: jnp.expand_dims(x, 0), jnp.ones(2)),
        (lambda x: x.T, jnp.ones((2, 3))),
        (lambda x: x.reshape(4), jnp.ones((2, 2))),
        (lambda x: jnp.sum(x), jnp.ones(2)),
        (lambda x: {"different": x["value"]}, {"value": jnp.ones(2)}),
    ],
)
def test_boundary_mismatches_raise_before_inversion(transform, fn, value):
    with pytest.raises(ValueError, match="matching input/output"):
        transform(fn)(value)
    with pytest.raises(ValueError, match="matching input/output"):
        jax.jit(transform(fn)).lower(value)


def test_analysis_is_lazy_and_cached_without_dynamic_values(monkeypatch):
    builds = []
    original = recovery.normalize_inverse_graph

    def tracked(closed):
        builds.append(closed.jaxpr)
        return original(closed)

    monkeypatch.setattr(recovery, "normalize_inverse_graph", tracked)
    y = jnp.ones(2)
    inverse(lambda x: 2 * jnp.exp(x) + 1)(y)
    assert not builds

    fn = lambda x, c: jnp.exp(c * x - x)
    inv = inverse(fn)
    for c in (jnp.array(3.0), jnp.array(4.0)):
        np.testing.assert_allclose(inv(fn(y, c), c), y, atol=1e-6)
    assert len(builds) == 1
    # Reuse analysis under fresh traces without leaking the eager coefficients.
    jax.jit(inv).lower(y, jnp.array(5.0)).compile()
    assert len(builds) <= 2  # Weak-type differences may create another signature.


def test_single_section_size_and_multiple_sections_do_not_rescan(monkeypatch):
    visits = 0
    original = recovery.affine_equation

    def tracked(eqn, dependent):
        nonlocal visits
        visits += 1
        return original(eqn, dependent)

    monkeypatch.setattr(recovery, "affine_equation", tracked)
    for depth in (4, 16):

        def fn(x, depth=depth):
            for _ in range(depth):
                x = jnp.log(jnp.exp(1.1 * x - 0.1 * x))
            return x

        inv = inverse(fn)
        visits = 0
        result = jax.jit(inv)(jnp.array([0.1, 0.2]))
        np.testing.assert_allclose(result, jnp.array([0.1, 0.2]), atol=3e-6)
        assert visits <= 5 * depth


def test_value_only_recovery_does_not_compute_determinants(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("value-only inversion constructed a determinant")

    monkeypatch.setattr(jnp.linalg, "slogdet", forbidden)
    x = jnp.array([0.1, 0.2])
    fn = lambda x: jnp.exp(3 * x - x)
    np.testing.assert_allclose(jax.jit(inverse(fn))(fn(x)), x, atol=1e-6)


@pytest.mark.parametrize("transform", [inverse, inverse_and_logabsdet])
def test_singular_section_stays_nonfinite(transform):
    fn = lambda x: jnp.exp(x - x)
    result = jax.jit(transform(fn))(jnp.ones(2))
    for leaf in jax.tree_util.tree_leaves(result):
        assert jnp.all(~jnp.isfinite(leaf))


def test_coupled_multiple_leaf_affine_fallback_is_preserved():
    def fn(pair):
        x, z = pair
        return (2 * x + z, x - 3 * z)

    x = (jnp.array(0.2), jnp.array(0.4))
    y = fn(x)
    recovered, logdet = jax.jit(inverse_and_logabsdet(fn))(y)
    np.testing.assert_allclose(recovered, x, atol=1e-6)
    np.testing.assert_allclose(logdet, -jnp.log(7.0), atol=1e-6)


def test_custom_inverse_contract_between_affine_sections():
    from probjax.core import custom_inverse

    @custom_inverse
    def nonlinear(x):
        return jnp.exp(x)

    nonlinear.definv(jnp.log)
    nonlinear.definv_and_logdet(lambda y: (jnp.log(y), -jnp.log(y).sum()))

    def fn(x):
        z = nonlinear(3 * x - x)
        return 3 * z - z

    x = jnp.array([0.1, 0.2])
    recovered, logdet = jax.jit(inverse_and_logabsdet(fn))(fn(x))
    np.testing.assert_allclose(recovered, x, atol=1e-6)
    reference = -jnp.linalg.slogdet(jax.jacfwd(fn)(x))[1]
    np.testing.assert_allclose(logdet, reference, atol=1e-6)


def test_repeated_jit_body_with_different_inputs_is_not_merged():
    block = jax.jit(lambda x: 3 * x - x)

    def fn(x):
        return jnp.log(block(jnp.exp(block(x))))

    x = jnp.array([0.1, 0.2])
    recovered, logdet = jax.jit(inverse_and_logabsdet(fn))(fn(x))
    np.testing.assert_allclose(recovered, x, atol=1e-6)
    np.testing.assert_allclose(logdet, -2 * jnp.log(2.0), atol=1e-6)


def test_normalization_preserves_custom_inverse_calls():
    from probjax.core import custom_inverse
    from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p

    @custom_inverse
    def nonlinear(x):
        return jnp.exp(x)

    nonlinear.definv(jnp.log)
    nonlinear.definv_and_logdet(lambda y: (jnp.log(y), -jnp.log(y).sum()))
    closed = jax.make_jaxpr(lambda x: nonlinear(x) + nonlinear(x))(jnp.ones(2))
    normalized, _ = recovery.normalize_inverse_graph(closed)
    assert sum(e.primitive is custom_inverse_call_p for e in normalized.eqns) == 2
