"""Dense-oracle regressions for operator composition and compiled filtering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.utils.linear_operator import LinearOperator
from probjax.inference.filtering.kalman_filter import (
    default_logdet,
    default_solve,
    kalman_filter,
)


@pytest.mark.parametrize("compiled", [False, True])
def test_rectangular_products(compiled):
    a = jnp.array([[1.0, 2.0, 3.0], [-1.0, 4.0, 2.0]])
    right = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4)
    left = jnp.array([[2.0, -1.0], [1.0, 3.0], [0.0, 4.0]])
    x, y = jnp.array([1.0, -2.0, 4.0]), jnp.array([2.0, 3.0])

    def products(a, right, left, x, y):
        op = LinearOperator.from_array(a)
        assert op.shape == a.shape
        assert op.T.shape == a.T.shape
        return (
            op @ x,
            y @ op,
            (op @ right).as_array(),
            (left @ op).as_array(),
            (op @ LinearOperator.from_array(right)).as_array(),
        )

    actual = (jax.jit(products) if compiled else products)(a, right, left, x, y)
    expected = (a @ x, y @ a, a @ right, left @ a, a @ right)
    for result, oracle in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, oracle, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("observed", [False, True])
def test_compiled_operator_filter_matches_dense(observed):
    a = jnp.array([[0.9, 0.3], [-0.1, 0.8]])
    q = jnp.array([[0.1, 0.02], [0.02, 0.2]])
    c = jnp.array([[1.0, 2.0]])
    r = jnp.array([[0.3]])
    mean = jnp.array([1.0, -0.5])
    cov = jnp.array([[0.7, 0.2], [0.2, 1.3]])
    y = jnp.array([0.8]) if observed else None

    def run(a, operator):
        convert = LinearOperator.from_array if operator else lambda x: x
        kernel = kalman_filter(
            lambda t0, t: (convert(a), convert(q)),
            lambda t: (convert(c), r),
        )
        return kernel.step(kernel.init(mean, cov, t=0.0), t=1.0, observed=y)

    dense = jax.jit(lambda a: run(a, False))(a)
    actual = jax.jit(lambda a: run(a, True))(a)
    for result, oracle in zip(
        jax.tree.leaves(actual), jax.tree.leaves(dense), strict=True
    ):
        np.testing.assert_allclose(result, oracle, rtol=2e-5, atol=2e-6)
    loss = lambda a, operator: jnp.sum(run(a, operator)[0].cov)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda a: loss(a, True)))(a),
        jax.jit(jax.grad(lambda a: loss(a, False)))(a),
        rtol=2e-5,
        atol=2e-6,
    )


# ---------------------------------------------------------------------------
# P1-5: dense-oracle coverage for the LinearOperator algebra surface
# ---------------------------------------------------------------------------


@pytest.fixture
def square_pair():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.5, -1.0], [2.0, 0.25]])
    return (
        LinearOperator.from_array(jnp.array(a)),
        LinearOperator.from_array(jnp.array(b)),
        a,
        b,
    )


def test_scalar_arithmetic_vs_dense(square_pair):
    A, _, a, _ = square_pair
    eye = np.eye(2)
    cases = {
        "2*A": (2 * A, 2 * a),
        "A*2": (A * 2, a * 2),
        "A/2": (A / 2, a / 2),
        "A+1.0": (A + 1.0, a + eye),  # 0-d acts as c * I
        "A-1.0": (A - 1.0, a - eye),
        "1.0+A": (1.0 + A, eye + a),
        "1.0-A": (1.0 - A, eye - a),
        "-A": (-A, -a),
        "row scaling": (A * jnp.array([1.0, 2.0]), a * np.array([1.0, 2.0])[:, None]),
    }
    for name, (op, oracle) in cases.items():
        np.testing.assert_allclose(op.as_array(), oracle, rtol=1e-6, err_msg=name)
        assert op.shape == a.shape, name
        assert op.dtype == jnp.dtype("float32"), name


def test_add_sub_operator_vs_dense(square_pair):
    A, B, a, b = square_pair
    np.testing.assert_allclose((A + B).as_array(), a + b, rtol=1e-6)
    np.testing.assert_allclose((A - B).as_array(), a - b, rtol=1e-6)
    np.testing.assert_allclose((A + b).as_array(), a + b, rtol=1e-6)
    # Note: ``b - A`` with an ndarray left operand does NOT dispatch to
    # __rsub__ (NumPy/JAX consume the operation first); the 2-d branch is
    # exercised via a direct call.
    np.testing.assert_allclose(A.__rsub__(jnp.array(b)).as_array(), b - a, rtol=1e-6)


def test_transpose_algebra_vs_dense(square_pair):
    A, B, a, b = square_pair
    np.testing.assert_allclose((A + B).T.as_array(), (a + b).T, rtol=1e-6)
    np.testing.assert_allclose((A @ B).T.as_array(), (a @ b).T, rtol=1e-6)
    np.testing.assert_allclose((B.T @ A.T).as_array(), (a @ b).T, rtol=1e-6)
    np.testing.assert_allclose(A.T.T.as_array(), a, rtol=1e-6)
    assert A.T is A.T  # .T is a cached_property


def test_non_square_as_array_and_transpose():
    m = np.array([[1.0, 2.0, 3.0], [-1.0, 4.0, 2.0]])
    op = LinearOperator.from_array(jnp.array(m))
    np.testing.assert_allclose(op.as_array(), m, rtol=1e-6)
    assert op.shape == (2, 3)
    assert op.T.shape == (3, 2)
    np.testing.assert_allclose(op.T.as_array(), m.T, rtol=1e-6)
    np.testing.assert_allclose(op.T.T.as_array(), m, rtol=1e-6)


def test_construction_time_validation(square_pair):
    A, B, a, _ = square_pair
    C = LinearOperator.from_array(jnp.ones((3, 3)))
    with pytest.raises(ValueError):
        A + C  # shape mismatch between operators
    with pytest.raises(ValueError):
        A + jnp.ones((3, 3))  # wrong dense shape (was a deferred XLA TypeError)
    with pytest.raises(ValueError):
        A + jnp.ones(3)  # 1-d arrays are not valid addends
    with pytest.raises(ValueError):
        A - jnp.ones((2, 3))
    with pytest.raises(ValueError):
        A * jnp.ones((2, 2))  # only scalar or (out_dim,) factors
    with pytest.raises(ValueError):
        A * jnp.ones(3)  # wrong-length vector factor
    with pytest.raises(ValueError):
        A / jnp.ones((2, 2))
    with pytest.raises(ValueError):
        A @ C  # incompatible matmul shapes
    # Valid shapes still pass.
    assert (A + jnp.ones((2, 2))).shape == (2, 2)
    assert (A * jnp.ones(2)).shape == (2, 2)
    assert (A / 2.0).shape == (2, 2)
    assert (B - a).shape == (2, 2)


def test_dtype_truthfulness_x64():
    # Regression for the P1-1 bug: arithmetic must report the promoted dtype,
    # matching what as_array() actually materializes.
    with jax.enable_x64():
        A = LinearOperator.from_array(
            jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        )
        assert A.dtype == jnp.dtype("float32")
        for op in (A * np.float64(2.0), A + np.float64(1.0), A - np.float64(1.0),
                   2.0 * A, A / np.float64(2.0)):
            assert op.dtype == jnp.dtype("float64"), op.dtype
            assert op.as_array().dtype == jnp.dtype("float64")
        # Negation never promotes.
        assert (-A).dtype == jnp.dtype("float32")
        # Unpromoted ops keep float32.
        assert (A * 2).dtype == jnp.dtype("float32")
        assert (A + A).dtype == jnp.dtype("float32")
        # dtype=None probes the callable once at construction.
        inferred = LinearOperator(lambda x: jnp.float64(1.0) * x, 2, 2)
        assert inferred.dtype == jnp.dtype("float64")
        assert inferred.as_array().dtype == jnp.dtype("float64")
        # Probing falls back to the global default when unsafe; the attribute
        # is then assumed, not inferred.
        def bad(x):
            raise RuntimeError("cannot probe me")

        assumed = LinearOperator(bad, 2, 2)
        assert assumed.dtype == jnp.result_type(jnp.ones(1))


def test_pytree_roundtrip(square_pair):
    A, _, a, _ = square_pair
    leaves, treedef = jax.tree.flatten(A)
    B = jax.tree.unflatten(treedef, leaves)
    assert B.shape == A.shape
    assert B.dtype == A.dtype
    np.testing.assert_allclose(B.as_array(), a, rtol=1e-6)
    # Each unflatten creates a fresh instance with its own .T cache.
    assert B.T is not A.T
    np.testing.assert_allclose(B.T.as_array(), a.T, rtol=1e-6)


def test_jit_grad_through_add_mul():
    def loss(a):
        A = LinearOperator.from_array(a)
        return jnp.sum(((A + A) * 2.0).as_array())

    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    grad = jax.jit(jax.grad(loss))(a)
    np.testing.assert_allclose(grad, 4.0 * jnp.ones_like(a), rtol=1e-6)


def _spd_fixture():
    m = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    return LinearOperator.from_array(m), m


def test_solve_vs_dense():
    S, m = _spd_fixture()
    rhs = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = jnp.linalg.solve(m, rhs)
    np.testing.assert_allclose(S.solve(rhs, method="dense"), oracle, rtol=1e-6)
    np.testing.assert_allclose(
        S.solve(rhs, method="dense", assume_a="pos"), oracle, rtol=1e-6
    )
    # 1-d rhs.
    np.testing.assert_allclose(
        S.solve(rhs[:, 0], method="dense"), jnp.linalg.solve(m, rhs[:, 0]), rtol=1e-6
    )
    # Iterative path (forced via dense_mem_limit=0) agrees with dense.
    np.testing.assert_allclose(
        S.solve(rhs, dense_mem_limit=0), oracle, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        S.solve(rhs[:, 0], dense_mem_limit=0),
        jnp.linalg.solve(m, rhs[:, 0]),
        rtol=1e-3,
        atol=1e-4,
    )
    with pytest.raises(ValueError):
        S.solve(rhs, method="bogus")
    with pytest.raises(ValueError):
        LinearOperator.from_array(jnp.ones((2, 3))).solve(rhs)


def test_logdet_vs_dense():
    S, m = _spd_fixture()
    oracle = jnp.linalg.slogdet(m).logabsdet
    np.testing.assert_allclose(S.logdet(method="dense"), oracle, rtol=1e-6)
    # Iterative path (forced via dense_mem_limit=0); full probe basis here,
    # so the Lanczos estimate is exact up to floating-point error.
    np.testing.assert_allclose(S.logdet(dense_mem_limit=0), oracle, rtol=1e-3)
    with pytest.raises(ValueError):
        S.logdet(method="bogus")
    with pytest.raises(ValueError):
        LinearOperator.from_array(jnp.ones((2, 3))).logdet()


def test_default_solve_logdet_wrappers_backward_compat():
    # default_solve/default_logdet keep their (S, res, dense_mem_limit)
    # signatures and agree with the new LinearOperator methods.
    S, m = _spd_fixture()
    res = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # (nrhs, obs_dim) KF convention
    np.testing.assert_allclose(
        default_solve(S, res), S.solve(res.T, assume_a="pos").T, rtol=1e-6
    )
    np.testing.assert_allclose(
        default_solve(S, res, dense_mem_limit=0),
        S.solve(res.T, dense_mem_limit=0).T,
        rtol=1e-3,
        atol=1e-4,
    )
    # Plain-array S still takes the dense path.
    np.testing.assert_allclose(
        default_solve(m, res), jax.scipy.linalg.solve(m, res.T, assume_a="pos").T,
        rtol=1e-6,
    )
    np.testing.assert_allclose(default_logdet(S), S.logdet(), rtol=1e-6)
    np.testing.assert_allclose(
        default_logdet(S, dense_mem_limit=0), S.logdet(dense_mem_limit=0), rtol=1e-3
    )
    np.testing.assert_allclose(
        default_logdet(m), jnp.linalg.slogdet(m).logabsdet, rtol=1e-6
    )
