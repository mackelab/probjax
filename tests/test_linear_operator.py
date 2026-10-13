"""Dense-oracle regressions for structured linear operators and filtering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.inference.filtering.kalman_filter import (
    default_logdet,
    default_solve,
    kalman_filter,
)
from probjax.utils.linear_operator import (
    Dense,
    Diagonal,
    Function,
    Jacobian,
    LinearOperator,
    logdet,
    solve,
)

RNG = np.random.default_rng(0)
SQUARE = RNG.normal(size=(4, 4)).astype(np.float32)
SPD = (SQUARE @ SQUARE.T + 4 * np.eye(4)).astype(np.float32)
DIAG = np.array([1.0, 2.0, -0.5, 3.0], dtype=np.float32)


def _as_function(op: LinearOperator) -> Function:
    return Function(op.mv, op.in_shape)


def _operators(matrix):
    """The same matrix as Dense and as matrix-free Function."""
    dense = Dense(jnp.asarray(matrix))
    return {"dense": dense, "function": _as_function(dense)}


# ---------------------------------------------------------------------------
# Algebra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["dense", "function"])
def test_algebra_vs_dense(kind):
    A = _operators(SQUARE)[kind]
    B = Dense(jnp.asarray(SPD))
    D = Diagonal(jnp.asarray(DIAG))
    a, b, d = SQUARE, SPD, np.diag(DIAG)
    eye = np.eye(4)
    cases = {
        "A+B": (A + B, a + b),
        "A-B": (A - B, a - b),
        "A+D": (A + D, a + d),
        "A+array": (A + b, a + b),
        "A+1": (A + 1.0, a + eye),  # scalars act as c * I
        "1-A": (1.0 - A, eye - a),
        "2*A": (2 * A, 2 * a),
        "A/2": (A / 2, a / 2),
        "-A": (-A, -a),
        "A@B": (A @ B, a @ b),
        "D@A": (D @ A, d @ a),
        "A@D": (A @ D, a @ d),
        "A.T": (A.T, a.T),
        "(A@B).T": ((A @ B).T, (a @ b).T),
        "A.T.T": (A.T.T, a),
    }
    for name, (op, oracle) in cases.items():
        np.testing.assert_allclose(op.as_array(), oracle, rtol=1e-5, err_msg=name)
        assert op.shape == (4, 4), name


def test_structure_is_preserved_where_cheap():
    A = Dense(jnp.asarray(SQUARE))
    D = Diagonal(jnp.asarray(DIAG))
    F = _as_function(A)
    assert isinstance(D + D, Diagonal)
    assert isinstance(D @ D, Diagonal)
    assert isinstance(2.0 * D, Diagonal)
    assert isinstance(A + D, Dense)
    assert isinstance(A @ D, Dense)
    assert isinstance(D @ A, Dense)
    assert isinstance(A - 1.0, Dense)
    assert isinstance(A @ A, Dense)
    # Anything involving a matrix-free operator stays matrix-free.
    for op in (F + A, F @ A, A @ F, F.T, 2.0 * F):
        assert not isinstance(op, (Dense, Diagonal))


def test_scalar_diagonal():
    c = Diagonal(jnp.asarray(2.0), 3)
    assert c.in_shape == (3,)
    np.testing.assert_allclose(c.as_array(), 2.0 * np.eye(3))
    np.testing.assert_allclose(c @ jnp.ones(3), 2.0 * np.ones(3))


def test_vector_and_matrix_products():
    a = jnp.array([[1.0, 2.0, 3.0], [-1.0, 4.0, 2.0]])
    right = jnp.arange(12.0).reshape(3, 4)
    left = jnp.array([[2.0, -1.0], [1.0, 3.0], [0.0, 4.0]])
    x, y = jnp.array([1.0, -2.0, 4.0]), jnp.array([2.0, 3.0])
    for op in _operators(a).values():
        assert op.shape == (2, 3) and op.T.shape == (3, 2)
        np.testing.assert_allclose(op @ x, a @ x, rtol=1e-6)
        np.testing.assert_allclose(y @ op, y @ a, rtol=1e-6)
        np.testing.assert_allclose((op @ right).as_array(), a @ right, rtol=1e-6)
        np.testing.assert_allclose((left @ op).as_array(), left @ a, rtol=1e-6)
        # mv with a trailing batch axis applies column-wise.
        np.testing.assert_allclose(op.mv(right), a @ right, rtol=1e-6)


def test_arbitrary_in_and_out_shapes():
    m = jnp.asarray(RNG.normal(size=(4, 6)), dtype=jnp.float32)
    x = jnp.asarray(RNG.normal(size=(2, 3)), dtype=jnp.float32)
    dense = Dense(m, in_shape=(2, 3), out_shape=(4,))
    function = Function(lambda v: m @ v.ravel(), (2, 3))
    for op in (dense, function):
        assert (op.in_shape, op.out_shape, op.shape) == ((2, 3), (4,), (4, 6))
        np.testing.assert_allclose(op @ x, m @ x.ravel(), rtol=1e-5)
        np.testing.assert_allclose(op.as_array(), m, rtol=1e-6)
        assert (op.T.in_shape, op.T.out_shape) == ((4,), (2, 3))
        np.testing.assert_allclose(
            op.T @ jnp.ones(4), (m.T @ jnp.ones(4)).reshape(2, 3), rtol=1e-5
        )
        gram = op.T @ op
        assert (gram.in_shape, gram.out_shape) == ((2, 3), (2, 3))
        np.testing.assert_allclose(gram.as_array(), m.T @ m, rtol=1e-4)
        batch = jnp.stack([x, 2 * x], axis=-1)
        np.testing.assert_allclose(
            op.mv(batch), jnp.stack([m @ x.ravel(), 2 * m @ x.ravel()], -1), rtol=1e-5
        )

    jac = jax.jacfwd(lambda v: jnp.sin(v).sum(axis=0))(x)
    for op in (
        Dense.jacfwd(lambda v: jnp.sin(v).sum(axis=0), x),
        Jacobian(lambda v: jnp.sin(v).sum(axis=0), x),
    ):
        assert (op.in_shape, op.out_shape) == ((2, 3), (3,))
        np.testing.assert_allclose(op.as_array(), jac.reshape(3, 6), rtol=1e-6)


def test_validation():
    A = Dense(jnp.asarray(SQUARE))
    C = Dense(jnp.ones((3, 3)))
    rectangular = Dense(jnp.ones((2, 3)))
    for bad in (
        lambda: A + C,
        lambda: A + jnp.ones((3, 3)),
        lambda: A + jnp.ones(4),
        lambda: A * jnp.ones(4),  # only scalar scaling
        lambda: A / jnp.ones(4),
        lambda: A @ C,
        lambda: A @ jnp.ones(3),
        lambda: rectangular + 1.0,  # c * I needs a square operator
        lambda: Function(lambda x: x[:2], 3, out_shape=(3,)),
        lambda: solve(rectangular, jnp.ones(2)),
        lambda: solve(A, jnp.ones(3)),
        lambda: solve(A, jnp.ones(4), solver="bogus"),
        lambda: logdet(rectangular),
        lambda: logdet(_as_function(A)),  # matrix-free logdet needs psd
        lambda: _as_function(A).expm(),
    ):
        with pytest.raises((ValueError, NotImplementedError)):
            bad()


def test_function_never_runs_matvec_to_infer_shape():
    calls = []

    def matvec(x):
        calls.append(type(x).__name__)
        return 2.0 * x

    op = Function(matvec, 3)
    assert (op.out_shape, op.dtype) == ((3,), jnp.result_type(float))
    assert all("Tracer" in name for name in calls)


def test_dtype_promotion_x64():
    with jax.enable_x64():
        A = Dense(jnp.asarray(SQUARE))
        assert A.dtype == jnp.float32
        for op in (A * np.float64(2.0), A + np.float64(1.0), A / np.float64(2.0)):
            assert op.dtype == jnp.float64
            assert op.as_array().dtype == jnp.float64
        assert (-A).dtype == jnp.float32
        assert (A + A).dtype == jnp.float32


# ---------------------------------------------------------------------------
# Pytrees and transformations
# ---------------------------------------------------------------------------


def test_pytree_leaves_are_the_arrays():
    A, D = Dense(jnp.asarray(SQUARE)), Diagonal(jnp.asarray(DIAG))
    assert len(jax.tree.leaves(A)) == 1 and len(jax.tree.leaves(D)) == 1
    assert jax.tree.leaves(_as_function(A)) == []
    leaves, treedef = jax.tree.flatten(A)
    B = jax.tree.unflatten(treedef, leaves)
    np.testing.assert_allclose(B.as_array(), SQUARE)


def test_rebuilt_operators_do_not_retrace():
    traces = []

    @jax.jit
    def apply(op, x):
        traces.append(None)
        return op @ x

    def matvec(v):
        return 2.0 * v

    def fn(v, scale):
        return scale * jnp.sin(v)

    x = jnp.ones(3)
    builders = {
        "sum": lambda m: Function(matvec, 3) + Dense(m),
        "product": lambda m: Dense(m) @ Function(matvec, 3),
        "transpose": lambda m: (Function(matvec, 3) @ Dense(m)).T,
        "jacobian": lambda m: Jacobian(fn, m[0], (m[1, 1],)),
    }
    for name, build in builders.items():
        traces.clear()
        for scale in (1.0, 2.0, 3.0):
            out = apply(build(scale * jnp.eye(3)), x)
        assert len(traces) == 1, name
        # Arrays inside composites are traced inputs, not baked-in constants.
        np.testing.assert_allclose(out, build(3.0 * jnp.eye(3)) @ x, rtol=1e-6)


def test_jacobian_and_transpose_match_dense():
    x = jnp.array([0.3, -1.2, 2.0])

    def fn(v, t):
        return jnp.stack([t * v[0] * v[1], jnp.sin(v[2]), v.sum()])

    jac = jax.jacfwd(fn)(x, 1.5)
    J = Jacobian(fn, x, (1.5,))
    np.testing.assert_allclose(J.as_array(), jac, rtol=1e-6)
    np.testing.assert_allclose(J.T.as_array(), jac.T, rtol=1e-6)
    np.testing.assert_allclose((J.T @ J).as_array(), jac.T @ jac, rtol=1e-5)
    grad = jax.grad(lambda t: jnp.sum(Jacobian(fn, x, (t,)) @ jnp.ones(3)))(1.5)
    np.testing.assert_allclose(grad, x[0] + x[1], rtol=1e-6)


def test_jit_grad_and_vmap_through_operators():
    def loss(a):
        A = Dense(a)
        return jnp.sum(((A + A) * 2.0).as_array())

    a = jnp.asarray(SQUARE)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(a), 4.0 * np.ones((4, 4)))

    # Operators cross jit boundaries as pytrees, with arrays traced not baked in.
    apply = jax.jit(lambda op, x: op @ x)
    np.testing.assert_allclose(apply(Dense(a), jnp.ones(4)), a @ jnp.ones(4), rtol=1e-6)

    batch = jnp.stack([a, 2 * a])
    out = jax.vmap(lambda m: Dense(m) @ jnp.ones(4))(batch)
    np.testing.assert_allclose(out[1], 2 * (a @ jnp.ones(4)), rtol=1e-6)


# ---------------------------------------------------------------------------
# Solve and logdet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, solver, matrix, psd",
    [
        ("dense", "auto", SQUARE, False),
        ("dense", "auto", SPD, True),
        ("dense", "lu", SQUARE, False),
        ("dense", "cholesky", SPD, True),
        ("function", "auto", SPD, True),
        ("function", "auto", SQUARE, False),
        ("function", "cg", SPD, True),
        ("function", "gmres", SQUARE, False),
        ("function", "bicgstab", SPD, False),
    ],
)
def test_solve_vs_dense(kind, solver, matrix, psd):
    op = _operators(matrix)[kind]
    rhs = jnp.asarray(RNG.normal(size=(4, 3)), dtype=jnp.float32)
    oracle = np.linalg.solve(matrix.astype(np.float64), np.asarray(rhs, np.float64))
    kwargs = dict(solver=solver, psd=psd, tol=1e-7)
    np.testing.assert_allclose(solve(op, rhs, **kwargs), oracle, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(
        op.solve(rhs[:, 0], **kwargs), oracle[:, 0], rtol=2e-3, atol=2e-4
    )


def test_solve_diagonal_and_shaped():
    D = Diagonal(jnp.asarray(DIAG).reshape(2, 2))
    b = jnp.ones((2, 2))
    np.testing.assert_allclose(solve(D, b), 1 / DIAG.reshape(2, 2), rtol=1e-6)
    S = Dense(jnp.asarray(SPD), in_shape=(2, 2), out_shape=(2, 2))
    rhs = jnp.arange(8.0).reshape(2, 2, 2)
    oracle = np.linalg.solve(SPD, np.asarray(rhs).reshape(4, 2)).reshape(2, 2, 2)
    for op in (S, _as_function(S)):
        np.testing.assert_allclose(
            solve(op, rhs, psd=True, tol=1e-7), oracle, rtol=1e-3, atol=1e-4
        )


@pytest.mark.parametrize("solver", ["auto", "cg"])
def test_solve_gradient_matches_dense(solver):
    def loss(a, operator):
        s = a @ a.T + 4 * jnp.eye(4)
        rhs = jnp.arange(4.0)
        if operator:
            return jnp.sum(solve(Dense(s), rhs, solver, psd=True, tol=1e-8) ** 2)
        return jnp.sum(jnp.linalg.solve(s, rhs) ** 2)

    a = jnp.asarray(SQUARE)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda a: loss(a, True)))(a),
        jax.grad(lambda a: loss(a, False))(a),
        rtol=1e-3,
        atol=1e-4,
    )


def test_cholesky_gradient_is_symmetric():
    # Cholesky reads one triangle; the gradient must not depend on which.
    grads = [
        jax.grad(lambda m, fn=fn: jnp.sum(fn(Dense(m))))(jnp.asarray(SPD))
        for fn in (
            lambda op: solve(op, jnp.arange(4.0), "cholesky"),
            lambda op: logdet(op, psd=True),
        )
    ]
    for grad in grads:
        np.testing.assert_allclose(grad, grad.T, rtol=1e-5, atol=1e-6)


def test_logdet_vs_slogdet():
    oracle = np.linalg.slogdet(SPD.astype(np.float64))[1]
    S = Dense(jnp.asarray(SPD))
    np.testing.assert_allclose(logdet(S), oracle, rtol=1e-5)
    np.testing.assert_allclose(logdet(S, psd=True), oracle, rtol=1e-5)
    # A full probe basis makes the Lanczos estimate exact.
    np.testing.assert_allclose(logdet(_as_function(S), psd=True), oracle, rtol=1e-4)
    np.testing.assert_allclose(
        Diagonal(jnp.asarray(DIAG)).logdet(), np.sum(np.log(np.abs(DIAG))), rtol=1e-6
    )
    np.testing.assert_allclose(
        logdet(Diagonal(jnp.asarray(2.0), 3)), 3 * np.log(2.0), rtol=1e-6
    )


def test_lineax_solvers():
    lx = pytest.importorskip("lineax")
    rhs = jnp.ones((4, 2))
    for op in (*_operators(SPD).values(), Diagonal(jnp.asarray(DIAG))):
        oracle = np.linalg.solve(op.as_array(), rhs)
        for solver in (lx.AutoLinearSolver(well_posed=True), lx.LU()):
            np.testing.assert_allclose(
                solve(op, rhs, solver), oracle, rtol=1e-4, atol=1e-5
            )
    cg = solve(_operators(SPD)["function"], rhs, lx.CG(rtol=1e-7, atol=1e-7), psd=True)
    np.testing.assert_allclose(cg, np.linalg.solve(SPD, rhs), rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Matrix functions
# ---------------------------------------------------------------------------


def test_expm_and_phi():
    with jax.enable_x64():
        diag = jnp.array([-1.0, 1e-9, 0.5])
        t = 0.7
        z = t * diag
        phi1 = np.where(np.abs(z) > 1e-6, np.expm1(z) / z, 1.0 + z / 2)
        phi2 = np.where(np.abs(z) > 1e-6, (np.expm1(z) - z) / z**2, 0.5 + z / 6)
        for op in (Diagonal(diag), Dense(jnp.diag(diag))):
            phis = [p.as_array() for p in op.phi(t, k=2)]
            np.testing.assert_allclose(np.diag(phis[0]), np.exp(z), rtol=1e-10)
            np.testing.assert_allclose(np.diag(phis[1]), phi1, rtol=1e-8)
            np.testing.assert_allclose(np.diag(phis[2]), phi2, rtol=1e-8)
            np.testing.assert_allclose(
                np.diag(op.expm(t).as_array()), np.exp(z), rtol=1e-10
            )
        k1 = Diagonal(diag).phi(t, k=1)
        np.testing.assert_allclose(k1[1].diag, phi1, rtol=1e-8)

        # Non-diagonal: dt * phi_1(A dt) b is the exact affine-ODE increment.
        a = jnp.array([[-1.0, 2.0], [0.0, -3.0]])
        b = jnp.array([1.0, -2.0])
        aug = jnp.zeros((3, 3)).at[:2, :2].set(a).at[:2, 2].set(b)
        exact = jax.scipy.linalg.expm(t * aug)[:2, 2]
        np.testing.assert_allclose(t * (Dense(a).phi(t)[1] @ b), exact, rtol=1e-10)


# ---------------------------------------------------------------------------
# Kalman filter integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("observed", [False, True])
def test_compiled_operator_filter_matches_dense(observed):
    a = jnp.array([[0.9, 0.3], [-0.1, 0.8]])
    q = jnp.array([[0.1, 0.02], [0.02, 0.2]])
    c = jnp.array([[1.0, 2.0]])
    r = jnp.array([[0.3]])
    mean = jnp.array([1.0, -0.5])
    cov = jnp.array([[0.7, 0.2], [0.2, 1.3]])
    y = jnp.array([0.8]) if observed else None

    def run(a, convert):
        kernel = kalman_filter(
            lambda t0, t: (convert(a), convert(q)),
            lambda t: (convert(c), r),
        )
        return kernel.step(kernel.init(mean, cov, t=0.0), t=1.0, observed=y)

    as_function = lambda m: _as_function(Dense(m))
    dense = jax.jit(lambda a: run(a, lambda m: m))(a)
    for convert in (Dense, as_function):
        actual = jax.jit(lambda a, convert=convert: run(a, convert))(a)
        for result, oracle in zip(
            jax.tree.leaves(actual), jax.tree.leaves(dense), strict=True
        ):
            np.testing.assert_allclose(result, oracle, rtol=2e-5, atol=2e-6)
    loss = lambda a, convert: jnp.sum(run(a, convert)[0].cov)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda a: loss(a, as_function)))(a),
        jax.jit(jax.grad(lambda a: loss(a, lambda m: m)))(a),
        rtol=2e-5,
        atol=2e-6,
    )


def test_default_solve_and_logdet():
    m = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    res = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # (nrhs, obs_dim) KF convention
    oracle = np.linalg.solve(m, res.T).T
    for S in (Dense(m), _as_function(Dense(m))):
        np.testing.assert_allclose(default_solve(S, res), oracle, rtol=1e-5)
        # dense_mem_limit=0 forces the matrix-free path for Function operators.
        np.testing.assert_allclose(
            default_solve(S, res, dense_mem_limit=0), oracle, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            default_logdet(S, dense_mem_limit=0),
            np.linalg.slogdet(m)[1],
            rtol=1e-4,
        )
    np.testing.assert_allclose(default_solve(m, res), oracle, rtol=1e-5)
    np.testing.assert_allclose(default_logdet(m), np.linalg.slogdet(m)[1], rtol=1e-6)
