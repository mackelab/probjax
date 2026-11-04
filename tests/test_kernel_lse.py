"""Streaming log-domain kernel reduction ``kernel_logsumexp`` vs dense references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.pallas_kernels import kernel_logsumexp, kernel_logsumexp_dense


def _requires_accelerator():
    if jax.default_backend() == "cpu":
        pytest.skip("requires an accelerator backend")


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def _problem(n, m, d, o, seed=0):
    k = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.normal(k[0], (n, d))
    y = jax.random.normal(k[1], (m, d)) * 0.8 + 0.3
    lw = jax.random.normal(k[2], (m,))
    v = jax.random.normal(k[3], (m, o)) if o else None
    return x, y, lw, v


def _loss_and_grads(fn, x, y, lw, v, eps):
    def loss(x, y, lw, v, eps):
        r = fn(x, y, values=v, log_weights=lw, epsilon=eps)
        if v is None:
            return jnp.sum(jnp.sin(r)), r
        lse, out = r
        return jnp.sum(jnp.sin(lse)) + jnp.sum(jnp.cos(out)), r

    argnums = (0, 1, 2, 3, 4) if v is not None else (0, 1, 2, 4)
    (_, r), g = jax.jit(jax.value_and_grad(loss, argnums=argnums, has_aux=True))(
        x, y, lw, v, eps
    )
    return jax.tree.leaves((r, g))


SHAPES = [
    (37, 53, 3, 2, 16, 16),  # odd sizes, several key blocks
    (300, 517, 5, 3, 64, 128),
    (128, 64, 2, None, 64, 64),  # no values
    (1, 7, 1, 1, 16, 16),
]


def _check(impl, n, m, d, o, bq, bk, tol):
    x, y, lw, v = _problem(n, m, d, o)
    eps = jnp.float32(0.3)
    got = _loss_and_grads(
        lambda *a, **kw: kernel_logsumexp(*a, impl=impl, block_q=bq, block_k=bk, **kw),
        x,
        y,
        lw,
        v,
        eps,
    )
    want = _loss_and_grads(kernel_logsumexp_dense, x, y, lw, v, eps)
    for a, b in zip(got, want, strict=True):
        assert _rel(a, b) < tol


@pytest.mark.parametrize("n, m, d, o, bq, bk", SHAPES)
def test_xla_matches_dense_values_and_grads(n, m, d, o, bq, bk):
    _check("xla", n, m, d, o, bq, bk, 1e-5)


@pytest.mark.parametrize("n, m, d, o, bq, bk", SHAPES)
def test_pallas_matches_dense_values_and_grads(n, m, d, o, bq, bk):
    _requires_accelerator()
    _check("pallas", n, m, d, o, bq, bk, 3e-5)


@pytest.mark.parametrize("impl", ["xla", "pallas"])
def test_zero_weights_and_large_logits_are_finite(impl):
    if impl == "pallas":
        _requires_accelerator()
    x, y, lw, _ = _problem(40, 70, 3, 0)
    # A whole key block of zero weights, plus a tiny temperature (logits ~ -1e4).
    lw = lw.at[16:48].set(-jnp.inf)
    eps = 1e-3
    got = kernel_logsumexp(
        x, y, log_weights=lw, epsilon=eps, impl=impl, block_q=16, block_k=16
    )
    want = kernel_logsumexp_dense(x, y, log_weights=lw, epsilon=eps)
    assert np.all(np.isfinite(np.asarray(got)))
    assert _rel(got, want) < 1e-5
    g = jax.grad(
        lambda x: jnp.sum(
            kernel_logsumexp(
                x, y, log_weights=lw, epsilon=eps, impl=impl, block_q=16, block_k=16
            )
        )
    )(x)
    g_ref = jax.grad(
        lambda x: jnp.sum(kernel_logsumexp_dense(x, y, log_weights=lw, epsilon=eps))
    )(x)
    assert _rel(g, g_ref) < 1e-4


@pytest.mark.parametrize("impl", ["xla", "pallas"])
def test_callable_cost(impl):
    if impl == "pallas":
        _requires_accelerator()

    def cost(a, b):
        diff = a[:, None, :] - b[None, :, :]
        return jnp.sum(jnp.sqrt(diff * diff + 0.1), -1)

    x, y, _, _ = _problem(50, 70, 4, 0)
    f = lambda fn: jax.value_and_grad(  # noqa: E731
        lambda x, y: jnp.sum(fn(x, y, cost=cost, epsilon=0.5) ** 2), argnums=(0, 1)
    )(x, y)
    got = f(
        lambda *a, **kw: kernel_logsumexp(*a, impl=impl, block_q=16, block_k=32, **kw)
    )
    want = f(kernel_logsumexp_dense)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        assert _rel(a, b) < 1e-5


def test_callable_matmul_cost_small_dim_auto_dispatch():
    # Triton dots need D >= 16; the automatic dispatch must route small-D
    # callables to the XLA path (correct on every backend).
    x, y, _, _ = _problem(64, 64, 4, 0)
    cost = lambda a, b: -(a @ b.T)  # noqa: E731
    got = kernel_logsumexp(x, y, cost=cost)
    want = kernel_logsumexp_dense(x, y, cost=cost)
    assert _rel(got, want) < 1e-5


@pytest.mark.parametrize("impl", ["xla", "pallas"])
def test_bf16_inputs(impl):
    if impl == "pallas":
        _requires_accelerator()
    x, y, _, _ = _problem(96, 80, 8, 0)
    xb, yb = x.astype(jnp.bfloat16), y.astype(jnp.bfloat16)
    got = kernel_logsumexp(xb, yb, epsilon=0.5, impl=impl, block_q=32, block_k=32)
    assert got.dtype == jnp.float32
    want = kernel_logsumexp_dense(
        xb.astype(jnp.float32), yb.astype(jnp.float32), epsilon=0.5
    )
    assert _rel(got, want) < 2e-2
    g = jax.grad(
        lambda x: jnp.sum(
            kernel_logsumexp(x, yb, epsilon=0.5, impl=impl, block_q=32, block_k=32)
        )
    )(xb)
    assert g.dtype == jnp.bfloat16
    g_ref = jax.grad(
        lambda x: jnp.sum(
            kernel_logsumexp_dense(x, yb.astype(jnp.float32), epsilon=0.5)
        )
    )(xb.astype(jnp.float32))
    assert _rel(g, g_ref) < 3e-2


def test_vmap_and_default_dispatch():
    x, y, _, _ = _problem(33, 45, 3, 0)
    xs = jnp.stack([x, x + 1.0])
    ys = jnp.stack([y, 2.0 * y])
    got = jax.vmap(lambda a, b: kernel_logsumexp(a, b, block_q=16, block_k=16))(xs, ys)
    want = jax.vmap(kernel_logsumexp_dense)(xs, ys)
    assert _rel(got, want) < 1e-5


def test_fp32_cross_term_is_exact_on_accelerator():
    # TF32 rounding of x.y would give O(1e-3 |x||y| / eps) errors in the logits.
    _requires_accelerator()
    x, y, _, _ = _problem(256, 512, 16, 0)
    x, y = 3.0 * x, 3.0 * y
    got = kernel_logsumexp(x, y, epsilon=0.05, impl="pallas")
    xn, yn = np.asarray(x, np.float64), np.asarray(y, np.float64)
    c = (xn**2).sum(1)[:, None] + (yn**2).sum(1)[None] - 2 * xn @ yn.T
    from scipy.special import logsumexp

    want = logsumexp(-c / 0.05, axis=1)
    assert _rel(got, want) < 1e-5


def test_input_validation():
    x, y, _, _ = _problem(8, 9, 2, 0)
    with pytest.raises(ValueError):
        kernel_logsumexp(x, y[:, :1])
    with pytest.raises(ValueError):
        kernel_logsumexp(x, y, log_weights=jnp.zeros(3))
    with pytest.raises(ValueError):
        kernel_logsumexp(x, y, impl="nope")
