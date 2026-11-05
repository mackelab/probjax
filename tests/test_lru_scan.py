"""The LRU cell's time-invariant linear recurrence scan."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn.layers.ssm import LRUCell, binary_operator_diag, linear_recurrence_scan


@pytest.fixture(autouse=True)
def _exact_matmuls():
    # GPUs default to TF32 matmuls; compare implementations at full precision.
    with jax.default_matmul_precision("highest"):
        yield


def _sequential(decay, inputs):
    def step(h, u):
        h = decay * h + u
        return h, h

    h0 = jnp.zeros(inputs.shape[:-2] + inputs.shape[-1:], inputs.dtype)
    _, hs = jax.lax.scan(step, h0, jnp.moveaxis(inputs, -2, 0))
    return jnp.moveaxis(hs, 0, -2)


def _rel(a, b):
    err = float(jnp.linalg.norm(jnp.ravel(a - b)))
    return err / max(float(jnp.linalg.norm(jnp.ravel(b))), 1e-12)


@pytest.mark.parametrize("length", [1, 2, 3, 7, 16, 37, 64])
@pytest.mark.parametrize("complex_decay", [False, True])
def test_linear_recurrence_scan_matches_sequential(length, complex_decay):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    decay = jax.random.uniform(k1, (5,), minval=0.3, maxval=0.99)
    inputs = jax.random.normal(k2, (2, length, 5))
    if complex_decay:
        decay = decay * jnp.exp(1j * jax.random.uniform(k3, (5,), maxval=6.0))
        inputs = inputs + 1j * jax.random.normal(k3, inputs.shape)
    out = linear_recurrence_scan(decay, inputs)
    assert out.shape == inputs.shape
    assert _rel(out, _sequential(decay, inputs)) < 1e-5

    # Gradients wrt the decay and inputs agree with the sequential scan.
    weights = jax.random.normal(k1, inputs.shape)

    def loss(fn, decay, inputs):
        return jnp.sum(jnp.real(fn(decay, inputs)) * weights)

    g = jax.grad(loss, argnums=(1, 2))(linear_recurrence_scan, decay, inputs)
    r = jax.grad(loss, argnums=(1, 2))(_sequential, decay, inputs)
    for a, b in zip(g, r, strict=True):
        assert _rel(a, b) < 1e-4

    # Forward mode (custom JVP) as well.
    tangents = (jnp.ones_like(decay) * 0.1, jnp.ones_like(inputs))
    _, t_new = jax.jvp(linear_recurrence_scan, (decay, inputs), tangents)
    _, t_ref = jax.jvp(_sequential, (decay, inputs), tangents)
    assert _rel(t_new, t_ref) < 1e-4


def test_linear_recurrence_scan_second_order():
    decay = jnp.array([0.5, 0.9, 0.7])
    inputs = jax.random.normal(jax.random.key(0), (9, 3))

    def loss(fn, decay):
        return jnp.sum(fn(decay, inputs) ** 2)

    hess = jax.hessian(loss, argnums=1)(linear_recurrence_scan, decay)
    ref = jax.hessian(loss, argnums=1)(_sequential, decay)
    assert _rel(hess, ref) < 1e-4


def _reference_lru(cell, x):
    """The original LRU forward (generic associative scan, complex matmuls)."""
    diag_lambda = jnp.exp(
        -jnp.exp(cell.nu_log[...]) + 1j * jnp.exp(cell.theta_log[...])
    )
    b_norm = (cell.B_re[...] + 1j * cell.B_im[...]) * jnp.exp(cell.gamma_log[...])
    c = cell.C_re[...] + 1j * cell.C_im[...]

    def single(x_td):
        lam = jnp.repeat(diag_lambda[None], x_td.shape[0], axis=0)
        bu = jnp.einsum("ih,ti->th", b_norm, x_td)
        _, h = jax.lax.associative_scan(binary_operator_diag, (lam, bu))
        out = jnp.real(jnp.einsum("th,oh->to", h, c))
        return out + jnp.einsum("ti,oi->to", x_td, cell.D[...])

    return jax.vmap(single)(x) if x.ndim == 3 else single(x)


@pytest.mark.parametrize("shape", [(3, 37, 16), (20, 16), (2, 1, 16)])
def test_lru_cell_matches_reference(shape):
    cell = LRUCell(16, nnx.Rngs(0), state_dim=24)
    x = jax.random.normal(jax.random.key(1), shape)
    weights = jax.random.normal(jax.random.key(2), shape)
    assert _rel(cell(x), _reference_lru(cell, x)) < 1e-5

    def loss(fn):
        return lambda cell, x: jnp.sum(fn(cell, x) * weights)

    g_new = nnx.grad(loss(lambda c, x: c(x)), argnums=(0, 1))(cell, x)
    g_ref = nnx.grad(loss(_reference_lru), argnums=(0, 1))(cell, x)
    assert _rel(g_new[1], g_ref[1]) < 1e-5
    new = dict(nnx.to_flat_state(nnx.state(g_new[0], nnx.Param)))
    ref = dict(nnx.to_flat_state(nnx.state(g_ref[0], nnx.Param)))
    for path in ref:
        assert _rel(new[path][...], ref[path][...]) < 1e-4, path
