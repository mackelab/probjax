"""GPU checks for the Triton grouped-GEMM dropless MoE path (dropless_impl="triton").

Run on a GPU node; skipped elsewhere (the CPU suite covers the kernel in
Pallas interpret mode).  Each check compares the Triton kernel against XLA's
``ragged_dot`` with identical parameters:

* forward and gradients at production-like token counts, fp32 and bf16;
* ragged groups: empty experts, one expert taking every row, uneven sizes;
* determinism (same inputs, same outputs, run to run);
* finiteness under large activations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probjax.nn.layers.moe import ExpertSwiGLU, MoELayer

pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU")

E, D, H = 8, 512, 512


def _experts(dtype):
    return ExpertSwiGLU(E, D, H, dtype=dtype, param_dtype=jnp.float32, rngs=nnx.Rngs(0))


def _sizes(kind, rows, key):
    if kind == "uniform":
        s = np.full(E, rows // E)
    elif kind == "one_expert":
        s = np.zeros(E, int)
        s[3] = rows
    elif kind == "empty_and_uneven":
        p = np.array([0, 0.4, 0, 0.05, 0.3, 0, 0.2, 0.05])
        s = np.floor(p * rows).astype(int)
    else:  # random
        p = np.asarray(jax.random.dirichlet(key, jnp.ones(E) * 0.5))
        s = np.floor(p * rows).astype(int)
    s[np.argmax(s)] += rows - s.sum()
    return jnp.asarray(s, jnp.int32)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def _fwd_and_grads(experts, x, sizes, backend):
    graphdef, params = nnx.split(experts, nnx.Param)

    def loss(params, x):
        m = nnx.merge(graphdef, params)
        if backend == "ragged_dot":
            y = m.ragged_forward(x, sizes)
        else:
            y = m.grouped_forward(x, sizes, backend=backend)
        return jnp.sum(jnp.sin(y.astype(jnp.float32))), y

    (_, y), (gp, gx) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(params, x)
    return y, gx, gp


# bf16 compares two bf16 computations with different accumulation order, so
# its tolerance is set by bf16 rounding, not by the kernel.
TOL = {jnp.float32: 2e-3, jnp.bfloat16: 3e-2}


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("rows", [4096, 65_536, 262_144])
@pytest.mark.parametrize("kind", ["uniform", "random", "empty_and_uneven", "one_expert"])
def test_triton_matches_ragged_dot(dtype, rows, kind):
    if rows == 262_144 and kind != "random":
        pytest.skip("largest size only for the random layout (time)")
    experts = _experts(dtype)
    x = jax.random.normal(jax.random.key(1), (rows, D), jnp.float32).astype(dtype)
    sizes = _sizes(kind, rows, jax.random.key(2))
    y_t, gx_t, gp_t = _fwd_and_grads(experts, x, sizes, "triton")
    y_r, gx_r, gp_r = _fwd_and_grads(experts, x, sizes, "ragged_dot")
    tol = TOL[dtype]
    assert np.isfinite(np.asarray(y_t, np.float32)).all()
    assert _rel(y_t, y_r) < tol, ("forward", _rel(y_t, y_r))
    assert _rel(gx_t, gx_r) < tol, ("grad x", _rel(gx_t, gx_r))
    for a, b in zip(jax.tree.leaves(gp_t), jax.tree.leaves(gp_r)):
        assert _rel(a, b) < tol, ("grad params", _rel(a, b))


def test_triton_is_deterministic():
    experts = _experts(jnp.bfloat16)
    x = jax.random.normal(jax.random.key(3), (65_536, D), jnp.bfloat16)
    sizes = _sizes("random", 65_536, jax.random.key(4))
    a = _fwd_and_grads(experts, x, sizes, "triton")
    b = _fwd_and_grads(experts, x, sizes, "triton")
    for la, lb in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(la, np.float32), np.asarray(lb, np.float32))


def test_triton_finite_under_large_activations():
    experts = _experts(jnp.bfloat16)
    x = (jax.random.normal(jax.random.key(5), (16_384, D)) * 30.0).astype(jnp.bfloat16)
    sizes = _sizes("random", 16_384, jax.random.key(6))
    y, gx, gp = _fwd_and_grads(experts, x, sizes, "triton")
    for leaf in jax.tree.leaves((y, gx, gp)):
        assert np.isfinite(np.asarray(leaf, np.float32)).all()


def test_moe_layer_triton_matches_ragged_dot_end_to_end():
    """Router + permutation + kernel: the whole layer, outputs and grads."""
    def layer(impl):
        return MoELayer(D, H, E, top_k=2, routing="dropless", dropless_impl=impl,
                        dtype=jnp.bfloat16, rngs=nnx.Rngs(7))

    x = jax.random.normal(jax.random.key(8), (32, 512, D), jnp.bfloat16)
    outs = {}
    for impl in ("triton", "ragged_dot"):
        m = layer(impl)
        graphdef, params = nnx.split(m, nnx.Param)

        def loss(params):
            return jnp.mean(nnx.merge(graphdef, params)(x).astype(jnp.float32) ** 2)

        outs[impl] = jax.jit(jax.value_and_grad(loss))(params)
    (lt, gt), (lr, gr) = outs["triton"], outs["ragged_dot"]
    assert abs(float(lt) - float(lr)) / abs(float(lr)) < 2e-2
    for a, b in zip(jax.tree.leaves(gt), jax.tree.leaves(gr)):
        assert _rel(a, b) < 5e-2
