"""Forward + reverse AD for the SSD kernel via jvp primitive + transpose.

The design guarantee under test: forward mode costs 4 forward-kernel
invocations (primal + the 3-call composite), and reverse mode stages exactly
one fused ssd_bwd call — identical to the old custom_vjp path. The composite
math is verified against jax.jvp of a pure-JAX reference; kernel-numerics
comparisons are GPU-gated in tests/test_pallas_old_vs_new.py.
"""

import importlib

import jax
import jax.numpy as jnp
import pytest

ssd_mod = importlib.import_module("probjax.nn.pallas_kernels.kernels.ssd")


def ssd_reference(q, k, v, log_alpha, h0):
    """Pure-JAX SSD: H_t = e^{a_t} H_{t-1} + k_t v_t^T ; o_t = q_t^T H_t."""
    _, groups, _, _ = q.shape
    _, heads, _, _ = v.shape
    rep = heads // groups
    qh = jnp.repeat(q, rep, axis=1)
    kh = jnp.repeat(k, rep, axis=1)

    def scan_head(h0, inputs):
        def step(carry, x):
            q_i, k_i, v_i, a_i = x
            new = jnp.exp(a_i) * carry + jnp.outer(k_i, v_i)
            return new, q_i @ new

        _, o = jax.lax.scan(step, h0, inputs)
        return o

    return jax.vmap(jax.vmap(scan_head))(h0, (qh, kh, v, log_alpha))


@pytest.fixture
def ssd_inputs():
    key = jax.random.key(0)
    batch, groups, heads, seq, dk, dv = 2, 2, 4, 16, 3, 5
    ks = jax.random.split(key, 10)
    q = jax.random.normal(ks[0], (batch, groups, seq, dk)) * 0.5
    k = jax.random.normal(ks[1], (batch, groups, seq, dk)) * 0.5
    v = jax.random.normal(ks[2], (batch, heads, seq, dv)) * 0.5
    log_alpha = -jnp.abs(jax.random.normal(ks[3], (batch, heads, seq))) * 0.3
    h0 = jax.random.normal(ks[4], (batch, heads, dk, dv)) * 0.5
    tangents = tuple(
        jax.random.normal(kx, a.shape) * 0.5
        for kx, a in zip(ks[5:], (q, k, v, log_alpha, h0))
    )
    return (q, k, v, log_alpha, h0), tangents


def test_jvp_composite_matches_reference(ssd_inputs):
    # The 3-call composite (evaluated on the pure-JAX reference forward)
    # must equal jax.jvp of the reference — this pins the derivation.
    primals, tangents = ssd_inputs
    q, k, v, log_alpha, h0 = primals
    dq, dk, dv, dla, dh0 = tangents

    o, t_ref = jax.jvp(ssd_reference, primals, tangents)

    cum = jnp.cumsum(dla, axis=-1)[..., None]
    composite = (
        ssd_reference(dq, k, v, log_alpha, h0)
        + ssd_reference(q, dk, v, log_alpha, jnp.zeros_like(h0))
        + ssd_reference(q, k, dv - cum * v, log_alpha, dh0)
        + cum * o
    )
    assert jnp.allclose(t_ref, composite, atol=1e-4)


def test_transpose_is_adjoint_of_jvp(ssd_inputs):
    # <ct, J dx> == <J^T ct, dx>: the fused backward (the transpose target)
    # is the exact adjoint of the composite linear map.
    primals, tangents = ssd_inputs
    o, tangent = jax.jvp(ssd_reference, primals, tangents)
    _, vjp_fn = jax.vjp(ssd_reference, *primals)
    ct = jax.random.normal(jax.random.key(7), o.shape)
    lhs = jnp.vdot(ct, tangent)
    rhs = sum(jnp.vdot(a, b) for a, b in zip(vjp_fn(ct), tangents))
    assert jnp.allclose(lhs, rhs, atol=1e-3)


def _count_primitives(jaxpr, counts=None):
    counts = {} if counts is None else counts
    for eqn in jaxpr.eqns:
        counts[eqn.primitive.name] = counts.get(eqn.primitive.name, 0) + 1
        for value in eqn.params.values():
            for item in value if isinstance(value, (list, tuple)) else [value]:
                if hasattr(item, "jaxpr"):
                    _count_primitives(
                        item.jaxpr if hasattr(item.jaxpr, "eqns") else item,
                        counts,
                    )
    return counts


def test_reverse_mode_stages_single_fused_backward(ssd_inputs):
    # Structural guarantee (abstract eval only — runs on CPU): grad stages
    # exactly one ssd_fwd + one ssd_bwd, and the jvp primitive is fully
    # transposed away.
    primals, _ = ssd_inputs

    def loss(q, k, v, log_alpha, h0):
        return jnp.sum(ssd_mod._ssd_op(q, k, v, log_alpha, h0) ** 2)

    jaxpr = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*primals)
    counts = _count_primitives(jaxpr.jaxpr)
    assert counts.get("ssd_fwd", 0) == 1, counts
    assert counts.get("ssd_bwd", 0) == 1, counts
    assert counts.get("ssd_jvp", 0) == 0, counts


def test_forward_mode_stages_jvp_primitive(ssd_inputs):
    primals, tangents = ssd_inputs

    jaxpr = jax.make_jaxpr(
        lambda p, t: jax.jvp(ssd_mod._ssd_op, p, t)
    )(primals, tangents)
    counts = _count_primitives(jaxpr.jaxpr)
    assert counts.get("ssd_fwd", 0) == 1, counts
    assert counts.get("ssd_jvp", 0) == 1, counts
    assert counts.get("ssd_bwd", 0) == 0, counts


def test_value_and_grad_stages_no_extra_forwards(ssd_inputs):
    # value_and_grad must not duplicate the forward.
    primals, _ = ssd_inputs

    def loss(*args):
        return jnp.sum(ssd_mod._ssd_op(*args) ** 2)

    jaxpr = jax.make_jaxpr(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4)))(
        *primals
    )
    counts = _count_primitives(jaxpr.jaxpr)
    assert counts.get("ssd_fwd", 0) == 1, counts
    assert counts.get("ssd_bwd", 0) == 1, counts
