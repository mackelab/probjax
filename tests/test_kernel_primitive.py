"""CPU-runnable tests for the declarative kernel-primitive machinery.

Includes the Phase-0 spike: a toy interpret-mode pallas kernel routed through
``make_kernel_primitive`` under jit + NamedSharding — validating the
primitive -> mlir.lower_fun -> custom_partitioning -> pallas_call nesting
that all real kernels rely on.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl
from jax.sharding import NamedSharding, PartitionSpec as P

from probjax.nn.pallas_kernels.kernel_utils.kernel_primitive import (
    KernelSpec,
    Operand,
    Output,
    bind_kernel,
    ct,
    derive_bwd_spec,
    grad,
    make_kernel_primitive,
    out_res,
    res,
    shardable_kernel,
    sharding_rule,
)

# ---------------------------------------------------------------------------
# Spec machinery
# ---------------------------------------------------------------------------

SSD_FWD_SPEC = KernelSpec(
    name="test_ssd_fwd",
    operands=(
        Operand("q", ("batch", "groups", "seq", "dk")),
        Operand("k", ("batch", "groups", "seq", "dk")),
        Operand("v", ("batch", "heads", "seq", "dv")),
        Operand("log_alpha", ("batch", "heads", "seq")),
        Operand("h0", ("batch", "heads", "dk", "dv")),
    ),
    outputs=(Output(("batch", "heads", "seq", "dv"), dtype_like="v"),),
)

SSD_BWD_SPEC = derive_bwd_spec(
    SSD_FWD_SPEC,
    name="test_ssd_bwd",
    operands=(
        ct(0, name="do"),
        res("q"),
        res("k"),
        res("v"),
        res("log_alpha"),
        res("h0"),
    ),
    outputs=(grad("q"), grad("k"), grad("v"), grad("log_alpha"), grad("h0")),
)


def test_sharding_rule_matches_handwritten_ssd_fwd():
    # Golden value: the hand-written rule from the pre-primitive ssd.py.
    rule, replication = sharding_rule(SSD_FWD_SPEC)
    assert rule == (
        "batch groups seq dk, batch groups seq dk, "
        "batch heads seq dv, batch heads seq, batch heads dk dv "
        "-> batch heads seq dv"
    )
    assert set(replication) == {"seq", "dk", "dv"}


def test_derived_bwd_rule_matches_handwritten_ssd_bwd():
    rule, replication = sharding_rule(SSD_BWD_SPEC)
    assert rule == (
        "batch heads seq dv, batch groups seq dk, batch groups seq dk, "
        "batch heads seq dv, batch heads seq, batch heads dk dv "
        "-> batch groups seq dk, batch groups seq dk, "
        "batch heads seq dv, batch heads seq, batch heads dk dv"
    )
    assert set(replication) == {"seq", "dk", "dv"}


def test_optional_operands_and_anonymous_dims():
    spec = KernelSpec(
        name="test_opt",
        operands=(
            Operand("q", ("batch", "seq", "heads", "head_dim")),
            Operand("rng", ()),
            Operand("bias", ("batch", "heads", "_", "_"), optional=True),
        ),
        outputs=(Output(("batch", "seq", "heads", "head_dim"), dtype_like="q"),),
    )
    rule_absent, factors_absent = sharding_rule(spec, present=())
    assert rule_absent == (
        "batch seq heads head_dim,  -> batch seq heads head_dim"
    )
    rule_present, factors_present = sharding_rule(spec, present=("bias",))
    assert rule_present == (
        "batch seq heads head_dim, , batch heads e_bias_2 e_bias_3 "
        "-> batch seq heads head_dim"
    )
    assert "e_bias_2" in factors_present and "e_bias_2" not in factors_absent


def test_real_ssd_and_mamba_specs_match_golden_rules():
    from probjax.nn.pallas_kernels.kernels.mamba import (
        _MAMBA_BWD_SPEC,
        _MAMBA_FWD_SPEC,
    )
    from probjax.nn.pallas_kernels.kernels.ssd import _SSD_BWD_SPEC, _SSD_FWD_SPEC

    assert sharding_rule(_SSD_FWD_SPEC) == sharding_rule(SSD_FWD_SPEC)
    assert sharding_rule(_SSD_BWD_SPEC) == sharding_rule(SSD_BWD_SPEC)

    # Golden values: the hand-written rules from the pre-primitive mamba.py.
    rule_fwd, repl_fwd = sharding_rule(_MAMBA_FWD_SPEC)
    assert rule_fwd == (
        "batch seq dim, state dim, batch seq state, "
        "batch seq state, batch seq dim, one dim "
        "-> batch seq dim"
    )
    assert set(repl_fwd) == {"seq", "dim", "state", "one"}

    rule_bwd, repl_bwd = sharding_rule(_MAMBA_BWD_SPEC)
    assert rule_bwd == (
        "batch seq dim, batch seq dim, state dim, "
        "batch seq state, batch seq state, batch seq dim, one dim "
        "-> batch seq dim, state dim, batch seq state, "
        "batch seq state, batch seq dim, one dim"
    )
    assert set(repl_bwd) == {"seq", "dim", "state", "one"}


# ---------------------------------------------------------------------------
# Toy pallas kernel through the factory (the spike)
# ---------------------------------------------------------------------------


def _toy_impl(x, y, bias, *, scale):
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] * y_ref[...] * scale

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True,
    )(x, y)
    if bias is not None:
        out = out + bias
    return out


TOY_SPEC = KernelSpec(
    name="test_toy_kernel",
    operands=(
        Operand("x", ("batch", "feat")),
        Operand("y", ("batch", "feat")),
        Operand("bias", ("batch", "feat"), optional=True),
    ),
    outputs=(Output(("batch", "feat"), dtype_like="x"),),
    shardable=frozenset({"batch"}),
)

toy_p = make_kernel_primitive(TOY_SPEC, impl=_toy_impl)


def _toy(x, y, bias=None, scale=2.0):
    return shardable_kernel(toy_p, {"x": x, "y": y, "bias": bias}, scale=scale)


def _reference(x, y, bias=None, scale=2.0):
    out = x * y * scale
    return out if bias is None else out + bias


@pytest.fixture
def toy_inputs():
    key = jax.random.key(0)
    kx, ky, kb = jax.random.split(key, 3)
    x = jax.random.normal(kx, (8, 4))
    y = jax.random.normal(ky, (8, 4))
    bias = jax.random.normal(kb, (8, 4))
    return x, y, bias


def test_toy_eager_and_jit(toy_inputs):
    x, y, bias = toy_inputs
    assert jnp.allclose(_toy(x, y), _reference(x, y), atol=1e-6)
    assert jnp.allclose(_toy(x, y, bias), _reference(x, y, bias), atol=1e-6)
    assert jnp.allclose(jax.jit(_toy)(x, y), _reference(x, y), atol=1e-6)


def test_toy_abstract_eval_shapes(toy_inputs):
    x, y, _ = toy_inputs
    out_shape = jax.eval_shape(_toy, x, y)
    assert out_shape.shape == (8, 4)
    assert out_shape.dtype == x.dtype


def test_toy_vmap(toy_inputs):
    x, y, bias = toy_inputs
    xs = jnp.stack([x, x + 1.0, x + 2.0])
    ys = jnp.stack([y, y - 1.0, y - 2.0])
    got = jax.vmap(lambda a, b: _toy(a, b, bias))(xs, ys)
    want = jax.vmap(lambda a, b: _reference(a, b, bias))(xs, ys)
    assert jnp.allclose(got, want, atol=1e-6)


def test_toy_spike_partition_under_mesh(toy_inputs):
    # The load-bearing spike: primitive -> lower_fun -> custom_partitioning
    # -> pallas_call, under jit with NamedSharding'd inputs.
    if jax.device_count() < 4:
        pytest.skip("spike test requires at least 4 devices")
    x, y, bias = toy_inputs
    mesh = jax.make_mesh(
        (4,), ("data",), devices=jax.devices()[:4],
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    sharding = NamedSharding(mesh, P("data", None))
    xs = jax.device_put(x, sharding)
    ys = jax.device_put(y, sharding)
    bs = jax.device_put(bias, sharding)

    out = jax.jit(_toy)(xs, ys)
    assert jnp.allclose(out, _reference(x, y), atol=1e-6)
    assert out.sharding.spec[0] == "data"

    out_b = jax.jit(lambda a, b, c: _toy(a, b, c))(xs, ys, bs)
    assert jnp.allclose(out_b, _reference(x, y, bias), atol=1e-6)

    # No all-gathers: the partition callback ran the kernel per-shard.
    hlo = jax.jit(_toy).lower(xs, ys).compile().as_text()
    assert "all-gather" not in hlo


def test_toy_validator_rejects_feat_sharding(toy_inputs):
    if jax.device_count() < 4:
        pytest.skip("requires at least 4 devices")
    x, y, _ = toy_inputs
    mesh = jax.make_mesh(
        (4,), ("data",), devices=jax.devices()[:4],
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    xs = jax.device_put(x, NamedSharding(mesh, P(None, "data")))
    ys = jax.device_put(y, NamedSharding(mesh, P(None, "data")))
    # The validator raises inside the XLA partition callback, surfacing
    # as a runtime error that carries the original message.
    with pytest.raises(Exception, match="feat"):
        jax.jit(_toy)(xs, ys).block_until_ready()


def test_bind_kernel_missing_required_raises():
    with pytest.raises(ValueError, match="required operand 'y'"):
        bind_kernel(toy_p, {"x": jnp.ones((2, 2))}, scale=1.0)
