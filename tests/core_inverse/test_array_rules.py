"""Array-manipulation primitive tests for the inverse interpreter.

Covers gather, scatter, slice, broadcast, reshape, transpose, squeeze, split,
concatenate, convert_element_type, bitcast_convert_type.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import checkify

from probjax.core import inverse, inverse_and_logabsdet
from probjax.core.registry import REGISTRY, Context

from .helpers import logdet_via_autodiff


# ---------------------------------------------------------------------------
# Gather / indexing
# ---------------------------------------------------------------------------


def test_inverse_gather_permutation():
    indices = jnp.array([2, 0, 1], dtype=jnp.int32)

    def f(x):
        return x[indices]

    x0 = jnp.array([0.3, -1.2, 2.5])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_gather_duplicate_indices():
    def f(x, indices):
        return jax.lax.gather(
            x,
            indices[:, None],
            jax.lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(0,),
                start_index_map=(0,),
            ),
            slice_sizes=(1,),
        )

    x0 = jnp.array([1.0, 2.0, 3.0])
    indices = jnp.array([0, 0, 1], dtype=jnp.int32)
    eqn = next(
        eqn
        for eqn in jax.make_jaxpr(f)(x0, indices).jaxpr.eqns
        if eqn.primitive is jax.lax.gather_p
    )
    rule = REGISTRY.get(jax.lax.gather_p, Context.INVERSE)
    x_rec = rule(eqn, [None, indices[:, None]], [f(x0, indices)]).resolved_vals[0]
    assert jnp.allclose(x_rec[:2], jnp.array([1.0, 2.0]))
    assert jnp.isnan(x_rec[2])


def test_inverse_gather_omitted_indices():
    def f(x, indices):
        return jax.lax.gather(
            x,
            indices[:, None],
            jax.lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(0,),
                start_index_map=(0,),
            ),
            slice_sizes=(1,),
        )

    x0 = jnp.array([1.0, 2.0, 3.0])
    indices = jnp.array([0, 2], dtype=jnp.int32)
    eqn = next(
        eqn
        for eqn in jax.make_jaxpr(f)(x0, indices).jaxpr.eqns
        if eqn.primitive is jax.lax.gather_p
    )
    rule = REGISTRY.get(jax.lax.gather_p, Context.INVERSE)
    x_rec = rule(eqn, [None, indices[:, None]], [f(x0, indices)]).resolved_vals[0]
    selected = jnp.array([0, 2])
    assert jnp.allclose(x_rec[selected], x0[selected])
    assert jnp.isnan(x_rec[1])


def test_logabsdet_gather_permutation():
    indices = jnp.array([2, 0, 1], dtype=jnp.int32)

    def f(x):
        return jnp.exp(x[indices])

    x0 = jnp.array([0.3, -1.2, 2.5])
    y0 = f(x0)
    inv_and_det = inverse_and_logabsdet(f)
    x_rec, log_det = inv_and_det(y0)

    expected = -jnp.sum(jnp.log(jnp.abs(y0)))
    expected_jac = logdet_via_autodiff(f, x0)

    assert jnp.allclose(x_rec, x0, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(log_det, expected, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(log_det, expected_jac, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------


def test_inverse_scatter_update():
    x0 = jnp.array([1.0, 2.0, 3.0])

    def f(x):
        y = jnp.ones((10, 3))
        y = y.at[-1].set(x * 2.0)
        return y[-1]

    inv_f = inverse(f)
    x_inv = inv_f(f(x0))
    assert jnp.allclose(x0, x_inv, atol=1e-6, rtol=1e-6)


def test_inverse_scatter_overwrite_nonconstant():
    operand = jnp.array([1.0, 2.0, 3.0])
    index = jnp.array([[1]], dtype=jnp.int32)
    update = jnp.array([10.0])

    def f(x):
        return jax.lax.scatter(
            operand,
            index,
            x,
            jax.lax.ScatterDimensionNumbers(
                update_window_dims=(),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            ),
        )

    y = f(update)
    eqn = jax.make_jaxpr(f)(update).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.scatter_p, Context.INVERSE)
    result = rule(eqn, [operand, index, None], [y])

    assert result.resolved_vars == [eqn.invars[2]]
    reconstructed_update = result.resolved_vals[0]
    assert jnp.allclose(reconstructed_update, update, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Slice / dynamic_slice
# ---------------------------------------------------------------------------


def test_inverse_strided_slice():
    x0 = jnp.array([1.0, 2.0, 3.0, 4.0])

    def f(x):
        return x[::2]

    y = f(x0)
    eqn = jax.make_jaxpr(f)(x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.slice_p, Context.INVERSE)
    result = rule(eqn, [None], [y])

    x_rec = result.resolved_vals[0]
    if hasattr(x_rec, "value"):
        x_rec = x_rec.value

    assert jnp.all(jnp.isnan(x_rec[1::2]))
    assert jnp.allclose(x_rec[::2], x0[::2])


# ---------------------------------------------------------------------------
# Reshape / transpose / squeeze / rev
# ---------------------------------------------------------------------------


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


def test_logabsdet_nested_jit_flip():
    def f(z):
        return jnp.exp(jnp.flip(jnp.exp(z)))

    x = jnp.array([0.3, 0.7])
    y = f(x)
    x_rec, log_det = inverse_and_logabsdet(f)(y)
    expected = logdet_via_autodiff(f, x)
    assert jnp.allclose(x_rec, x, atol=1e-5)
    assert jnp.allclose(log_det, expected, atol=1e-5)


def test_logabsdet_transpose_chain():
    def f(x):
        return jnp.exp(jnp.transpose(x, (1, 0)))

    x0 = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
    y0 = f(x0)
    inv_and_det = inverse_and_logabsdet(f)
    x_rec, log_det = inv_and_det(y0)

    expected = -jnp.sum(jnp.log(jnp.abs(y0)))
    expected_jac = logdet_via_autodiff(f, x0)

    assert jnp.allclose(x_rec, x0, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(log_det, expected, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(log_det, expected_jac, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------


def test_inverse_broadcast_scalar_to_vector_consistent():
    def f(x):
        return jnp.broadcast_to(x, (4,))

    x0 = jnp.array(2.0)
    y = f(x0)
    eqn = jax.make_jaxpr(f)(x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.broadcast_in_dim_p, Context.INVERSE)
    x_rec = rule(eqn, [None], [y]).resolved_vals[0]
    assert x_rec.shape == x0.shape
    assert jnp.allclose(x_rec, x0)


def test_inverse_broadcast_scalar_to_vector_inconsistent():
    def f(x):
        return jnp.broadcast_to(x, (4,))

    y = jnp.array([2.0, 2.0, 3.0, 2.0])
    x0 = jnp.array(2.0)
    eqn = jax.make_jaxpr(f)(x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.broadcast_in_dim_p, Context.INVERSE)
    x_rec = rule(eqn, [None], [y]).resolved_vals[0]
    assert x_rec.shape == ()
    assert jnp.isnan(x_rec)


def test_inverse_and_logabsdet_broadcast_is_undefined():
    def f(x):
        return jnp.broadcast_to(x, (4,))

    x = jnp.array(2.0)
    eqn = jax.make_jaxpr(f)(x).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.broadcast_in_dim_p, Context.INVERSE_LOGDET)
    result = rule(eqn, [None], [f(x)], context=None)
    assert jnp.allclose(result.resolved_vals[0], x)
    assert jnp.isnan(result.state[eqn.invars[0]])


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


def test_inverse_convert_element_type_widen():
    def f(x):
        return x.astype(jnp.int32)

    x0 = jnp.arange(-2, 3, dtype=jnp.int16)
    eqn = jax.make_jaxpr(f)(x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.convert_element_type_p, Context.INVERSE)
    x_rec = rule(eqn, [None], [f(x0)]).resolved_vals[0]
    assert x_rec.dtype == x0.dtype
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_bitcast_convert_type():
    def f(x):
        return jax.lax.bitcast_convert_type(x, jnp.uint32)

    x0 = jnp.array([0.0, 1.5, -2.25, 3.75], dtype=jnp.float32)
    eqn = jax.make_jaxpr(f)(x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.bitcast_convert_type_p, Context.INVERSE)
    result = rule(eqn, [None], [f(x0)])
    x_rec = result.resolved_vals[0]
    assert x_rec.dtype == x0.dtype
    assert jnp.array_equal(
        jax.lax.bitcast_convert_type(x_rec, jnp.uint32),
        jax.lax.bitcast_convert_type(x0, jnp.uint32),
    )


def test_inverse_convert_element_type_narrowing():
    x0 = jnp.array([1.5, 2.7, -3.2])

    def f(x):
        return x.astype(jnp.int32)

    y = f(x0)
    eqn = jax.make_jaxpr(f)(x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.convert_element_type_p, Context.INVERSE)
    result = rule(eqn, [None], [y])

    x_rec = result.resolved_vals[0]
    assert jnp.all(jnp.isnan(x_rec))


def test_inverse_invalid_exact_conversion_raises():
    """A lossy cast is known at trace time, so it raises rather than deferring.

    This used to be reported through ``checkify.check``, which meant the failure
    was only visible under ``checkify.checkify`` -- and made every integer
    inverse crash under a plain ``jit`` with "Cannot abstractly evaluate a
    checkify.check which was not functionalized". Whether ``int32 -> float32``
    is lossy is a property of the dtypes, not of the values, so a plain
    exception is both correct and visible everywhere.
    """

    def f(x):
        return x.astype(jnp.float32)

    x0 = jnp.array([1, 2], dtype=jnp.int32)
    eqn = jax.make_jaxpr(f)(x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.convert_element_type_p, Context.INVERSE)
    with pytest.raises(NotImplementedError, match="does not have a unique inverse"):
        rule(eqn, [None], [f(x0)])


# ---------------------------------------------------------------------------
# Split / concatenate
# ---------------------------------------------------------------------------


def test_inverse_split():
    x = jnp.stack([jnp.arange(10.0), 100.0 + jnp.arange(10.0)], axis=-1)

    def f(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x2, x1], axis=-1)

    inv_f = inverse(f)
    inv_x = inv_f(f(x))
    assert jnp.allclose(x, inv_x, atol=1e-3, rtol=1e-3)


def test_inverse_and_logabsdet_split_rule():
    def f(x):
        return jnp.split(x, 2)

    x = jnp.arange(6.0)
    eqn = next(
        eqn
        for eqn in jax.make_jaxpr(f)(x).jaxpr.eqns
        if eqn.primitive is jax.lax.split_p
    )
    outputs = f(x)
    rule = REGISTRY.get(jax.lax.split_p, Context.INVERSE_LOGDET)
    result = rule(eqn, [None], list(outputs))
    assert jnp.allclose(result.resolved_vals[0], x)
    assert jnp.allclose(result.state[eqn.invars[0]], 0.0)


def test_inverse_and_logabsdet_squeeze_rule():
    def f(x):
        return jnp.squeeze(x, axis=0)

    x = jnp.arange(6.0).reshape(1, 2, 3)
    eqn = jax.make_jaxpr(f)(x).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.squeeze_p, Context.INVERSE_LOGDET)
    result = rule(eqn, [None], [f(x)], context=None)
    assert jnp.allclose(result.resolved_vals[0], x)
    assert jnp.allclose(result.state[eqn.invars[0]], 0.0)


def test_logabsdet_reshape_chain():
    def f(x):
        return 2.0 * jnp.reshape(x, (-1,))

    x0 = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    y0 = f(x0)
    inv_and_det = inverse_and_logabsdet(f)
    x_rec, log_det = inv_and_det(y0)

    expected = -x0.size * jnp.log(2.0)
    expected_jac = logdet_via_autodiff(f, x0)

    assert jnp.allclose(jnp.reshape(x_rec, x0.shape), x0, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(log_det, expected, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(log_det, expected_jac, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Log-determinants of structural primitives
# ---------------------------------------------------------------------------
#
# These rearrange elements without scaling any of them, so log|J| = 0. Before
# they had explicit rules they fell through to the diagonal autodiff fallback,
# which differentiates elementwise under nested vmap: reshape produced a (4, 4)
# tangent for a 4-element array and failed MLIR verification, and concatenate
# leaked the engine's internal Knowness wrapper out as a TypeError.


@pytest.mark.parametrize(
    "name,fn,y",
    [
        ("reshape", lambda x: jnp.reshape(jnp.exp(x), (4,)), jnp.exp(jnp.zeros((2, 2)))),
        ("ravel", lambda x: jnp.exp(x).ravel(), jnp.exp(jnp.zeros((2, 2)))),
        (
            "concatenate",
            lambda x: jnp.concatenate([jnp.exp(x[:1]), jnp.exp(x[1:])]),
            jnp.ones(2),
        ),
        # A sliced passthrough recombined by concatenate: invertible overall,
        # so the slice must contribute 0 rather than refusing because its
        # element count changed.
        (
            "concat_with_passthrough",
            lambda x: jnp.concatenate([jnp.exp(x[:1]), x[1:]]),
            jnp.ones(2),
        ),
        ("slice", lambda x: jnp.exp(x)[:2], jnp.ones(2)),
        ("transpose", lambda x: jnp.exp(x).T, jnp.exp(jnp.zeros((2, 3)))),
    ],
)
def test_logabsdet_rearrangements_are_zero(name, fn, y):
    x, logdet = inverse_and_logabsdet(fn)(y)
    assert jnp.all(jnp.isfinite(x)), f"{name} lost the inverse"
    assert jnp.allclose(logdet, 0.0), f"{name} gave log-det {logdet}, expected 0"


def test_logabsdet_dynamic_slice_is_refused():
    """dynamic_slice has no log-det rule, and says so rather than guessing.

    Its inverse recovers only the window, leaving the operand partially known,
    and a partially recovered input has no square Jacobian to take a
    determinant of. This used to crash on the engine's internal Knowness
    wrapper; a rearrangement rule that claimed a zero log-det instead made the
    engine re-schedule the equation forever.
    """

    def f(x):
        return jax.lax.dynamic_slice(jnp.exp(x), (1,), (2,))

    # The plain inverse is unaffected -- only the log-det is unavailable.
    assert jnp.all(jnp.isfinite(inverse(f)(jnp.ones(2))))
    with pytest.raises(NotImplementedError, match="dynamic_slice"):
        inverse_and_logabsdet(f)(jnp.ones(2))


def test_logabsdet_scatter_is_zero():
    base = jnp.arange(4.0)

    def f(x):
        return base.at[jnp.array([0, 2])].set(jnp.exp(x))

    _, logdet = inverse_and_logabsdet(f)(jnp.ones(2))
    assert jnp.allclose(logdet, 0.0)


def test_logabsdet_convert_element_type_is_zero():
    def f(x):
        return jnp.exp(x).astype(jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)

    _, logdet = inverse_and_logabsdet(f)(jnp.ones(3))
    assert jnp.allclose(logdet, 0.0)


def test_logabsdet_bitcast_is_zero():
    def f(x):
        return jax.lax.bitcast_convert_type(x, jnp.int32)

    _, logdet = inverse_and_logabsdet(f)(jax.lax.bitcast_convert_type(jnp.ones(3), jnp.int32))
    assert jnp.allclose(logdet, 0.0)


def test_logabsdet_conj_is_zero():
    """Conjugation is volume preserving: its real Jacobian is diag(1, -1)."""

    def f(x):
        return jax.lax.conj(x)

    y = jnp.array([1.0 + 2.0j, 3.0 - 1.0j])
    _, logdet = inverse_and_logabsdet(f)(y)
    assert jnp.allclose(logdet, 0.0)


@pytest.mark.parametrize("proj", [jnp.real, jnp.imag])
def test_logabsdet_projection_is_nan(proj):
    """real/imag discard a component, so no determinant exists.

    The input must be complex: on a real array these are the identity and JAX
    does not emit the primitive at all.
    """
    _, logdet = inverse_and_logabsdet(proj)(jnp.array([1.0 + 2.0j, 3.0 - 1.0j]))
    assert jnp.isnan(logdet)
