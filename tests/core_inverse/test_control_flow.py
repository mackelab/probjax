"""Control-flow primitive tests for the inverse interpreter.

Covers cond, switch, scan (carry-only and with outputs), while_loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probjax.core import inverse, inverse_and_logabsdet
from probjax.core.interpreters.inverse.rules import (
    _make_inverse_processing_rule,
    _get_inverse_cost_fn,
)
from probjax.core.registry import REGISTRY, Context


# ---------------------------------------------------------------------------
# cond / switch
# ---------------------------------------------------------------------------


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


def test_logabsdet_cond_then_exp_does_not_double_count():
    def f(pred, x):
        branch_output = jax.lax.cond(
            pred,
            lambda value: jnp.exp(value),
            lambda value: 3.0 * value,
            x,
        )
        return jnp.exp(branch_output)

    inv_and_det = inverse_and_logabsdet(f, invertible_arg=1)

    x_true = jnp.array(0.6)
    recovered_true, logdet_true = inv_and_det(True, f(True, x_true))
    expected_true = -(x_true + jnp.exp(x_true))

    x_false = jnp.array(-0.2)
    recovered_false, logdet_false = inv_and_det(False, f(False, x_false))
    expected_false = -(jnp.log(3.0) + 3.0 * x_false)

    assert jnp.allclose(recovered_true, x_true, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(logdet_true, expected_true, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(recovered_false, x_false, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(logdet_false, expected_false, atol=1e-5, rtol=1e-5)


def test_inverse_cond_under_jit():
    def f(pred, x):
        return jax.lax.cond(
            pred,
            lambda t: jnp.exp(t),
            lambda t: t + 2.5,
            x,
        )

    inv_f = jax.jit(inverse(f, invertible_arg=1))
    x_true = jnp.array(0.7)
    y_true = f(True, x_true)
    assert jnp.allclose(inv_f(True, y_true), x_true, atol=1e-6, rtol=1e-6)

    x_false = jnp.array(-1.3)
    y_false = f(False, x_false)
    assert jnp.allclose(inv_f(False, y_false), x_false, atol=1e-6, rtol=1e-6)


def test_inverse_switch_under_jit():
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

    inv_f = jax.jit(inverse(f, invertible_arg=1))
    for index, x in enumerate((0.2, -0.4, 1.5)):
        index = jnp.asarray(index, dtype=jnp.int32)
        x = jnp.asarray(x)
        assert jnp.allclose(inv_f(index, f(index, x)), x, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_cond_under_jit():
    def f(pred, x):
        return jax.lax.cond(
            pred,
            lambda value: jnp.exp(value),
            lambda value: 3.0 * value - 1.0,
            x,
        )

    inv_and_det = jax.jit(inverse_and_logabsdet(f, invertible_arg=1))

    x_true = jnp.array(0.6)
    recovered, logdet = inv_and_det(True, f(True, x_true))
    assert jnp.allclose(recovered, x_true, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, -x_true, atol=1e-6, rtol=1e-6)

    x_false = jnp.array(-0.2)
    recovered, logdet = inv_and_det(False, f(False, x_false))
    assert jnp.allclose(recovered, x_false, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, -jnp.log(3.0), atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_switch_under_jit():
    def f(index, x):
        return jax.lax.switch(
            index,
            (
                lambda value: value + 1.0,
                lambda value: jnp.exp(value),
                lambda value: 2.0 * value - 1.0,
            ),
            x,
        )

    inv_and_det = jax.jit(inverse_and_logabsdet(f, invertible_arg=1))
    expected_logdets = (0.0, -0.4, -jnp.log(2.0))
    for index, expected_logdet in enumerate(expected_logdets):
        index = jnp.asarray(index, dtype=jnp.int32)
        x = jnp.asarray(0.4)
        recovered, logdet = inv_and_det(index, f(index, x))
        assert jnp.allclose(recovered, x, atol=1e-6, rtol=1e-6)
        assert jnp.allclose(logdet, expected_logdet, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_identity_cond_branch_under_jit():
    def f(pred, x):
        return jax.lax.cond(pred, jnp.exp, lambda value: value, x)

    inv_and_det = jax.jit(inverse_and_logabsdet(f, invertible_arg=1))
    x = jnp.array(0.4)

    recovered, logdet = inv_and_det(False, f(False, x))
    assert jnp.allclose(recovered, x, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, 0.0, atol=1e-6, rtol=1e-6)

    recovered, logdet = inv_and_det(True, f(True, x))
    assert jnp.allclose(recovered, x, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, -x, atol=1e-6, rtol=1e-6)


def test_nested_runtime_control_flow_under_jit():
    def f(outer_index, inner_pred, x):
        return jax.lax.switch(
            outer_index,
            (
                lambda value: jax.lax.cond(
                    inner_pred,
                    jnp.exp,
                    lambda operand: 3.0 * operand,
                    value,
                ),
                lambda value: 2.0 * value - 1.0,
            ),
            x,
        )

    inv = jax.jit(inverse(f, invertible_arg=2))
    inv_and_det = jax.jit(inverse_and_logabsdet(f, invertible_arg=2))
    x = jnp.array(0.4)
    cases = (
        (0, False, -jnp.log(3.0)),
        (0, True, -x),
        (1, False, -jnp.log(2.0)),
    )

    for outer_index, inner_pred, expected_logdet in cases:
        outer_index = jnp.asarray(outer_index, dtype=jnp.int32)
        output = f(outer_index, inner_pred, x)
        assert jnp.allclose(
            inv(outer_index, inner_pred, output), x, atol=1e-6, rtol=1e-6
        )
        recovered, logdet = inv_and_det(outer_index, inner_pred, output)
        assert jnp.allclose(recovered, x, atol=1e-6, rtol=1e-6)
        assert jnp.allclose(logdet, expected_logdet, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


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


def test_inverse_scan_reverse_true():
    scale = 2.0

    def f(c0, xs):
        def body(carry, x):
            return scale * carry + x, None

        carry_final, _ = jax.lax.scan(body, c0, xs, reverse=True)
        return carry_final

    c0 = jnp.array(0.7)
    xs = jnp.array([0.2, -0.1, 0.4, 1.0])
    y = f(c0, xs)

    inv_f = inverse(f, invertible_arg=0)
    c_rec = inv_f(y, xs)
    assert jnp.allclose(c0, c_rec, atol=1e-6, rtol=1e-6)


def test_inverse_scan_zero_length():
    def f(c0, xs):
        def body(carry, x):
            return carry + x, None

        carry_final, _ = jax.lax.scan(body, c0, xs)
        return carry_final

    c0 = jnp.array(0.7)
    xs = jnp.array([], dtype=jnp.float32)
    y = f(c0, xs)

    inv_f = inverse(f, invertible_arg=0)
    c_rec = inv_f(y, xs)
    assert jnp.allclose(c0, c_rec, atol=1e-6, rtol=1e-6)

    inv_and_det = inverse_and_logabsdet(f, invertible_arg=0)
    c_rec2, logdet = inv_and_det(y, xs)
    assert jnp.allclose(c0, c_rec2, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, 0.0, atol=1e-6, rtol=1e-6)


def test_inverse_scan_multiple_carries():
    scale1 = 1.5
    scale2 = 2.0

    def f(c0, xs):
        def body(carry, x):
            c1, c2 = carry
            return (scale1 * c1 + x, scale2 * c2 - x), None

        return jax.lax.scan(body, c0, xs)

    c0 = (jnp.array(1.0), jnp.array(2.0))
    xs = jnp.array([0.1, 0.2, 0.3])
    carry_final, _ = f(c0, xs)

    eqn = jax.make_jaxpr(f)(c0, xs).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.scan_p, Context.INVERSE)
    result = rule(eqn, [None, None, xs], [carry_final[0], carry_final[1]])

    assert result.resolved_vars == [eqn.invars[0], eqn.invars[1]]
    assert jnp.allclose(result.resolved_vals[0], c0[0], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(result.resolved_vals[1], c0[1], atol=1e-5, rtol=1e-5)

    rule_logdet = REGISTRY.get(jax.lax.scan_p, Context.INVERSE_LOGDET)
    result_logdet = rule_logdet(
        eqn, [None, None, xs], [carry_final[0], carry_final[1]], context=None
    )

    assert result_logdet.resolved_vars == [eqn.invars[0], eqn.invars[1]]
    assert jnp.allclose(result_logdet.resolved_vals[0], c0[0], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(result_logdet.resolved_vals[1], c0[1], atol=1e-5, rtol=1e-5)

    expected_per_carry = {
        eqn.invars[0]: -xs.shape[0] * jnp.log(jnp.abs(scale1)),
        eqn.invars[1]: -xs.shape[0] * jnp.log(jnp.abs(scale2)),
    }
    for var in result_logdet.resolved_vars:
        assert var in expected_per_carry
        assert jnp.allclose(
            result_logdet.state.get(var, 0.0),
            expected_per_carry[var],
            atol=1e-5,
            rtol=1e-5,
        )


def test_inverse_scan_with_outputs_logdet_rejects():
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
    rule = REGISTRY.get(jax.lax.scan_p, Context.INVERSE_LOGDET)
    with pytest.raises(NotImplementedError, match="carry-only scans"):
        rule(eqn, [None, xs], [carry_final, ys], context=None)


# ---------------------------------------------------------------------------
# while_loop
# ---------------------------------------------------------------------------


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


@pytest.mark.xfail(
    strict=True,
    reason="public wrapper does not support multi-output while_loop (returns only out[0])",
)
def test_inverse_while_public_wrapper():
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

    inv_f = inverse(f, invertible_arg=1)
    x_rec = inv_f(i0, out_i, out_x)
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)

    inv_and_det = inverse_and_logabsdet(f, invertible_arg=1)
    x_rec2, logdet = inv_and_det(i0, out_i, out_x)
    assert jnp.allclose(x0, x_rec2, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, 0.0, atol=1e-6, rtol=1e-6)


def test_inverse_while_zero_iterations():
    def f(i0, x0):
        def cond(state):
            i, _ = state
            return i < 0

        def body(state):
            i, x = state
            return i + 1, x + 2.0

        return jax.lax.while_loop(cond, body, (i0, x0))

    i0 = jnp.array(5, dtype=jnp.int32)
    x0 = jnp.array(1.1)
    out_i, out_x = f(i0, x0)

    eqn = jax.make_jaxpr(f)(i0, x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.while_p, Context.INVERSE)
    result = rule(eqn, [i0, None], [out_i, out_x])

    assert result.resolved_vars == [eqn.invars[1]]
    assert jnp.allclose(result.resolved_vals[0], x0, atol=1e-6, rtol=1e-6)

    rule_logdet = REGISTRY.get(jax.lax.while_p, Context.INVERSE_LOGDET)
    result_logdet = rule_logdet(eqn, [i0, None], [out_i, out_x], context=None)

    assert result_logdet.resolved_vars == [eqn.invars[1]]
    assert jnp.allclose(result_logdet.resolved_vals[0], x0, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(result_logdet.state[eqn.invars[1]], 0.0, atol=1e-6, rtol=1e-6)


def test_inverse_while_no_anchor():
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
    with pytest.raises(NotImplementedError, match="at least one known input state"):
        rule(eqn, [None, None], [out_i, out_x])


def test_inverse_while_ambiguous_anchor():
    def f(i0, x0):
        def cond(state):
            i, _ = state
            return i < 1

        def body(state):
            i, x = state
            return i + 1, x * 0.0

        return jax.lax.while_loop(cond, body, (i0, x0))

    i0 = jnp.array(0, dtype=jnp.int32)
    x0 = jnp.array(2.0)
    out_i, out_x = f(i0, x0)

    eqn = jax.make_jaxpr(f)(i0, x0).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.while_p, Context.INVERSE)
    with pytest.raises(NotImplementedError, match="could not determine loop iteration count"):
        rule(eqn, [i0, None], [out_i, out_x])
