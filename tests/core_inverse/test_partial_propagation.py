"""Tests for PARTIAL value propagation and scheduling in the inverse interpreter.

Covers:
- Partial slices combined with forward reconstruction
- Disjoint partial slices that merge into a complete value
- Inconsistent complete inputs/outputs (engine-level conflict handling)
- Non-uniform select_n partial resolution
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
from probjax.core.jaxpr_propagation.engine import propagate
from probjax.core.registry import REGISTRY, Context


# ---------------------------------------------------------------------------
# Slice / dynamic_slice partial propagation
# ---------------------------------------------------------------------------


def test_inverse_disjoint_slices_scheduling():
    def f(x):
        part1 = jax.lax.dynamic_slice(x, (0,), (2,))
        part2 = jax.lax.dynamic_slice(x, (2,), (2,))
        return jnp.concatenate([part1, part2], axis=0)

    x0 = jnp.array([1.0, 2.0, 3.0, 4.0])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_partial_slice_then_forward():
    def f(x):
        part1 = jax.lax.dynamic_slice(x, (0,), (2,))
        part2 = jax.lax.dynamic_slice(x, (2,), (2,))
        combined = jnp.concatenate([part1, part2], axis=0)
        return jnp.exp(combined)

    x0 = jnp.array([1.0, 2.0, 3.0, 4.0])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Inconsistent complete inputs should not be overwritten by output authority
# ---------------------------------------------------------------------------


def test_inverse_inconsistent_complete_input_output():
    x = jnp.array(3.0)
    y = jnp.array(2.0)
    z_inconsistent = jnp.array(7.0)

    eqn = jax.make_jaxpr(lambda a, b: a * b)(x, y).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.mul_p, Context.INVERSE)

    # Both inputs known, output known but inconsistent
    result = rule(eqn, [x, y], [z_inconsistent])
    assert result is None, "Rule should return None when both inputs are already known"

    # Engine-level: existing COMPLETE inputs are preserved
    closed_jaxpr = jax.make_jaxpr(lambda a, b: a * b)(x, y)
    jaxpr = closed_jaxpr.jaxpr
    out = propagate(
        jaxpr,
        closed_jaxpr.consts,
        list(jaxpr.invars) + list(jaxpr.outvars),
        [x, y, z_inconsistent],
        list(jaxpr.invars),
        process_eqn=_make_inverse_processing_rule(),
        cost_fn=_get_inverse_cost_fn(),
        process_all_eqns=True,
    )
    assert jnp.allclose(out[0], x, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(out[1], y, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Non-uniform select_n
# ---------------------------------------------------------------------------


def test_inverse_uniform_select_n():
    def f(x):
        which = jnp.ones_like(x, dtype=jnp.int32)
        return jax.lax.select_n(which, -x, x)

    x = jnp.array([0.5, 1.2, 3.4])
    assert jnp.allclose(inverse(f)(f(x)), x, atol=1e-6, rtol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="non-uniform select_n inverse sets all unknown cases to the same PARTIAL placeholder",
)
def test_inverse_nonuniform_select_n():
    def f(x):
        which = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        return jax.lax.select_n(which, x, -x)

    x0 = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = f(x0)

    inv_f = inverse(f)
    x_rec = inv_f(y)
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


def test_logabsdet_select_n_is_zero():
    """select_n picks between operands elementwise; it scales nothing."""
    import jax.numpy as _jnp
    from probjax.core import inverse_and_logabsdet as _ild

    def f(x):
        return _jnp.where(_jnp.array([True, False]), _jnp.exp(x), _jnp.exp(x))

    _, logdet = _ild(f)(_jnp.ones(2))
    assert _jnp.allclose(logdet, 0.0)
