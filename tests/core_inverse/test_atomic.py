"""Parameterized atomic-primitive tests for the inverse interpreter.

Covers:
- Round-trip recovery for bijective and principal-branch primitives
- Log-determinant correctness (golden = autodiff)
- Runtime-invalid outputs produce NaN (inexact) or checkify (exact)
- Registry coverage for all univariate / bivariate rules
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probjax.core import inverse, inverse_and_logabsdet
from probjax.core.registry import REGISTRY, Context

from .cases import (
    ATOMIC_ROUNDTRIP_CASES,
    PRINCIPAL_BRANCH_CASES,
    INVALID_OUTPUT_CASES,
    LOGDET_EXPLICIT_CASES,
    STRUCTURALLY_INVALID_CASES,
)
from .helpers import logdet_via_autodiff

# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ATOMIC_ROUNDTRIP_CASES, ids=lambda c: c.id)
def test_atomic_roundtrip(case):
    x = case.args_factory()
    y = case.forward(x)
    assert case.primitive in {eqn.primitive for eqn in jax.make_jaxpr(case.forward)(x).jaxpr.eqns}

    inv_fun = inverse(lambda z: case.forward(z))
    x_rec = jnp.asarray(inv_fun(y))

    assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6), (
        f"Round-trip failed for {case.id}: {x} vs {x_rec}"
    )


@pytest.mark.parametrize("case", PRINCIPAL_BRANCH_CASES, ids=lambda c: c.id)
def test_principal_branch_roundtrip(case):
    """Principal-branch inverses recover the original on the principal domain."""
    x = case.args_factory()
    y = case.forward(x)
    assert case.primitive in {eqn.primitive for eqn in jax.make_jaxpr(case.forward)(x).jaxpr.eqns}

    inv_fun = inverse(lambda z: case.forward(z))
    x_rec = jnp.asarray(inv_fun(y))

    # For even functions (x**2, cosh), the principal branch returns the non-negative value.
    if case.id in ("integer_pow_even", "cosh"):
        assert jnp.allclose(x_rec, jnp.abs(x), atol=1e-6, rtol=1e-6)
    else:
        assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Log-determinant tests (golden = autodiff)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", LOGDET_EXPLICIT_CASES, ids=lambda c: c.id)
def test_atomic_logdet(case):
    x = case.args_factory()
    y = case.forward(x)
    assert REGISTRY.get(case.primitive, Context.INVERSE_LOGDET) is not None

    def f(z):
        return case.forward(z)

    inv_and_det = inverse_and_logabsdet(f)
    x_rec, logdet = inv_and_det(y)

    assert jnp.allclose(x_rec, x, atol=1e-5, rtol=1e-5), (
        f"Recovery failed for {case.id}"
    )

    expected = logdet_via_autodiff(f, x)

    assert jnp.allclose(logdet, expected, atol=1e-5, rtol=1e-5), (
        f"Logdet mismatch for {case.id}: got {logdet}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Invalid-output tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", INVALID_OUTPUT_CASES, ids=lambda c: c.id)
def test_invalid_output_nan(case):
    """Runtime-invalid elements become NaN without poisoning valid elements."""
    y_invalid = case.output_factory()

    # The rule should be retrievable
    rule = REGISTRY.get(case.primitive, Context.INVERSE)
    assert rule is not None, f"No inverse rule for {case.id}"

    # Build a minimal eqn to call the rule directly
    x_dummy = jnp.zeros_like(y_invalid)
    eqn = jax.make_jaxpr(lambda z: case.forward(z))(x_dummy).jaxpr.eqns[0]

    result = rule(eqn, [None], [y_invalid])

    assert result is not None
    recovered = result.resolved_vals[0]
    assert jnp.allclose(recovered, case.expected_factory(), equal_nan=True)


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in INVALID_OUTPUT_CASES
        if REGISTRY.get(case.primitive, Context.INVERSE_LOGDET) is not None
    ],
    ids=lambda case: case.id,
)
def test_invalid_output_logdet_is_nan(case):
    output = case.output_factory()
    eqn = jax.make_jaxpr(case.forward)(jnp.zeros_like(output)).jaxpr.eqns[0]
    rule = REGISTRY.get(case.primitive, Context.INVERSE_LOGDET)
    result = rule(eqn, [None], [output])
    assert jnp.allclose(
        result.resolved_vals[0], case.expected_factory(), equal_nan=True
    )
    assert jnp.isnan(result.state[eqn.invars[0]])


@pytest.mark.parametrize("case", STRUCTURALLY_INVALID_CASES, ids=lambda c: c.id)
def test_structurally_noninjective_returns_nan(case):
    x = case.args_factory()
    y = case.forward(x)
    eqn = next(
        eqn
        for eqn in jax.make_jaxpr(case.forward)(x).jaxpr.eqns
        if eqn.primitive is case.primitive
    )
    rule = REGISTRY.get(case.primitive, Context.INVERSE)
    result = rule(eqn, [None], [y])
    assert result is not None
    assert result.resolved_vals[0].shape == x.shape
    assert jnp.all(jnp.isnan(result.resolved_vals[0]))


# ---------------------------------------------------------------------------
# Autodiff fallback logdet tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,fun",
    [
        ("pow", lambda x: jnp.power(x, 3.0)),
        ("tan", jnp.tan),
    ],
)
def test_autodiff_fallback_logdet(name, fun):
    x = jnp.linspace(0.1, 1.0, 7)
    y = fun(x)

    def f(z):
        return fun(z)

    inv_and_det = inverse_and_logabsdet(f)
    x_rec, logdet = inv_and_det(y)

    assert jnp.allclose(x_rec, x, atol=1e-5, rtol=1e-5)
    expected = logdet_via_autodiff(f, x)
    assert jnp.allclose(logdet, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Binary direction-isolation logdet tests
# ---------------------------------------------------------------------------

_BINARY_LOGDET_DIRECTION_CASES = [
    ("mul", lambda x, c: c * x, lambda x, c: -x.size * jnp.log(jnp.abs(c))),
    ("div", lambda x, c: x / c, lambda x, c: x.size * jnp.log(jnp.abs(c))),
    ("add", lambda x, c: x + c, lambda x, c: 0.0),
    ("c_sub_x", lambda x, c: c - x, lambda x, c: 0.0),
    ("x_sub_c", lambda x, c: x - c, lambda x, c: 0.0),
]


@pytest.mark.parametrize("name,fun,expected_fn", _BINARY_LOGDET_DIRECTION_CASES, ids=[t[0] for t in _BINARY_LOGDET_DIRECTION_CASES])
def test_binary_logdet_direction_isolation(name, fun, expected_fn):
    c = 2.5
    x = jnp.array([0.5, 1.0, -0.3])
    y = fun(x, c)

    def f(z):
        return fun(z, c)

    inv_and_det = inverse_and_logabsdet(f)
    x_rec, logdet = inv_and_det(y)

    expected_formula = expected_fn(x, c)
    expected_jac = logdet_via_autodiff(f, x)

    assert jnp.allclose(x_rec, x, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(logdet, expected_formula, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(logdet, expected_jac, atol=1e-5, rtol=1e-5)
