from types import SimpleNamespace

import jax
import pytest

import probjax.core.interpreters.inverse.dispatch as inverse_dispatch
from probjax.core.interpreters.inverse.dispatch import (
    DispatchAction,
    PrimitiveKind,
    classify_primitive,
    compute_knownness,
    select_dispatch_action,
)
from probjax.core.interpreters.inverse.registry import (
    BIVARIATE_INVERSE_REGISTRY,
    UNIVARIATE_INVERSE_REGISTRY,
    inverse_cost_fn,
)


def _eqn(primitive, n_invars, n_outvars):
    return SimpleNamespace(
        primitive=primitive,
        invars=[object() for _ in range(n_invars)],
        outvars=[object() for _ in range(n_outvars)],
    )


@pytest.mark.parametrize(
    "eqn,known_invars,known_outvars,expected_action,prefer_resolve_conflict,custom_rules",
    [
        (
            _eqn(jax.lax.reshape_p, 1, 1),
            [None],
            [1.0],
            DispatchAction.CUSTOM_RULE,
            True,
            {jax.lax.reshape_p: object()},
        ),
        (
            _eqn(jax.lax.add_p, 2, 1),
            [None, 1.0],
            [1.0],
            DispatchAction.BIVARIATE,
            True,
            {},
        ),
        (
            _eqn(jax.lax.dot_general_p, 2, 1),
            [None, 1.0],
            [1.0],
            DispatchAction.BIVARIATE,
            True,
            {},
        ),
        (
            _eqn(jax.lax.exp_p, 1, 1),
            [None],
            [1.0],
            DispatchAction.UNIVARIATE,
            True,
            {},
        ),
        (
            _eqn(jax.lax.sin_p, 1, 1),
            [1.0],
            [None],
            DispatchAction.FORWARD,
            True,
            {},
        ),
        (
            _eqn(inverse_dispatch.pjit_p, 1, 1),
            [None],
            [None],
            DispatchAction.PJIT_MISSING_INPUTS,
            True,
            {},
        ),
        (
            _eqn(jax.lax.exp_p, 1, 1),
            [1.0],
            [1.0],
            DispatchAction.RESOLVE_CONFLICT,
            True,
            {},
        ),
        (
            _eqn(jax.lax.exp_p, 1, 1),
            [1.0],
            [1.0],
            DispatchAction.UNIVARIATE,
            False,
            {},
        ),
        (
            _eqn(jax.lax.abs_p, 1, 1),
            [None],
            [None],
            DispatchAction.FAIL,
            True,
            {},
        ),
    ],
    ids=[
        "custom-rule",
        "bivariate",
        "dot-general-bivariate",
        "univariate",
        "forward",
        "pjit-missing-inputs",
        "resolve-conflict",
        "univariate-no-conflict",
        "fail",
    ],
)
def test_select_dispatch_action_paths(
    eqn,
    known_invars,
    known_outvars,
    expected_action,
    prefer_resolve_conflict,
    custom_rules,
):
    primitive_kind = classify_primitive(
        eqn,
        custom_rules=custom_rules,
        univariate_registry=UNIVARIATE_INVERSE_REGISTRY,
        bivariate_registry=BIVARIATE_INVERSE_REGISTRY,
    )
    knownness = compute_knownness(known_invars, known_outvars)
    action = select_dispatch_action(
        primitive_kind,
        knownness,
        prefer_resolve_conflict=prefer_resolve_conflict,
    )
    assert action is expected_action


def test_classify_custom_rule_precedes_univariate():
    eqn = _eqn(jax.lax.exp_p, 1, 1)
    primitive_kind = classify_primitive(
        eqn,
        custom_rules={jax.lax.exp_p: object()},
        univariate_registry=UNIVARIATE_INVERSE_REGISTRY,
        bivariate_registry=BIVARIATE_INVERSE_REGISTRY,
    )
    assert primitive_kind is PrimitiveKind.CUSTOM_RULE


def test_inverse_cost_cond_requires_known_predicate():
    eqn = _eqn(jax.lax.cond_p, 2, 1)

    unknown_predicate_cost = inverse_cost_fn(eqn, [False, False], [True])
    known_predicate_cost = inverse_cost_fn(eqn, [True, False], [True])

    assert unknown_predicate_cost == pytest.approx(float("inf"))
    assert float(known_predicate_cost) == pytest.approx(0.5)


def test_inverse_cost_scan_requires_known_sequences():
    def f(c0, xs):
        def body(carry, x):
            return carry + x, None

        carry_final, _ = jax.lax.scan(body, c0, xs)
        return carry_final

    eqn = jax.make_jaxpr(f)(
        jax.numpy.array(0.0), jax.numpy.array([1.0, 2.0])
    ).jaxpr.eqns[0]

    unknown_xs_cost = inverse_cost_fn(eqn, [False, False], [True])
    known_xs_cost = inverse_cost_fn(eqn, [False, True], [True])

    assert unknown_xs_cost == pytest.approx(float("inf"))
    assert float(known_xs_cost) == pytest.approx(0.5)


def test_inverse_cost_while_requires_anchor():
    def f(i0, x0):
        def cond(state):
            i, _ = state
            return i < 3

        def body(state):
            i, x = state
            return i + 1, x + 1.0

        return jax.lax.while_loop(cond, body, (i0, x0))

    eqn = jax.make_jaxpr(f)(
        jax.numpy.array(0, dtype=jax.numpy.int32),
        jax.numpy.array(1.0),
    ).jaxpr.eqns[0]

    no_anchor_cost = inverse_cost_fn(eqn, [False, False], [True, True])
    anchor_cost = inverse_cost_fn(eqn, [True, False], [True, True])

    assert no_anchor_cost == pytest.approx(float("inf"))
    assert float(anchor_cost) == pytest.approx(0.5)
