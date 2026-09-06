"""Opting out of ``custom_inverse`` handling, e.g. for a buggy inverse.

By default a ``custom_inverse`` wrapper emits ``custom_inverse_call_p`` under
tracing and ``inverse``/``inverse_and_logabsdet`` use the registered inverse.
Inside :func:`disable_custom_inverse` the wrapper is transparent: forward
calls run the plain function, and inversion falls back to structural rules.

Note: ``jax.make_jaxpr`` caches per function identity, so each test below
builds a fresh wrapper -- reusing one across flag states would serve the
first call's cached jaxpr.
"""

import sys
import threading

import jax
import jax.numpy as jnp
import pytest

from probjax import custom_inverse_enabled, disable_custom_inverse
from probjax.core import custom_inverse, inverse, inverse_and_logabsdet

# The module shares its name with the class, which shadows it as an attribute
# of the parent package -- resolve through sys.modules instead.
ci_mod = sys.modules["probjax.core.custom_primitives.custom_inverse"]


def _make_shift():
    f = custom_inverse(lambda x: x + 1.0)
    f.definv_and_logdet(lambda y: (y - 1.0, jnp.asarray(0.0)))
    return f


def _make_buggy_shift(error):
    """Shift-by-one whose registered inverse is off by ``error``."""
    f = custom_inverse(lambda x: x + 1.0)
    f.definv_and_logdet(lambda y: (y - 1.0 - error, jnp.asarray(0.0)))
    return f


def _prims(closed_jaxpr):
    return [str(eqn.primitive) for eqn in closed_jaxpr.jaxpr.eqns]


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def test_flag_defaults_on():
    assert custom_inverse_enabled() is True


def test_primitive_emitted_by_default():
    f = _make_shift()

    def g(x):
        return f(x)

    jaxpr = jax.make_jaxpr(g)(jnp.asarray(1.0))
    assert "custom_inverse_call_p" in _prims(jaxpr)


def test_no_primitive_when_disabled():
    f = _make_shift()

    def g(x):
        return f(x)

    with disable_custom_inverse():
        jaxpr = jax.make_jaxpr(g)(jnp.asarray(1.0))
        assert "custom_inverse_call_p" not in _prims(jaxpr)
        assert custom_inverse_enabled() is False
    assert custom_inverse_enabled() is True
    assert float(g(jnp.asarray(2.0))) == pytest.approx(3.0)


def test_forward_values_match_under_transforms():
    f = _make_shift()

    def g(x):
        return f(x)

    x = jnp.asarray(2.0)
    with disable_custom_inverse():
        assert float(jax.jit(g)(x)) == pytest.approx(3.0)
        assert jax.vmap(g)(jnp.stack([x, x])).shape == (2,)
        assert float(jax.grad(g)(x)) == pytest.approx(1.0)
    assert float(jax.jit(g)(x)) == pytest.approx(3.0)


def test_unregistered_wrapper_still_raises_when_disabled():
    f = custom_inverse(lambda x: x + 1.0)
    with disable_custom_inverse(), pytest.raises(AttributeError, match="No inverse"):
        f(jnp.asarray(1.0))


# ---------------------------------------------------------------------------
# The use case: bypassing a buggy registered inverse
# ---------------------------------------------------------------------------


def test_buggy_inverse_used_by_default():
    f = _make_buggy_shift(error=4.0)
    assert float(inverse(f)(jnp.asarray(4.0))) == pytest.approx(-1.0)


def test_buggy_inverse_bypassed_when_disabled():
    f = _make_buggy_shift(error=4.0)
    with disable_custom_inverse():
        # Hold the context across construction (strategy choice) and the
        # call (forward tracing): otherwise the primitive leaks back in.
        assert float(inverse(f)(jnp.asarray(4.0))) == pytest.approx(3.0)


def test_buggy_inverse_and_logdet_bypassed_when_disabled():
    f = _make_buggy_shift(error=4.0)
    with disable_custom_inverse():
        value, logdet = inverse_and_logabsdet(f)(jnp.asarray(4.0))
    assert float(value) == pytest.approx(3.0)
    assert float(logdet) == pytest.approx(0.0)


def test_direct_inv_calls_unaffected():
    # The opt-out governs tracing/inversion only; explicit calls to the
    # registered inverse still run it, bugs included.
    f = _make_buggy_shift(error=4.0)
    with disable_custom_inverse():
        assert float(f.inv(jnp.asarray(4.0))) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Context semantics
# ---------------------------------------------------------------------------


def test_context_nests_and_unwinds_on_exception():
    assert custom_inverse_enabled() is True
    with disable_custom_inverse():
        assert custom_inverse_enabled() is False
        with disable_custom_inverse():
            assert custom_inverse_enabled() is False
        assert custom_inverse_enabled() is False
        with pytest.raises(RuntimeError, match="boom"), disable_custom_inverse():
            raise RuntimeError("boom")
        assert custom_inverse_enabled() is False
    assert custom_inverse_enabled() is True


def test_context_is_thread_local():
    seen = {}

    def work():
        seen["in_thread"] = custom_inverse_enabled()

    with disable_custom_inverse():
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        assert custom_inverse_enabled() is False
    assert seen["in_thread"] is True


def test_env_var_escape_hatch(monkeypatch):
    monkeypatch.setattr(ci_mod, "_DISABLE_CUSTOM_INVERSE", True)
    assert custom_inverse_enabled() is False

    f = _make_shift()

    def g(x):
        return f(x)

    jaxpr = jax.make_jaxpr(g)(jnp.asarray(1.0))
    assert "custom_inverse_call_p" not in _prims(jaxpr)
