"""Deprecation warnings for legacy API names.

Verifies that deprecated names (``observe``, ``intervene``,
``log_potential_fn``, ``log_prob_fn``) emit ``DeprecationWarning`` and
return identical results to their canonical counterparts, while canonical
names emit no warnings.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from probjax.core import (
    condition,
    do,
    intervene,
    joint_sample,
    log_joint_fn,
    log_potential_fn,
    log_prob_fn,
    observe,
    trace,
)
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.stats import norm


def _make_model():
    """Build a simple two-site model."""
    def model(key):
        k1, k2 = jax.random.split(key)
        z = rv_p.bind(k1, 0.0, 1.0, dist=norm, name="z")
        y = rv_p.bind(k2, z, 0.5, dist=norm, name="y")
        return y
    return model


# ---------------------------------------------------------------------------
# observe vs condition
# ---------------------------------------------------------------------------


def test_observe_emits_deprecation():
    model = _make_model()
    with pytest.warns(DeprecationWarning, match="observe"):
        observe(model, {"y": jnp.array(0.25)})


def test_observe_no_deprecation():
    model = _make_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        condition(model, {"y": jnp.array(0.25)})


def test_observe_returns_same_result_as_condition():
    model = _make_model()
    obs = jnp.array(0.25)
    z_value = jnp.array(-0.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        conditioned = condition(model, {"y": obs})
        observed = observe(model, {"y": obs})

    got_cond = log_joint_fn(conditioned)(z=z_value)
    got_obs = log_joint_fn(observed)(z=z_value)
    assert jnp.allclose(got_cond, got_obs)


# ---------------------------------------------------------------------------
# intervene vs do
# ---------------------------------------------------------------------------


def test_intervene_emits_deprecation():
    model = _make_model()
    with pytest.warns(DeprecationWarning, match="intervene"):
        intervene(model, {"z": jnp.array(1.5)})


def test_intervene_no_deprecation():
    model = _make_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        do(model, {"z": jnp.array(1.5)})


def test_intervene_returns_same_result_as_do():
    model = _make_model()
    fixed = {"z": jnp.array(1.5), "y": jnp.array(-0.5)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        intervened = intervene(model, fixed)
        do_model = do(model, fixed)

    out_int = intervened(jax.random.PRNGKey(0))
    out_do = do_model(jax.random.PRNGKey(0))
    assert jnp.allclose(out_int, fixed["y"])
    assert jnp.allclose(out_do, fixed["y"])
    assert jnp.allclose(out_int, out_do)


# ---------------------------------------------------------------------------
# log_potential_fn vs log_joint_fn
# ---------------------------------------------------------------------------


def test_log_potential_fn_emits_deprecation():
    model = _make_model()
    samples = joint_sample(model)(jax.random.PRNGKey(0))
    with pytest.warns(DeprecationWarning, match="log_potential_fn"):
        log_potential_fn(model)(**samples)


def test_log_potential_fn_no_deprecation():
    model = _make_model()
    samples = joint_sample(model)(jax.random.PRNGKey(0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        log_joint_fn(model)(**samples)


def test_log_potential_fn_returns_same_result_as_log_joint_fn():
    model = _make_model()
    samples = joint_sample(model)(jax.random.PRNGKey(0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        got_lp = log_potential_fn(model)(**samples)
    got_ljf = log_joint_fn(model)(**samples)
    assert jnp.allclose(got_lp, got_ljf)


# ---------------------------------------------------------------------------
# log_prob_fn vs log_joint_fn
# ---------------------------------------------------------------------------


def test_log_prob_fn_emits_deprecation():
    model = _make_model()
    samples = joint_sample(model)(jax.random.PRNGKey(0))
    with pytest.warns(DeprecationWarning, match="log_prob_fn"):
        log_prob_fn(model)(**samples)


def test_log_prob_fn_no_deprecation():
    model = _make_model()
    samples = joint_sample(model)(jax.random.PRNGKey(0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        log_joint_fn(model)(**samples)


def test_log_prob_fn_returns_same_result_as_log_joint_fn():
    model = _make_model()
    samples = joint_sample(model)(jax.random.PRNGKey(0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        got_lpf = log_prob_fn(model)(**samples)
    got_ljf = log_joint_fn(model)(**samples)
    assert jnp.allclose(got_lpf, got_ljf)


# ---------------------------------------------------------------------------
# trace kind_labels="legacy"
# ---------------------------------------------------------------------------


def test_trace_legacy_emits_deprecation():
    model = _make_model()
    with pytest.warns(DeprecationWarning, match="legacy"):
        trace(model, sites=True, kind_labels="legacy")(jax.random.PRNGKey(0))


def test_trace_probabilistic_no_deprecation():
    model = _make_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        trace(model, sites=True, kind_labels="probabilistic")(jax.random.PRNGKey(0))
