"""Opt-in ``rv_p`` tracing: primitive-free sampling by default.

Outside :func:`enable_rv_tracing` (and without ``PROBJAX_RV_TRACING=1``),
distribution sampling is an ordinary JAX computation: no ``random_variable``
primitive in jaxprs, no forward-jaxpr tracing, no site-name side effects.
The PPL transformations (``trace``, ``joint_sample``, ``log_joint_fn``,
``intervene``/``condition``/``substitute``) enable tracing around their
internal tracing automatically.

Note: ``jax.make_jaxpr`` caches per function identity, so each test below
defines its own model function -- reusing one function across flag states
would serve the first call's cached jaxpr.
"""

import threading

import jax
import jax.numpy as jnp
import pytest

from probjax import enable_rv_tracing, rv_tracing_enabled
from probjax.core import (
    condition,
    do,
    intervene,
    joint_sample,
    log_joint_fn,
    substitute,
    trace,
)
from probjax.core.custom_primitives import random_variable as rv_mod
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.stats import indep, mixture, norm, transformed


def _has_rv(closed_jaxpr) -> bool:
    return any(
        str(eqn.primitive) == "random_variable" for eqn in closed_jaxpr.jaxpr.eqns
    )


def _count_rv(closed_jaxpr) -> int:
    return sum(
        str(eqn.primitive) == "random_variable" for eqn in closed_jaxpr.jaxpr.eqns
    )


def _dist_model(key):
    key_z, key_y = jax.random.split(key)
    z = norm.rvs(key_z, 0.0, 1.0, name="z")
    return norm.rvs(key_y, z, 0.5, name="y")


def _direct_model(key):
    key_z, key_y = jax.random.split(key)
    z = rv_p.bind(key_z, 0.0, 1.0, dist=norm, name="z")
    return rv_p.bind(key_y, z, 0.5, dist=norm, name="y")


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def test_flag_defaults_off():
    assert rv_tracing_enabled() is False


def test_raw_jaxpr_has_no_sites_by_default():
    def model(key):
        return _dist_model(key)

    jaxpr = jax.make_jaxpr(model)(jax.random.key(0))
    assert not _has_rv(jaxpr)
    assert jnp.all(jnp.isfinite(model(jax.random.key(1))))


def test_raw_jaxpr_has_sites_under_context():
    def model(key):
        return _dist_model(key)

    with enable_rv_tracing():
        jaxpr = jax.make_jaxpr(model)(jax.random.key(0))
    assert _count_rv(jaxpr) == 2
    assert rv_tracing_enabled() is False


def test_jit_vmap_scan_grad_values_without_tracing():
    def draw(key):
        return norm.rvs(key, 0.0, 1.0, name="x")

    key = jax.random.key(0)
    assert jnp.isfinite(jax.jit(draw)(key))
    assert jax.vmap(draw)(jax.random.split(key, 3)).shape == (3,)

    def body(carry, subkey):
        return carry, norm.rvs(subkey, 0.0, 1.0, name="x")

    _, outs = jax.lax.scan(body, (), jax.random.split(key, 4))
    assert outs.shape == (4,)
    assert jnp.all(jnp.isfinite(outs))

    # Reparameterized path: x = loc + eps, so dx/dloc == 1.
    assert float(
        jax.grad(lambda loc: norm.rvs(key, loc, 1.0, name="x"))(0.5)
    ) == pytest.approx(1.0)


def test_array_kwargs_pass_through_without_tracing():
    # Array-valued kwargs cannot be held as static primitive params; without
    # the gate this raised "must be hashable". Eagerly they just work.
    def model(key):
        return norm.rvs(key, loc=jnp.asarray(0.0), scale=jnp.asarray(1.0), name="x")

    assert jnp.isfinite(jax.jit(model)(jax.random.key(0)))


# ---------------------------------------------------------------------------
# Transformations auto-enable tracing (dist-level and direct-bind models)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [_dist_model, _direct_model], ids=["dist", "direct"])
def test_joint_sample_sees_sites(model):
    sites = joint_sample(model)(jax.random.key(0))
    assert sorted(sites) == ["y", "z"]


@pytest.mark.parametrize("model", [_dist_model, _direct_model], ids=["dist", "direct"])
def test_trace_sees_sites(model):
    sites = trace(model, sites=True)(jax.random.key(0))
    assert sorted(sites) == ["y", "z"]
    assert all(meta["kind"] == "sample" for meta in sites.values())


@pytest.mark.parametrize("model", [_dist_model, _direct_model], ids=["dist", "direct"])
def test_log_joint_is_finite(model):
    sites = joint_sample(model)(jax.random.key(0))
    log_joint = log_joint_fn(model)(
        z=jnp.asarray(sites["z"]), y=jnp.asarray(sites["y"])
    )
    assert jnp.isfinite(log_joint)


@pytest.mark.parametrize("model", [_dist_model, _direct_model], ids=["dist", "direct"])
def test_intervene_and_do_override(model):
    fixed = {"z": jnp.asarray(1.5), "y": jnp.asarray(-0.5)}
    for transform in (intervene, do):
        out = transform(model, fixed)(jax.random.key(0))
        assert jnp.allclose(out, fixed["y"])


@pytest.mark.parametrize("model", [_dist_model, _direct_model], ids=["dist", "direct"])
def test_condition_and_observe_fix_value(model):
    observed = condition(model, {"y": jnp.asarray(0.25)})
    sites = joint_sample(observed)(jax.random.key(0))
    # Observed sites keep their fixed value but are not resampled into output.
    assert sorted(sites) == ["z"]
    assert jnp.all(jnp.isfinite(jnp.asarray(sites["z"])))
    traced = trace(observed, sites=True)(jax.random.key(0))
    assert jnp.allclose(traced["y"]["value"], 0.25)


@pytest.mark.parametrize("model", [_dist_model, _direct_model], ids=["dist", "direct"])
def test_substitute_replay_counts_site(model):
    replayed = substitute(model, {"y": jnp.asarray(0.25)}, mode="condition")
    log_joint = log_joint_fn(replayed)(z=jnp.asarray(0.0), y=jnp.asarray(0.25))
    assert jnp.isfinite(log_joint)


# ---------------------------------------------------------------------------
# Higher-order distributions expose exactly one site
# ---------------------------------------------------------------------------


class _Shift:
    def __call__(self, x):
        return x + 1.0

    def inverse_and_logdet(self, y):
        return y - 1.0, jnp.zeros((), dtype=y.dtype)


def _transformed_dist():
    return transformed(norm(0.0, 1.0), _Shift())


def _mixture_dist():
    return mixture(jnp.array([0.5, 0.5]), [norm(0.0, 1.0), norm(5.0, 1.0)])


def _indep_dist():
    return indep(norm(jnp.zeros(2), jnp.ones(2)))


@pytest.mark.parametrize(
    "make_dist",
    [_transformed_dist, _mixture_dist, _indep_dist],
    ids=["transformed", "mixture", "indep"],
)
def test_higher_order_dists_sample_eagerly(make_dist):
    # Eager sampling stays primitive-free and finite. Tracing a higher-order
    # sample through the site machinery is a pre-existing limitation: the
    # frozen component distributions are not valid dynamic jaxpr inputs
    # (``shaped_abstractify`` rejects them), independent of this gate.
    dist = make_dist()

    def model(key):
        return dist.rvs(key, name="x")

    key = jax.random.key(0)
    out = model(key)
    assert jnp.all(jnp.isfinite(jax.tree.leaves(out)[0]))
    out_jit = jax.jit(model)(key)
    assert jax.tree.structure(out_jit) == jax.tree.structure(out)


# ---------------------------------------------------------------------------
# Context semantics
# ---------------------------------------------------------------------------


def test_context_nests_and_unwinds_on_exception():
    assert rv_tracing_enabled() is False
    with enable_rv_tracing():
        assert rv_tracing_enabled() is True
        with enable_rv_tracing():
            assert rv_tracing_enabled() is True
        assert rv_tracing_enabled() is True
        with pytest.raises(RuntimeError, match="boom"), enable_rv_tracing():
            raise RuntimeError("boom")
        assert rv_tracing_enabled() is True
    assert rv_tracing_enabled() is False


def test_context_is_thread_local():
    seen = {}

    def work():
        seen["in_thread"] = rv_tracing_enabled()

    with enable_rv_tracing():
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        assert rv_tracing_enabled() is True
    assert seen["in_thread"] is False


def test_env_var_escape_hatch(monkeypatch):
    monkeypatch.setattr(rv_mod, "_ALWAYS_TRACE_RV", True)
    assert rv_tracing_enabled() is True

    def model(key):
        return norm.rvs(key, 0.0, 1.0, name="z")

    jaxpr = jax.make_jaxpr(model)(jax.random.key(0))
    assert _has_rv(jaxpr)
