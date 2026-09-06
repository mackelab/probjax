"""Unit tests for the structural volume-preservation proof.

`is_volume_preserving` decides from the jaxpr alone whether a program has
|det J| = 1 in its target, so these tests call the predicate directly -- no
propagation engine involved. The bar is asymmetry-proof: False for anything
that scales, duplicates, drops, or indexes by the target; True only for
size-preserving rearrangements and pointwise unit-Jacobian ops (plus nesting
through `jit`, which the engine also descends into).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probjax.core.interpreters.inverse.affine import is_volume_preserving


def _check(fn, *args, targets=None):
    closed = jax.make_jaxpr(fn)(*args)
    if targets is None:
        targets = list(closed.jaxpr.invars)
    return is_volume_preserving(closed.jaxpr, targets)


def test_translations_and_sign_flips_preserve_volume():
    x = jnp.array([1.0, -2.0])
    assert _check(lambda t: -t, x)
    assert _check(lambda t: t + 1.0, x)
    assert _check(lambda t: t - jnp.asarray([0.5, 0.5]), x)
    assert _check(lambda t: jnp.conj(t), x.astype(jnp.complex64))


def test_unit_scalings_preserve_volume():
    x = jnp.array([1.0, -2.0])
    assert _check(lambda t: t * 1.0, x)
    assert _check(lambda t: -1.0 * t, x)
    assert _check(lambda t: t / -1.0, x)


def test_rearrangements_preserve_volume():
    assert _check(lambda t: jnp.flip(t), jnp.arange(4.0))
    assert _check(lambda t: jnp.transpose(jnp.reshape(t, (2, 2))), jnp.arange(4.0))
    assert _check(lambda t: jnp.reshape(t, (2, 2)).reshape(-1), jnp.arange(4.0))
    assert _check(lambda t: jnp.squeeze(t), jnp.zeros((1, 3)))
    assert _check(lambda t: t.astype(jnp.float32), jnp.arange(3.0))


def test_nested_jit_is_transparent():
    x = jnp.arange(4.0)
    assert _check(lambda t: jax.jit(lambda u: jnp.flip(u) + 1.0)(t), x)
    assert not _check(lambda t: jax.jit(lambda u: u + u)(t), x)
    assert not _check(lambda t: jax.jit(lambda u: 2.0 * u)(t), x)


def test_static_mask_pick_stays_on_the_general_path():
    mask = jnp.array([True, False, True])
    x = jnp.arange(3.0)
    assert not _check(lambda t: jnp.where(mask, t, -t), x)


def test_scaling_disqualifies():
    x = jnp.array([1.0, -2.0])
    assert not _check(lambda t: t + t, x)
    assert not _check(lambda t: t - t, x)
    assert not _check(lambda t: 2.0 * t, x)
    assert not _check(lambda t: t / 2.0, x)
    assert not _check(lambda t: t * t, x)


def test_target_dependent_scale_disqualifies():
    # ±1 held in a tracer is unprovable statically, so it stays slow-path.
    assert not _check(lambda t, c: t * c, jnp.ones(2), jnp.ones(2))
    assert not _check(lambda t, c: t / c, jnp.ones(2), jnp.ones(2))
    assert not _check(lambda t: 1.0 / t, jnp.ones(2))


def test_size_changing_ops_disqualify():
    assert not _check(lambda t: t[:2], jnp.arange(4.0))
    assert not _check(lambda t: (t, t), jnp.arange(2.0))
    assert not _check(lambda t: jnp.concatenate([t, t]), jnp.arange(2.0), targets=None)
    assert not _check(lambda t: jnp.sum(t), jnp.arange(4.0))
    assert not _check(
        lambda t: jnp.broadcast_to(t, (2, 2)), jnp.arange(2.0), targets=None
    )


def test_index_dependent_ops_disqualify():
    x = jnp.arange(4.0)
    assert not _check(lambda t, i: t[i], x, jnp.asarray(1))
    assert not _check(lambda t: jax.lax.dynamic_slice(t, (1,), (2,)), x, targets=None)


def test_target_dependent_selector_disqualifies():
    # A selector computed from the target makes the pick nonlinear.
    assert not _check(
        lambda t: jnp.where(t > 0, t, -t), jnp.array([1.0, -2.0]), targets=None
    )


def test_nonlinear_and_coupled_ops_disqualify():
    assert not _check(lambda t: jnp.exp(t), jnp.array([1.0]))
    assert not _check(lambda t: jnp.tanh(t), jnp.array([1.0]))
    assert not _check(lambda t: jnp.sum(t) * jnp.ones(2) - t, jnp.ones(2))


def test_scan_body_is_not_descended_into():
    # Conservative like the engine: scan carries state, so even a VP body
    # keeps the slow path.
    def f(t):
        def body(carry, _):
            return carry + t, None

        out, _ = jax.lax.scan(body, jnp.zeros(2), None, length=3)
        return out

    assert not _check(f, jnp.ones(2))


def test_empty_targets_are_not_volume_preserving():
    closed = jax.make_jaxpr(lambda t: -t)(jnp.ones(2))
    assert not is_volume_preserving(closed.jaxpr, [])
