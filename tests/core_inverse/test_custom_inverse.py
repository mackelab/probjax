"""Custom inverse, vmap, JVP, cache, and multi-output tests.

Also covers custom_inverse logdet priority, re-registration invalidation,
and public API edge cases.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from probjax.core import custom_inverse, inverse, inverse_and_logabsdet


# ---------------------------------------------------------------------------
# Basic custom inverse
# ---------------------------------------------------------------------------


def test_inverse_of_custom_inverse_returns_custom_inverse():
    @custom_inverse
    def f(x):
        return 3.0 * x + 1.0

    f.definv(lambda y: (y - 1.0) / 3.0)
    f.definv_and_logdet(lambda y: ((y - 1.0) / 3.0, -jnp.asarray(2.5)))

    inv_f = inverse(f)
    assert isinstance(inv_f, custom_inverse)

    x0 = jnp.asarray(0.7)
    y0 = f(x0)
    assert jnp.allclose(inv_f(y0), x0, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(inverse(inv_f)(x0), y0, atol=1e-6, rtol=1e-6)


def test_inverse_of_custom_inverse_logdet_priority_and_fallback():
    @custom_inverse
    def g(x):
        return x + 1.0

    g.definv_and_logdet(lambda y: (y - 1.0, -jnp.asarray(1.0)))
    g.defvalue_and_logdet(lambda x: (x + 1.0, jnp.asarray(9.0)))
    inv_g = inverse(g)
    y_g, logdet_g = inverse_and_logabsdet(inv_g)(jnp.asarray(2.0))
    assert jnp.allclose(y_g, jnp.asarray(3.0), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet_g, jnp.asarray(9.0), atol=1e-6, rtol=1e-6)

    @custom_inverse
    def h(x):
        return 4.0 * x - 2.0

    h.definv_and_logdet(lambda y: ((y + 2.0) / 4.0, -jnp.asarray(7.0)))
    inv_h = inverse(h)
    y_h, logdet_h = inverse_and_logabsdet(inv_h)(jnp.asarray(0.5))
    assert jnp.allclose(y_h, jnp.asarray(0.0), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet_h, jnp.asarray(7.0), atol=1e-6, rtol=1e-6)


def test_inverse_of_custom_inverse_respects_configuration_guards():
    @custom_inverse
    def f(x):
        return x + 1.0

    f.definv_and_logdet(lambda y: (y - 1.0, jnp.asarray(0.0)))

    with pytest.raises(ValueError):
        inverse(f, invertible_arg=1)

    with pytest.raises(ValueError):
        inverse(f, static_argnums=(0,))


def test_inverse_of_custom_inverse_requires_registered_inverse():
    @custom_inverse
    def f(x):
        return x + 1.0

    with pytest.raises(AttributeError):
        inverse(f)


# ---------------------------------------------------------------------------
# vmap + custom inverse
# ---------------------------------------------------------------------------


def test_inverse_vmap_custom_inverse():
    @partial(custom_inverse, inv_argnum=1)
    def f(scale, x):
        return scale * x + 1.0

    f.definv(lambda scale, y: (y - 1.0) / scale)
    f.definv_and_logdet(
        lambda scale, y: (
            (y - 1.0) / scale,
            jnp.full_like(y, -jnp.log(jnp.abs(scale))),
        )
    )

    scale = 2.0
    xs = jnp.array([1.0, 2.0, 3.0])
    ys = jax.vmap(lambda x: f(scale, x))(xs)

    inv_vmap = inverse(jax.vmap(f, in_axes=(None, 0)), invertible_arg=1)
    x_rec = inv_vmap(scale, ys)
    assert jnp.allclose(x_rec, xs, atol=1e-6, rtol=1e-6)

    inv_vmap_jit = jax.jit(inv_vmap)
    x_rec_jit = inv_vmap_jit(scale, ys)
    assert jnp.allclose(x_rec_jit, xs, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_vmap_custom_inverse_logdet():
    @partial(custom_inverse, inv_argnum=1)
    def f(scale, x):
        return scale * x + 1.0

    f.definv_and_logdet(
        lambda scale, y: (
            (y - 1.0) / scale,
            jnp.full_like(y, -jnp.log(jnp.abs(scale))),
        )
    )

    scale = 2.0
    xs = jnp.array([1.0, 2.0, 3.0])
    ys = jax.vmap(lambda x: f(scale, x))(xs)

    inv_det_vmap = inverse_and_logabsdet(
        jax.vmap(f, in_axes=(None, 0)), invertible_arg=1
    )
    x_rec, logdet = inv_det_vmap(scale, ys)
    assert jnp.allclose(x_rec, xs, atol=1e-6, rtol=1e-6)
    expected_total = -xs.size * jnp.log(jnp.abs(scale))
    assert jnp.allclose(logdet, expected_total, atol=1e-6, rtol=1e-6)


def test_inverse_vmap_custom_inverse_nonzero_axis():
    @custom_inverse
    def f(x):
        return 2.0 * x + 1.0

    f.definv(lambda y: (y - 1.0) / 2.0)

    xs = jnp.arange(6.0).reshape(2, 3)
    ys = jax.vmap(f, in_axes=1, out_axes=1)(xs)

    inv_vmap = inverse(jax.vmap(f, in_axes=1, out_axes=1))
    x_rec = inv_vmap(ys)
    assert jnp.allclose(x_rec, xs, atol=1e-6, rtol=1e-6)

    inv_vmap_jit = jax.jit(inv_vmap)
    x_rec_jit = inv_vmap_jit(ys)
    assert jnp.allclose(x_rec_jit, xs, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_vmap_custom_inverse_nonzero_axis():
    @custom_inverse
    def f(x):
        return 2.0 * x + 1.0

    f.definv_and_logdet(
        lambda y: ((y - 1.0) / 2.0, jnp.full_like(y, -jnp.log(jnp.array(2.0))))
    )

    xs = jnp.arange(6.0).reshape(2, 3)
    ys = jax.vmap(f, in_axes=1, out_axes=1)(xs)

    inv_det_vmap = inverse_and_logabsdet(jax.vmap(f, in_axes=1, out_axes=1))
    x_rec, logdet = inv_det_vmap(ys)
    assert jnp.allclose(x_rec, xs, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, -jnp.log(2.0) * xs.size, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Multi-output custom inverse
# ---------------------------------------------------------------------------


def test_custom_inverse_multi_output():
    @custom_inverse
    def f(x):
        return 2.0 * x, x + 1.0

    f.definv_and_logdet(lambda y: (y[0] / 2.0, -jnp.log(2.0) * y[0].size))

    x = jnp.array([1.0, 2.0])
    y = f(x)
    x_rec = inverse(f)(y)
    x_rec_ld, logdet = inverse_and_logabsdet(f)(y)

    assert jnp.allclose(x_rec, x)
    assert jnp.allclose(x_rec_ld, x)
    assert jnp.allclose(logdet, -jnp.log(2.0) * x.size)


def test_custom_inverse_changes_pytree_structure():
    @custom_inverse
    def f(x):
        return {"scaled": 2.0 * x["a"], "shifted": x["b"] + 1.0}

    f.definv_and_logdet(
        lambda y: (
            {"a": y["scaled"] / 2.0, "b": y["shifted"] - 1.0},
            -jnp.log(2.0) * y["scaled"].size,
        )
    )

    x = {"a": jnp.array([0.2, 0.5]), "b": jnp.array([-1.0, 3.0])}
    y = f(x)
    inv = jax.jit(inverse(f))
    inv_and_det = jax.jit(inverse_and_logabsdet(f))

    x_rec = inv(y)
    x_rec_ld, logdet = inv_and_det(y)

    assert jax.tree.all(jax.tree.map(jnp.allclose, x_rec, x))
    assert jax.tree.all(jax.tree.map(jnp.allclose, x_rec_ld, x))
    expected_logdet = -jnp.log(2.0) * x["a"].size
    assert jnp.allclose(logdet, expected_logdet)


def test_custom_inverse_multi_output_nested_vmap():
    @custom_inverse
    def f(x):
        return 2.0 * x[0], 3.0 * x[1]

    f.definv_and_logdet(
        lambda y: (
            (y[0] / 2.0, y[1] / 3.0),
            (
                jnp.full_like(y[0], -jnp.log(jnp.asarray(2.0))),
                jnp.full_like(y[1], -jnp.log(jnp.asarray(3.0))),
            ),
        )
    )

    mapped_once = jax.vmap(f, in_axes=((1, 1),), out_axes=(1, 1))
    mapped = jax.vmap(mapped_once, in_axes=((0, 0),), out_axes=(0, 0))
    x = (
        jnp.arange(24.0).reshape(2, 3, 4),
        jnp.arange(24.0, 48.0).reshape(2, 3, 4),
    )
    y = mapped(x)

    x_rec = jax.jit(inverse(mapped))(y)
    x_rec_ld, logdet = jax.jit(inverse_and_logabsdet(mapped))(y)

    assert jax.tree.all(jax.tree.map(jnp.allclose, x_rec, x))
    assert jax.tree.all(jax.tree.map(jnp.allclose, x_rec_ld, x))
    expected_logdet = -x[0].size * (jnp.log(2.0) + jnp.log(3.0))
    assert jnp.allclose(logdet, expected_logdet)


@pytest.mark.xfail(
    strict=True,
    reason="a reduced logdet cannot separate mapped and unmapped target terms",
)
def test_custom_inverse_vmap_mixed_target_axes_logdet():
    @custom_inverse
    def f(x):
        return 2.0 * x[0], 3.0 * x[1]

    f.definv_and_logdet(
        lambda y: (
            (y[0] / 2.0, y[1] / 3.0),
            (
                jnp.full_like(y[0], -jnp.log(jnp.asarray(2.0))),
                jnp.full_like(y[1], -jnp.log(jnp.asarray(3.0))),
            ),
        )
    )

    mapped = jax.vmap(f, in_axes=((1, None),), out_axes=(1, None))
    x = (jnp.arange(6.0).reshape(2, 3), jnp.arange(3.0))
    y = mapped(x)

    x_rec, logdet = jax.jit(inverse_and_logabsdet(mapped))(y)

    assert jax.tree.all(jax.tree.map(jnp.allclose, x_rec, x))
    expected_logdet = -x[0].size * jnp.log(2.0) - x[1].size * jnp.log(3.0)
    assert jnp.allclose(logdet, expected_logdet)


def test_custom_inverse_negative_static_argnum():
    @partial(custom_inverse, inv_argnum=1, static_argnums=(-2,))
    def f(mode, x):
        return 2.0 * x if mode == "double" else x

    f.definv_and_logdet(
        lambda mode, y: (
            y / 2.0 if mode == "double" else y,
            -jnp.log(2.0) * y.size if mode == "double" else jnp.asarray(0.0),
        )
    )

    x = jnp.array([1.0, 2.0])
    y = f("double", x)
    inv = jax.jit(
        inverse(f, static_argnums=(-2,), invertible_arg=1), static_argnums=0
    )

    assert jnp.allclose(inv("double", y), x)


# ---------------------------------------------------------------------------
# Vector / per-element logdet
# ---------------------------------------------------------------------------


def test_custom_inverse_vector_logdet_reduction():
    a = jnp.array([2.0, 3.0, 4.0])

    @custom_inverse
    def f(x):
        return a * x

    f.definv_and_logdet(lambda y: (y / a, -jnp.log(a)))

    def g(x):
        return jnp.exp(f(x))

    x = jnp.array([0.5, 1.0, 1.5])
    y = g(x)

    x_rec, logdet = inverse_and_logabsdet(g)(y)
    assert jnp.allclose(x_rec, x, atol=1e-6, rtol=1e-6)

    expected = -(jnp.sum(jnp.log(a)) + jnp.sum(a * x))
    assert jnp.allclose(logdet, expected, atol=1e-6, rtol=1e-6)


def test_custom_inverse_definv_nan_logdet():
    @custom_inverse
    def f(x):
        return 2.0 * x

    f.definv(lambda y: y / 2.0)

    x = jnp.array([1.0, 2.0, 3.0])
    y = f(x)

    x_rec = inverse(f)(y)
    assert jnp.allclose(x_rec, x, atol=1e-6, rtol=1e-6)

    x_rec_ld, logdet = inverse_and_logabsdet(f)(y)
    assert jnp.allclose(x_rec_ld, x, atol=1e-6, rtol=1e-6)
    assert jnp.isnan(logdet)


# ---------------------------------------------------------------------------
# JVP / cache / re-registration
# ---------------------------------------------------------------------------


def test_custom_inverse_jvp_matches_analytical():
    @custom_inverse
    def f(x):
        return 3.0 * x + 1.0

    f.definv(lambda y: (y - 1.0) / 3.0)

    x = jnp.array([0.2, 0.5, 0.8])
    t = jnp.array([1.0, -1.0, 2.0])

    _, dy = jax.jvp(f, (x,), (t,))
    assert jnp.allclose(dy, 3.0 * t, atol=1e-6, rtol=1e-6)


def test_custom_inverse_re_registration_invalidates_cache():
    @custom_inverse
    def f(x):
        return 2.0 * x + 1.0

    f.definv_and_logdet(lambda y: ((y - 1.0) / 2.0, jnp.asarray(-1.0)))

    x = jnp.array([1.0, 2.0])
    y = f(x)

    jit_inv_a = jax.jit(inverse_and_logabsdet(f))
    _, logdet_a = jit_inv_a(y)
    assert jnp.allclose(logdet_a, jnp.asarray(-1.0), atol=1e-6, rtol=1e-6)

    # Re-registering the same kind now warns; that is exactly what this test
    # is doing on purpose, so accept it rather than let it leak into the run.
    with pytest.warns(RuntimeWarning, match="called twice"):
        f.definv_and_logdet(lambda y: ((y - 1.0) / 2.0, jnp.asarray(-99.0)))

    jit_inv_b = jax.jit(inverse_and_logabsdet(f))
    _, logdet_b = jit_inv_b(y)
    assert jnp.allclose(logdet_b, jnp.asarray(-99.0), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Public API edge cases
# ---------------------------------------------------------------------------


def test_inverse_wrapper_cache_across_signatures():
    def f(x):
        return 2.0 * x + 1.0

    inv_f = inverse(f)

    x0 = jnp.array(0.5, dtype=jnp.float32)
    y0 = f(x0)
    x0_rec = inv_f(y0)
    assert jnp.allclose(x0_rec, x0, atol=1e-6, rtol=1e-6)

    x1 = jnp.array([0.1, -0.3, 1.2], dtype=jnp.float32)
    y1 = f(x1)
    x1_rec = inv_f(y1)
    assert jnp.allclose(x1_rec, x1, atol=1e-6, rtol=1e-6)

    x2 = jnp.array(2, dtype=jnp.int32)
    y2 = f(x2)
    x2_rec = inv_f(y2)
    assert jnp.allclose(x2_rec, x2, atol=1e-6, rtol=1e-6)


def test_invertible_arg_negative_and_invalid():
    def f(scale, x):
        return scale * x

    scale = jnp.array(2.5)
    x = jnp.array(1.5)
    y = f(scale, x)

    inv_f_neg1 = inverse(f, invertible_arg=-1)
    x_rec = inv_f_neg1(scale, y)
    assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6)

    inv_f_neg2 = inverse(f, invertible_arg=-2)
    scale_rec = inv_f_neg2(y, x)
    assert jnp.allclose(scale, scale_rec, atol=1e-6, rtol=1e-6)

    with pytest.raises(IndexError, match="out of range"):
        inverse(f, invertible_arg=-3)(y, x)

    with pytest.raises(IndexError, match="out of range"):
        inverse(f, invertible_arg=2)(y, x)


def test_inverse_static_argnums_success():
    def f(mode, x, scale):
        return jax.lax.cond(
            mode,
            lambda t: scale * t,
            lambda t: t,
            x,
        )

    x = jnp.array([0.3, -1.2, 2.5])
    scale = jnp.array(2.0)
    y = f(True, x, scale)

    inv_f = inverse(f, static_argnums=(0,), invertible_arg=1)
    x_rec = inv_f(True, y, scale)
    assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6)

    inv_f_jit = jax.jit(inv_f, static_argnums=(0,))
    x_rec_jit = inv_f_jit(True, y, scale)
    assert jnp.allclose(x, x_rec_jit, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_static_argnums_success():
    def f(mode, x, scale):
        return jax.lax.cond(
            mode,
            lambda t: scale * t,
            lambda t: t,
            x,
        )

    x = jnp.array([0.3, -1.2, 2.5])
    scale = jnp.array(2.0)
    y = f(True, x, scale)

    inv_and_det = inverse_and_logabsdet(f, static_argnums=(0,), invertible_arg=1)
    x_rec, logdet = inv_and_det(True, y, scale)
    expected_logdet = -x.size * jnp.log(jnp.abs(scale))

    assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, expected_logdet, atol=1e-6, rtol=1e-6)

    inv_and_det_jit = jax.jit(inv_and_det, static_argnums=(0,))
    x_rec_jit, logdet_jit = inv_and_det_jit(True, y, scale)
    assert jnp.allclose(x, x_rec_jit, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet_jit, expected_logdet, atol=1e-6, rtol=1e-6)


def test_inverse_apis_support_multiple_and_negative_static_argnums():
    def f(mode, x, offset, scale):
        if mode == "affine":
            return scale * x + offset
        return x

    x = jnp.array([0.3, -1.2, 2.5])
    offset = 1.25
    scale = jnp.array(2.0)
    output = f("affine", x, offset, scale)
    static_argnums = (0, -2)

    inv = inverse(f, static_argnums=static_argnums, invertible_arg=1)
    inv_and_det = inverse_and_logabsdet(
        f, static_argnums=static_argnums, invertible_arg=1
    )
    jit_inv = jax.jit(inv, static_argnums=(0, 2))
    jit_inv_and_det = jax.jit(inv_and_det, static_argnums=(0, 2))

    assert jnp.allclose(jit_inv("affine", output, offset, scale), x)
    recovered, logdet = jit_inv_and_det("affine", output, offset, scale)
    assert jnp.allclose(recovered, x)
    assert jnp.allclose(logdet, -x.size * jnp.log(scale))


def test_inverse_kwargs_scale_shift():
    def f(x, *, scale, shift):
        return scale * x + shift

    x = jnp.array([0.5, -0.3, 1.2])
    scale = jnp.array(2.0)
    shift = jnp.array(1.0)
    y = f(x, scale=scale, shift=shift)

    inv_f = inverse(f)
    x_rec = inv_f(y, scale=scale, shift=shift)
    assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6)

    inv_f_jit = jax.jit(inv_f)
    x_rec_jit = inv_f_jit(y, scale=scale, shift=shift)
    assert jnp.allclose(x, x_rec_jit, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_kwargs():
    def f(x, *, scale, shift):
        return scale * x + shift

    x = jnp.array([0.5, -0.3, 1.2])
    scale = jnp.array(2.0)
    shift = jnp.array(1.0)
    y = f(x, scale=scale, shift=shift)

    inv_and_det = inverse_and_logabsdet(f)
    x_rec, logdet = inv_and_det(y, scale=scale, shift=shift)
    expected_logdet = -x.size * jnp.log(jnp.abs(scale))

    assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, expected_logdet, atol=1e-6, rtol=1e-6)

    inv_and_det_jit = jax.jit(inv_and_det)
    x_rec_jit, logdet_jit = inv_and_det_jit(y, scale=scale, shift=shift)
    assert jnp.allclose(x, x_rec_jit, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet_jit, expected_logdet, atol=1e-6, rtol=1e-6)


def test_inverse_pytree_dict_input():
    def f(d):
        return jax.tree_util.tree_map(lambda x: 2.0 * x, d)

    x = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    y = f(x)

    x_rec = inverse(f)(y)
    assert isinstance(x_rec, dict)
    assert jnp.allclose(x_rec["a"], x["a"], atol=1e-6, rtol=1e-6)
    assert jnp.allclose(x_rec["b"], x["b"], atol=1e-6, rtol=1e-6)


def test_inverse_unresolved_target_returns_nan():
    def f(x):
        return jnp.sum(x)

    x = jnp.array([1.0, 2.0, 3.0])
    y = f(x)

    assert jnp.isnan(inverse(f)(y))
