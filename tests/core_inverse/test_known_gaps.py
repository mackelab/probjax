"""Maps that are uniquely invertible by hand but the interpreter cannot do yet.

Contract of this file: every case below has a *unique* inverse, argued in a
comment per test. Each test asserts the correct roundtrip and is marked
``xfail(strict=True)``. A strict xfail that starts passing fails the suite,
so landing support for a case means deleting its marker here -- these are
executable TODOs, not permanent characterization.

Deliberately NOT listed: maps the design refuses on purpose -- nonlinear
fan-out without a contraction certificate (``x + tanh(x)``,
``x**3 + x``; pinned by ``test_nonlinear_fan_out_is_still_refused``),
non-injective maps (``abs``, ``clip``, ``sort``), and underdetermined maps
(``sum``, ``diff`` without an initial value).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import lax

from probjax.core import inverse

_XFAIL = pytest.mark.xfail(strict=True, reason="hand-invertible but not yet supported")


def _assert_roundtrip(fn, recovered, y):
    assert jnp.allclose(
        jnp.asarray(jax.tree.leaves(fn(recovered))),
        jnp.asarray(jax.tree.leaves(y)),
        atol=1e-4,
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


@_XFAIL
def test_fori_loop_of_translations():
    # Unrolls to y = x + 3, so x = y - 3 uniquely.
    fn = lambda t: lax.fori_loop(0, 3, lambda i, v: v + 1.0, t)  # noqa: E731
    y = jnp.array([4.0, 5.0, 6.0])
    _assert_roundtrip(fn, inverse(fn)(y), y)


@_XFAIL
def test_bounded_while_loop():
    # Exactly three iterations of (x + 1), so x = y - 3 uniquely.
    def fn(t):
        return lax.while_loop(
            lambda s: s[1] < 3.0,
            lambda s: (s[0] + 1.0, s[1] + 1.0),
            (t, 0.0),
        )[0]

    y = jnp.array([4.0, 5.0, 6.0])
    _assert_roundtrip(fn, inverse(fn)(y), y)


@_XFAIL
def test_scan_with_a_counter_carry():
    # The counter is a deterministic function of the trip count, so
    # x = carry - 3 uniquely; only the x-lane needs solving.
    def fn(t):
        def body(c, _):
            xx, i = c
            return (xx + 1.0, i + 1.0), xx

        (c, _), _ = lax.scan(body, (t, 0.0), None, length=3)
        return c[0]

    y = jnp.array([4.0, 5.0, 6.0])
    _assert_roundtrip(fn, inverse(fn)(y), y)


@_XFAIL
def test_lax_map_of_an_affine_map():
    # Elementwise y = 2t + 1, so t = (y - 1) / 2 uniquely.
    fn = lambda t: lax.map(lambda u: 2.0 * u + 1.0, t)  # noqa: E731
    y = jnp.array([3.0, 5.0, 7.0])
    _assert_roundtrip(fn, inverse(fn)(y), y)


@_XFAIL
def test_cond_with_a_recoverable_branch():
    # The +1 branch outputs y[0] > 1 and the -1 branch y[0] <= -1, so at
    # y[0] = 2 the branch is known after the fact and x = y - 1 uniquely.
    fn = lambda t: lax.cond(  # noqa: E731
        t[0] > 0, lambda u: u + 1.0, lambda u: u - 1.0, t
    )
    y = jnp.array([2.0, 3.0, 4.0])
    _assert_roundtrip(fn, inverse(fn)(y), y)


# ---------------------------------------------------------------------------
# Overdetermined but consistent without a template: the trace is the wrong
# program (input shaped like the output), so these report NaN. With
# input_template they invert -- see test_input_template_* in
# test_core_inverse_and_log_det.py.
# ---------------------------------------------------------------------------


@_XFAIL
def test_tile_without_template():
    # y = [x, x], so x = y[:n] uniquely -- but only with a template.
    fn = lambda t: jnp.tile(t, 2)  # noqa: E731
    y = jnp.array([1.0, 2.0, 1.0, 2.0])
    recovered = inverse(fn)(y)
    assert recovered.shape == (2,)
    _assert_roundtrip(fn, recovered, y)


@_XFAIL
def test_concatenate_of_a_variable_with_itself_without_template():
    # y = [x, x], so x = y[:n] uniquely -- but only with a template.
    fn = lambda t: jnp.concatenate([t, t])  # noqa: E731
    y = jnp.array([1.0, 2.0, 1.0, 2.0])
    recovered = inverse(fn)(y)
    assert recovered.shape == (2,)
    _assert_roundtrip(fn, recovered, y)


@_XFAIL
def test_pad_without_template():
    # Edge padding is known constants, so x = y[1:-1] uniquely -- but only
    # with a template.
    fn = lambda t: jnp.pad(t, 1)  # noqa: E731
    y = jnp.array([0.0, 1.0, 2.0, 3.0, 0.0])
    recovered = inverse(fn)(y)
    assert recovered.shape == (3,)
    _assert_roundtrip(fn, recovered, y)


# ---------------------------------------------------------------------------
# Missing joint reasoning (each piece inverts alone, the sum still stalls)
# ---------------------------------------------------------------------------


@_XFAIL
def test_maximum_plus_minimum_is_the_identity():
    # max(t, 0) + min(t, 0) == t pointwise for all reals, uniquely. Each
    # extremum now has a guarded rule, but the outer add still needs two
    # unknowns at once -- closing this needs joint piecewise reasoning.
    fn = lambda t: jnp.maximum(t, 0.0) + jnp.minimum(t, 0.0)  # noqa: E731
    y = jnp.array([1.0, -2.0, 3.0])
    _assert_roundtrip(fn, inverse(fn)(y), y)


# ---------------------------------------------------------------------------
# Tracing / API limitations
# ---------------------------------------------------------------------------


@_XFAIL
def test_structure_changing_split_without_template():
    # x = concat(y) uniquely, but without a template inverse traces the
    # function with its own output, so an array->tuple map cannot be staged.
    fn = lambda t: jnp.split(t, 2)  # noqa: E731
    y = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    recovered = inverse(fn)(y)
    assert jnp.allclose(recovered, jnp.arange(1.0, 5.0))
