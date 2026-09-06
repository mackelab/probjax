"""Edge cases for the two custom primitives: ``custom_inverse`` and ``rv_p``.

Both are well behaved across the standard transforms -- the matrix at the bottom
pins that. What they were not robust to was the boundary: arguments that hide a
tracer, inverses that return the wrong shape, and registration mistakes that
used to pass silently.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from probjax.core import custom_inverse, inverse, inverse_and_logabsdet
from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p
from probjax.core.custom_primitives.random_variable import enable_rv_tracing, rv_p
from probjax.stats import norm


def make_scale():
    """A minimal invertible primitive: x * s."""
    fun = custom_inverse(lambda x, s: x * s)
    fun.definv_and_logdet(lambda y, s: (y / s, -jnp.log(jnp.abs(s))))
    return fun


class Opaque:
    """A plain object holding an array. Deliberately not a registered pytree."""

    def __init__(self, value):
        self.value = value


# ---------------------------------------------------------------------------
# Tracers that tree_leaves cannot see
# ---------------------------------------------------------------------------


def test_primitive_is_emitted_even_when_no_argument_exposes_a_tracer():
    """The fast path must key on the trace being active, not on finding tracers.

    ``tree_leaves`` only walks registered pytree nodes, so a tracer held by a
    plain object is invisible. The call then ran eagerly *inside* a trace and
    emitted no primitive at all, which left the registered inverse unreachable
    -- silently, because the forward value was still correct.
    """
    scale = make_scale()

    def f(x):
        # Every argument is either an opaque object or a Python float, so
        # nothing here looks like a tracer to tree_leaves.
        return scale(Opaque(x).value, 2.0)

    jaxpr = jax.make_jaxpr(f)(jnp.ones(3))
    assert any(eqn.primitive is custom_inverse_call_p for eqn in jaxpr.jaxpr.eqns)


def test_random_variable_is_not_emitted_without_rv_tracing():
    """Default contract: sampling is an ordinary JAX computation.

    Outside ``enable_rv_tracing`` no ``random_variable`` primitive appears in
    the jaxpr, even inside a trace (``jit``/``vmap``/``make_jaxpr``). The
    value is still correct -- only the site metadata is gone.
    """
    key = jax.random.key(0)
    loc, scale = jnp.zeros(3), jnp.ones(3)

    def f():
        return rv_p.bind(key, loc, scale, dist=norm, shape=(3,))

    jaxpr = jax.make_jaxpr(f)()
    assert not any(str(eqn.primitive) == "random_variable" for eqn in jaxpr.jaxpr.eqns)
    assert jnp.all(jnp.isfinite(f()))


def test_random_variable_is_emitted_for_concrete_arguments_inside_a_trace():
    """Opt-in rule for rv_p: with tracing enabled, a random variable must not
    vanish from the jaxpr.

    Every interpreter built on it -- trace, log_potential, intervene -- finds
    random variables by looking for this primitive. (Distinct function object
    from the test above: ``make_jaxpr`` caches per function identity.)
    """
    key = jax.random.key(0)
    loc, scale = jnp.zeros(3), jnp.ones(3)

    def g():
        return rv_p.bind(key, loc, scale, dist=norm, shape=(3,))

    with enable_rv_tracing():
        jaxpr = jax.make_jaxpr(g)()
    assert any(str(eqn.primitive) == "random_variable" for eqn in jaxpr.jaxpr.eqns)


def test_eager_call_outside_a_trace_still_skips_the_primitive():
    """The fast path has to survive, or every concrete call pays for tracing."""
    scale = make_scale()
    out = scale(np.asarray([1.0, 2.0]), 2.0)
    assert isinstance(out, (np.ndarray, jax.Array))
    assert np.allclose(np.asarray(out), [2.0, 4.0])


def test_a_traced_closure_is_still_rejected_clearly():
    @jax.jit
    def build_and_call(s):
        fun = custom_inverse(lambda x: x * s)
        fun.definv_and_logdet(lambda y: (y / s, -jnp.log(jnp.abs(s))))
        return fun(jnp.ones(3))

    with pytest.raises(TypeError, match="closed over traced JAX values"):
        build_and_call(jnp.asarray(2.0))


# ---------------------------------------------------------------------------
# Arguments that are not arrays
# ---------------------------------------------------------------------------


def test_non_pytree_argument_names_itself_and_the_way_out():
    """Previously a raw "does not have a dtype attribute" from inside JAX."""
    fun = custom_inverse(lambda box, s: box.value * s)
    fun.definv_and_logdet(lambda y, s: (Opaque(y / s), -jnp.log(jnp.abs(s))))

    with pytest.raises(TypeError, match="static_argnums|pytree"):
        jax.make_jaxpr(lambda x: fun(Opaque(x), 2.0))(jnp.ones(3))


def test_traced_value_in_a_static_slot_says_why():
    fun = custom_inverse(lambda x, cfg: x * cfg, static_argnums=(1,))
    fun.definv_and_logdet(lambda y, cfg: (y / cfg, -jnp.log(jnp.abs(cfg))))

    with pytest.raises(TypeError, match="traced value"):
        jax.jit(fun)(jnp.ones(3), jnp.asarray(2.0))


def test_traced_keyword_argument_says_why():
    fun = custom_inverse(lambda x, *, s=2.0: x * s)
    fun.definv_and_logdet(lambda y, *, s=2.0: (y / s, -jnp.log(jnp.abs(s))))

    with pytest.raises(TypeError, match="traced value"):
        jax.jit(lambda x, s: fun(x, s=s))(jnp.ones(3), jnp.asarray(2.0))


# ---------------------------------------------------------------------------
# A registered inverse that does not match the argument it inverts
# ---------------------------------------------------------------------------


def test_inverse_returning_the_wrong_shape_is_rejected():
    """The tree matched, so this used to be accepted and returned (1,)."""
    fun = custom_inverse(lambda x, s: x * s)
    fun.definv_and_logdet(lambda y, s: (y[:1], jnp.asarray(0.0)))

    with pytest.raises(ValueError, match="shape"):
        inverse(lambda t: fun(t, 2.0))(jnp.ones(3))


def test_inverse_returning_the_wrong_pytree_is_rejected():
    fun = custom_inverse(lambda x, s: x * s)
    fun.definv_and_logdet(lambda y, s: ((y, y), jnp.asarray(0.0)))

    with pytest.raises(ValueError, match="invertible argument"):
        inverse(lambda t: fun(t, 2.0))(jnp.ones(3))


def test_a_correct_inverse_is_unaffected_by_the_checks():
    scale = make_scale()
    recovered, logdet = inverse_and_logabsdet(lambda t: scale(t, 2.0))(
        jnp.asarray([2.0, 4.0, 6.0])
    )
    assert jnp.allclose(recovered, jnp.asarray([1.0, 2.0, 3.0]))
    # The registered log-det is the scalar -log|s|; the primitive reports what
    # was registered rather than broadcasting it over the event.
    assert float(logdet) == pytest.approx(-float(jnp.log(2.0)))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_negative_inv_argnum_counts_from_the_end():
    """Matches ``inverse(..., invertible_arg=-1)``, which already allowed it."""
    fun = custom_inverse(lambda s, x: x * s, inv_argnum=-1)
    fun.definv_and_logdet(lambda s, y: (y / s, -jnp.log(jnp.abs(s))))
    assert jnp.allclose(jax.jit(fun)(2.0, jnp.ones(3)), 2.0)


def test_out_of_range_inv_argnum_reports_the_real_reason():
    fun = custom_inverse(lambda x, s: x * s, inv_argnum=5)
    fun.definv(lambda y, s: y / s)
    with pytest.raises(ValueError, match="out of range"):
        jax.jit(fun)(jnp.ones(3), 2.0)


def test_inv_argnum_pointing_at_a_static_argument_says_so():
    fun = custom_inverse(lambda x, s: x * s, inv_argnum=1, static_argnums=(1,))
    fun.definv(lambda y, s: y / s)
    # The static argument stays concrete; only x is traced, so this reaches the
    # inv_argnum check rather than the "traced value in a static slot" one.
    with pytest.raises(ValueError, match="static_argnums"):
        jax.jit(lambda x: fun(x, 2.0))(jnp.ones(3))


def test_registering_the_same_inverse_twice_warns():
    fun = custom_inverse(lambda x: x * 2.0)
    fun.definv(lambda y: y / 2.0)
    with pytest.warns(RuntimeWarning, match="called twice"):
        fun.definv(lambda y: y / 3.0)


def test_definv_then_definv_and_logdet_is_silent():
    """The intended way to register both; it must not warn."""
    fun = custom_inverse(lambda x: x * 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fun.definv(lambda y: y / 2.0)
        fun.definv_and_logdet(lambda y: (y / 2.0, -jnp.log(2.0)))


def test_calling_before_registering_an_inverse_raises():
    fun = custom_inverse(lambda x: x * 2.0)
    with pytest.raises(AttributeError, match="No inverse defined"):
        jax.jit(fun)(jnp.ones(3))


# ---------------------------------------------------------------------------
# Values at the boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(jnp.asarray(1.0), id="scalar_0d"),
        pytest.param(jnp.ones(0), id="empty"),
        pytest.param(jnp.ones((2, 3)), id="2d"),
    ],
)
def test_unusual_shapes_round_trip(value):
    scale = make_scale()
    forward = jax.jit(lambda t: scale(t, 2.0))(value)
    assert forward.shape == value.shape
    assert jnp.allclose(inverse(lambda t: scale(t, 2.0))(forward), value)


def test_integer_inverse_must_return_integers():
    """True division silently widens to float, which is a real mismatch.

    The invertible argument is int32, so an inverse producing float32 does not
    reconstruct it -- worth reporting rather than quietly changing the dtype.
    """
    widening = custom_inverse(lambda x, s: x * s)
    widening.definv_and_logdet(lambda y, s: (y / s, jnp.asarray(0.0)))
    with pytest.raises(ValueError, match="dtype"):
        inverse(lambda t: widening(t, 2))(jnp.ones(3, jnp.int32) * 2)

    exact = custom_inverse(lambda x, s: x * s)
    exact.definv_and_logdet(lambda y, s: (y // s, jnp.asarray(0.0)))
    value = jnp.ones(3, jnp.int32) * 2
    recovered = inverse(lambda t: exact(t, 2))(jax.jit(lambda t: exact(t, 2))(value))
    assert recovered.dtype == jnp.int32
    assert jnp.array_equal(recovered, value)


def test_forward_with_no_outputs_is_rejected():
    fun = custom_inverse(lambda x: ())
    fun.definv_and_logdet(lambda y: (jnp.zeros(3), jnp.asarray(0.0)))
    with pytest.raises(ValueError, match="at least one output"):
        jax.jit(fun)(jnp.ones(3))


# ---------------------------------------------------------------------------
# The transform matrix, as a regression net
# ---------------------------------------------------------------------------


def _sum_scaled(x):
    return jnp.sum(make_scale()(x, 2.0))


TRANSFORMS = {
    "jit": lambda f, a: jax.jit(f)(a),
    "grad": lambda f, a: jax.grad(f)(a),
    "jit_of_grad": lambda f, a: jax.jit(jax.grad(f))(a),
    "vmap": lambda f, a: jax.vmap(f)(jnp.stack([a, a])),
    "nested_vmap": lambda f, a: jax.vmap(jax.vmap(f))(jnp.ones((2, 2) + a.shape)),
    "grad_of_vmap": lambda f, a: jax.grad(lambda z: jnp.sum(jax.vmap(f)(z)))(
        jnp.stack([a, a])
    ),
    "jacfwd": lambda f, a: jax.jacfwd(f)(a),
    "jacrev": lambda f, a: jax.jacrev(f)(a),
    "hessian": lambda f, a: jax.hessian(f)(a),
    "linearize": lambda f, a: jax.linearize(f, a)[1](a),
    "vjp": lambda f, a: jax.vjp(f, a)[1](jnp.asarray(1.0)),
    "jvp": lambda f, a: jax.jvp(f, (a,), (a,)),
    "eval_shape": lambda f, a: jax.eval_shape(f, a),
    "remat": lambda f, a: jax.grad(jax.checkpoint(f))(a),
    "cond": lambda f, a: jax.lax.cond(True, f, f, a),
    "scan": lambda f, a: jax.lax.scan(lambda c, _: (c, f(c)), a, None, length=2)[1],
    "checkify": lambda f, a: checkify.checkify(f)(a),
    "pmap": lambda f, a: jax.pmap(f)(jnp.stack([a])),
}


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_custom_inverse_survives_every_transform(name):
    out = TRANSFORMS[name](_sum_scaled, jnp.arange(3.0) + 1.0)
    assert jax.tree.leaves(out)


def test_random_variable_survives_the_structural_transforms():
    def draw(key):
        return jnp.sum(rv_p.bind(key, jnp.zeros(3), jnp.ones(3), dist=norm, shape=(3,)))

    key = jax.random.key(0)
    assert jnp.isfinite(jax.jit(draw)(key))
    assert jax.vmap(draw)(jax.random.split(key, 2)).shape == (2,)
    assert jax.eval_shape(draw, key).shape == ()
    assert jnp.isfinite(jax.lax.cond(True, draw, draw, key))
