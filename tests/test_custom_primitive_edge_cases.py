"""Edge cases for the two custom primitives: ``custom_inverse`` and ``rv_p``.

Both are well behaved across the standard transforms -- the matrix at the bottom
pins that. What they were not robust to was an argument that hides a tracer
where ``tree_leaves`` cannot see it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from probjax.core import custom_inverse
from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p
from probjax.core.custom_primitives.random_variable import rv_p
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


def test_random_variable_is_emitted_for_concrete_arguments_inside_a_trace():
    """Same rule for rv_p: a random variable must not vanish from the jaxpr.

    Every interpreter built on it -- trace, log_potential, intervene -- finds
    random variables by looking for this primitive.
    """
    key = jax.random.key(0)
    loc, scale = jnp.zeros(3), jnp.ones(3)

    def f():
        return rv_p.bind(key, loc, scale, dist=norm, shape=(3,))

    jaxpr = jax.make_jaxpr(f)()
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
