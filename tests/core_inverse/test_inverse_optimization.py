"""Guard the generated arithmetic and reuse of structural scheduling work."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.core import inverse, inverse_and_logabsdet
from probjax.core.jaxpr_propagation.engine import _EquationQueue


def test_elementwise_fanout_emits_only_the_handwritten_inverse():
    fn = lambda x: jnp.exp(3 * x - x)
    closed = jax.make_jaxpr(inverse(fn))(jnp.ones(100_000))
    assert [e.primitive.name for e in closed.jaxpr.eqns] == ["log", "div"]


@pytest.mark.parametrize("transform", [inverse, inverse_and_logabsdet])
def test_large_elementwise_section_never_constructs_a_jacobian(monkeypatch, transform):
    def forbidden(*args, **kwargs):
        raise AssertionError("elementwise inversion invoked dense linear algebra")

    monkeypatch.setattr(jax, "jacfwd", forbidden)
    monkeypatch.setattr(jnp.linalg, "solve", forbidden)
    monkeypatch.setattr(jnp.linalg, "slogdet", forbidden)
    fn = lambda x, c, b: jnp.exp(c * x - x + b)
    x = jnp.linspace(0.1, 0.2, 100_000)
    c, b = jnp.array(3.0), jnp.array(0.4)
    result = (
        jax.jit(transform(fn)).lower(fn(x, c, b), c, b).compile()(fn(x, c, b), c, b)
    )
    if transform is inverse_and_logabsdet:
        np.testing.assert_allclose(
            result[1], -jnp.sum(2 * x + b) - x.size * jnp.log(2.0), rtol=1e-6
        )
        result = result[0]
    np.testing.assert_allclose(result, x, atol=1e-6)


def test_broadcast_coefficients_and_reshapes_have_correct_multiplicity():
    def fn(x, c):
        z = (c * x - x).reshape(6)
        return jnp.exp(z + z).reshape(2, 3)

    x = jnp.arange(6.0).reshape(2, 3) / 20
    c = jnp.array([2.0, 3.0, 4.0])
    inv = inverse_and_logabsdet(fn)
    recovered, logdet = jax.jit(inv)(fn(x, c), c)
    np.testing.assert_allclose(recovered, x, atol=1e-6)
    reference = -jnp.linalg.slogdet(jax.jacfwd(fn)(x, c).reshape(6, 6))[1]
    np.testing.assert_allclose(logdet, reference, atol=1e-6)
    y = fn(x, c)
    np.testing.assert_allclose(
        jax.grad(lambda a: inv(y, a)[0].sum())(c),
        jax.grad(lambda a: (jnp.log(y) / (2 * (a - 1))).sum())(c),
        atol=1e-6,
    )


@pytest.mark.parametrize("with_logdet", [False, True])
def test_cached_normalized_graph_and_schedule_skip_queue_building(
    monkeypatch, with_logdet
):
    fn = lambda x, c: jnp.exp(c * jnp.exp(x) - jnp.exp(x))
    inv = (inverse_and_logabsdet if with_logdet else inverse)(fn)
    y, c = jnp.array([3.0, 4.0]), jnp.array(3.0)
    jax.make_jaxpr(inv)(y, c)

    def forbidden(*args, **kwargs):
        raise AssertionError("cached interpretation rebuilt a priority queue")

    monkeypatch.setattr(_EquationQueue, "_initialize", forbidden)
    # Fresh enclosing trace, same abstract signature, different runtime values.
    wrapped = lambda y, c: inv(y, c)
    result = jax.jit(wrapped)(y, c + 1)
    recovered = result[0] if with_logdet else result
    np.testing.assert_allclose(recovered, jnp.log(jnp.log(y) / 3), atol=1e-6)


def test_zero_diagonal_coefficients_report_invalid_coordinates():
    fn = lambda x, c: c * x - x
    result, logdet = jax.jit(inverse_and_logabsdet(fn))(
        jnp.ones(2), jnp.array([1.0, 3.0])
    )
    assert jnp.isnan(result[0])
    assert result[1] == 0.5
    assert jnp.isposinf(logdet)


def test_schedule_replay_falls_back_when_a_rule_changes_knownness():
    from probjax.core.jaxpr_propagation import propagate
    from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule

    closed = jax.make_jaxpr(lambda x: jnp.exp(x) + 1)(jnp.array(0.0))
    cache = {}
    allow = False
    forward = ForwardProcessingRule()

    def process(eqn, ins, outs, context):
        return forward(eqn, ins, outs) if allow else None

    def run():
        return propagate(
            closed.jaxpr,
            closed.consts,
            closed.jaxpr.invars,
            [jnp.array(0.0)],
            closed.jaxpr.outvars,
            process_eqn=process,
            schedule_cache=cache,
        )

    assert run() == [None]
    allow = True
    assert run()[0] == 2.0


def test_float32_coefficients_remain_float32_with_x64_enabled():
    with jax.enable_x64():
        y = jnp.ones(2, dtype=jnp.float32)
        result = jax.jit(inverse(lambda x: x + x))(y)
        assert result.dtype == jnp.float32
        np.testing.assert_allclose(result, 0.5)
