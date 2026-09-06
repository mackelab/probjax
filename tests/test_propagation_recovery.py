"""The engine's optional recovery protocol is independent of inverse algebra."""

import jax
import jax.numpy as jnp
import pytest

from probjax.core.jaxpr_propagation import StallRecoveryResult, propagate
from probjax.core.jaxpr_propagation.utils import Knowness, KnownessLevel


def test_recovery_requeues_neighbors_and_preserves_namespaced_state():
    closed = jax.make_jaxpr(jnp.exp)(jnp.array(1.0))
    (x,) = closed.jaxpr.invars
    calls = []

    def recover(context, targets, changed):
        calls.append(changed.copy())
        return StallRecoveryResult([x], [jnp.array(1.0)], state_updates={"new": 2})

    values, state, env = propagate(
        closed.jaxpr,
        closed.consts,
        [],
        [],
        closed.jaxpr.outvars,
        stall_recovery=recover,
        return_state=True,
        return_env=True,
        initial_state={"old": 1},
        state_namespace="test",
    )
    assert jnp.allclose(values[0], jnp.exp(1.0))
    assert len(calls) == 1
    assert state == {"old": 1, "new": 2}
    assert env.get_knowness_level(x) == KnownessLevel.COMPLETE
    assert env.read_run_state("test") == state


def test_no_progress_terminates():
    closed = jax.make_jaxpr(jnp.exp)(jnp.array(1.0))
    calls = []

    def recover(context, targets, changed):
        calls.append(1)
        return None

    assert propagate(
        closed.jaxpr,
        closed.consts,
        [],
        [],
        closed.jaxpr.outvars,
        stall_recovery=recover,
    ) == [None]
    assert calls == [1]


def test_consumed_equations_do_not_run_after_recovery():
    closed = jax.make_jaxpr(lambda x: 2 * x + 1)(jnp.array(1.0))
    calls = []
    recovered = False

    def process(eqn, inputs, outputs):
        assert not recovered, "consumed section was reprocessed"
        calls.append(eqn.primitive.name)
        return None

    def recover(context, targets, changed):
        nonlocal recovered
        recovered = True
        return StallRecoveryResult(
            closed.jaxpr.invars,
            [jnp.array(1.0)],
            tuple(e.eqn_id for e in context.extended_jaxpr.equations),
            {closed.jaxpr.invars[0]: jnp.array(-0.7)},
        )

    values, state = propagate(
        closed.jaxpr,
        closed.consts,
        closed.jaxpr.outvars,
        [jnp.array(3.0)],
        closed.jaxpr.invars,
        process_eqn=process,
        stall_recovery=recover,
        return_state=True,
        initial_state={},
        process_all_eqns=True,
    )
    assert values == [jnp.array(1.0)]
    assert calls == ["add"]
    assert state[closed.jaxpr.invars[0]] == jnp.array(-0.7)


@pytest.mark.parametrize("bad_value", [None, Knowness.partial(jnp.array(1.0))])
def test_recovery_rejects_incomplete_values(bad_value):
    closed = jax.make_jaxpr(jnp.exp)(jnp.array(1.0))
    with pytest.raises(ValueError, match="complete, non-None"):
        propagate(
            closed.jaxpr,
            closed.consts,
            [],
            [],
            closed.jaxpr.outvars,
            stall_recovery=lambda *args: StallRecoveryResult(
                closed.jaxpr.invars, [bad_value]
            ),
        )


def test_recovery_cannot_overwrite_authoritative_values():
    closed = jax.make_jaxpr(jnp.exp)(jnp.array(1.0))
    with pytest.raises(ValueError, match="overwrite a complete"):
        propagate(
            closed.jaxpr,
            closed.consts,
            closed.jaxpr.invars,
            [jnp.array(1.0)],
            closed.jaxpr.outvars,
            process_eqn=lambda *args: None,
            stall_recovery=lambda *args: StallRecoveryResult(
                closed.jaxpr.invars, [jnp.array(2.0)]
            ),
        )


def test_unresolved_nested_output_does_not_become_complete():
    closed = jax.make_jaxpr(jax.jit(lambda x: x + jnp.tanh(x)))(jnp.array(1.0))
    from probjax.core.interpreters.inverse.interpreter import InverseProcessingRule
    from probjax.core.interpreters.inverse.registry import inverse_cost_fn

    values, env = propagate(
        closed.jaxpr,
        closed.consts,
        closed.jaxpr.outvars,
        [jnp.array(1.0)],
        closed.jaxpr.invars,
        process_eqn=InverseProcessingRule(),
        cost_fn=inverse_cost_fn,
        return_env=True,
    )
    assert values == [None]
    assert env.get_knowness_level(closed.jaxpr.invars[0]) == KnownessLevel.UNKNOWN
