import jax
import jax.numpy as jnp

from probjax.core import intervene, inverse, joint_sample, log_potential_fn, trace
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.core.jaxpr_propagation import (
    ExtendedJaxpr,
    InterpreterPipeline,
    InterpreterSpec,
)
from probjax.core.jaxpr_propagation.interpret import interpret
from probjax.core.jaxpr_propagation.propagate import propagate
from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule
from probjax.stats import norm


def test_interpret_and_propagate_match_nested_jit():
    def f(x):
        def inner(z):
            return jnp.sin(z) + 2.0

        return jax.jit(inner)(x) * 3.0

    x = jnp.array(0.7)
    closed_jaxpr = jax.make_jaxpr(f)(x)

    interpreted = interpret(
        closed_jaxpr.jaxpr,
        closed_jaxpr.consts,
        closed_jaxpr.jaxpr.invars,
        (x,),
        closed_jaxpr.jaxpr.outvars,
    )
    propagated = propagate(
        closed_jaxpr.jaxpr,
        closed_jaxpr.consts,
        closed_jaxpr.jaxpr.invars,
        (x,),
        closed_jaxpr.jaxpr.outvars,
    )

    expected = f(x)
    assert jnp.allclose(interpreted[0], expected)
    assert jnp.allclose(propagated[0], expected)


def test_inverse_through_nested_jit():
    def f(x):
        return jax.jit(lambda z: jnp.exp(z) + 1.0)(x)

    x0 = jnp.array(0.3)
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))

    assert jnp.allclose(x_rec, x0, atol=1e-6, rtol=1e-6)


def test_trace_wrappers_do_not_share_state():
    traced_add = trace(lambda x: x + 1.0)
    traced_mul = trace(lambda x: x * 2.0)

    traced_add_out = traced_add(jnp.array(3.0))
    snapshot = {key: value for key, value in traced_add_out.items()}

    _ = traced_mul(jnp.array(3.0))

    assert list(traced_add_out.keys()) == list(snapshot.keys())
    for key in snapshot:
        assert jnp.allclose(traced_add_out[key], snapshot[key])


def test_intervene_overrides_random_variable_value():
    def model(key):
        return rv_p.bind(key, 0.0, 1.0, dist=norm, name="x")

    intervened_model = intervene(model, {"x": jnp.array(2.5)})

    out1 = intervened_model(jax.random.PRNGKey(0))
    out2 = intervened_model(jax.random.PRNGKey(1))

    assert jnp.allclose(out1, 2.5)
    assert jnp.allclose(out2, 2.5)


def test_interpret_reducer_and_equation_states():
    def f(x):
        return jnp.exp(x) + 2.0

    closed_jaxpr = jax.make_jaxpr(f)(jnp.array(0.7))

    def process_eqn_with_state(eqn, known_inputs, known_outputs):
        result = ForwardProcessingRule()(eqn, known_inputs, known_outputs)
        outvars, outvals = result[0], result[1]
        return outvars, outvals, 1

    def reducer(_, __, state, eqn_state):
        return state + (eqn_state or 0)

    outputs, state, env = interpret(
        closed_jaxpr.jaxpr,
        closed_jaxpr.consts,
        closed_jaxpr.jaxpr.invars,
        (jnp.array(0.7),),
        closed_jaxpr.jaxpr.outvars,
        process_eqn=process_eqn_with_state,
        reducer=reducer,
        initial_state=0,
        return_state=True,
        return_env=True,
    )

    assert jnp.allclose(outputs[0], f(jnp.array(0.7)))
    assert state == len(closed_jaxpr.jaxpr.eqns)
    assert env.read_state(0) == 1


def test_joint_sample_and_log_potential_use_equation_state():
    def model(key):
        k1, k2 = jax.random.split(key)
        x = rv_p.bind(k1, 0.0, 1.0, dist=norm, name="x")
        y = rv_p.bind(k2, 1.0, 2.0, dist=norm, name="y")
        return x + y

    samples = joint_sample(model)(jax.random.PRNGKey(0))
    assert set(samples.keys()) == {"x", "y"}

    log_potential = log_potential_fn(model)
    got = log_potential(**samples)
    expected = norm.logpdf(samples["x"], 0.0, 1.0) + norm.logpdf(samples["y"], 1.0, 2.0)
    assert jnp.allclose(got, expected)


def test_extended_jaxpr_contains_stable_equation_ids():
    def f(x):
        return jnp.sin(x) + 1.0

    closed_jaxpr = jax.make_jaxpr(f)(jnp.array(0.7))
    extended = ExtendedJaxpr.from_jaxpr(closed_jaxpr.jaxpr)

    assert len(extended.equations) == len(closed_jaxpr.jaxpr.eqns)
    assert [eqn.eqn_id for eqn in extended.equations] == [
        (index,) for index in range(len(closed_jaxpr.jaxpr.eqns))
    ]


def test_interpreter_pipeline_supports_dependency_context():
    def f(x):
        return jnp.exp(x) + 2.0

    closed_jaxpr = jax.make_jaxpr(f)(jnp.array(0.7))

    def primary_rule(eqn, known_inputs, known_outputs, context):
        result = ForwardProcessingRule()(eqn, known_inputs, known_outputs)
        outvars, outvals = result[0], result[1]
        return outvars, outvals, {"primitive": eqn.primitive.name}

    def observer_rule(_, __, ___, context):
        primary_state = context.read_transient_state("primary")
        return [], [], {"depends_on": primary_state["primitive"]}

    pipeline = InterpreterPipeline([
        InterpreterSpec(name="primary", rule=primary_rule),
        InterpreterSpec(
            name="observer",
            rule=observer_rule,
            observe_only=True,
            requires=("primary",),
        ),
    ])

    def reducer(_, __, state, eqn_state):
        state.append(eqn_state)
        return state

    outputs, state, env = interpret(
        closed_jaxpr.jaxpr,
        closed_jaxpr.consts,
        closed_jaxpr.jaxpr.invars,
        (jnp.array(0.7),),
        closed_jaxpr.jaxpr.outvars,
        process_eqn=pipeline,
        reducer=reducer,
        initial_state=[],
        return_state=True,
        return_env=True,
        state_namespace="pipeline",
    )

    assert jnp.allclose(outputs[0], f(jnp.array(0.7)))
    assert len(state) == len(closed_jaxpr.jaxpr.eqns)
    for eqn_state in state:
        assert eqn_state["observer"]["depends_on"] == eqn_state["primary"]["primitive"]
    assert env.read_state(0, namespace="pipeline") is not None
