from functools import wraps
from typing import Callable, Iterable, Optional, cast

import jax
from jax import numpy as jnp
from jaxtyping import Array

from probjax.core.interpreters import (
    INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
    IntervenedProcessingRule,
    InverseAndLogAbsDetProcessingRule,
    InverseProcessingRule,
    JointSampleProcessingRule,
    LogPotentialProcessingRule,
    TraceProcessingRule,
    inverse_and_logabsdet_state_reducer,
    inverse_cost_fn,
    joint_sample_state_reducer,
    log_potential_state_reducer,
    trace_state_reducer,
)
from probjax.core.jaxpr_propagation.interpret import interpret
from probjax.core.jaxpr_propagation.propagate import propagate


def _leaf_signature(leaf):
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return ("array", tuple(leaf.shape), str(leaf.dtype))

    try:
        hash(leaf)
        return ("py", leaf)
    except TypeError:
        return ("obj", type(leaf).__name__, repr(leaf))


def _trace_signature(args, kwargs):
    flat, tree = jax.tree_util.tree_flatten((args, kwargs))
    return tree, tuple(_leaf_signature(leaf) for leaf in flat)


def _cached_jaxpr_getter(fun: Callable, static_argnums=()):
    jaxpr_maker = jax.make_jaxpr(fun, static_argnums=static_argnums)
    cache = {}

    def get_jaxpr(*args, **kwargs):
        key = _trace_signature(args, kwargs)
        if key not in cache:
            cache[key] = jaxpr_maker(*args, **kwargs)
        return cache[key]

    return get_jaxpr


def joint_sample(fun: Callable, rvs: Optional[Iterable] = None) -> Callable:
    """Samples all random variables called in the probabilistic function. If rvs is
    given, it only samples the random variables in rvs.

    Args:
        fun (Callable): Probabilistic function
        rvs (Optional[Iterable], optional): Subset of random variables in the
            probabilistic program. Defaults to None.

    Returns:
        Callable: Sampling function
    """
    get_jaxpr = _cached_jaxpr_getter(fun)
    interventions = getattr(fun, "_probjax_interventions", None)

    def wrapped(*args, **kwargs):
        processing_rule = JointSampleProcessingRule(
            rvs=rvs, interventions=interventions
        )
        jaxpr = get_jaxpr(*args, **kwargs)
        joint_result = cast(
            tuple[list, dict],
            interpret(
                jaxpr.jaxpr,
                jaxpr.consts,
                jaxpr.jaxpr.invars,
                args,
                jaxpr.jaxpr.outvars,
                process_eqn=processing_rule,
                reducer=joint_sample_state_reducer,
                initial_state={},
                return_state=True,
            ),
        )
        joint_samples = joint_result[1]

        return joint_samples

    return wrapped


def intervene(fun: Callable, rvs: dict[str, Array], *args, **kwargs):
    """Fix the value of random variables in the probabilistic function.
    This does not sample the random variables, but fixes them to the given values.

    The wrapped function uses interpreter-level overrides for the selected
    random variables while leaving all other equations unchanged.

    Args:
        fun (Callable): A function to transform.
        rvs (dict[str, Array]): A dictionary of random variable names and values to
            intervene.

    Returns:
        _type_: _description_
    """

    jaxpr = jax.make_jaxpr(fun)(jax.random.PRNGKey(0), *args, **kwargs)
    tree_out = jax.tree_util.tree_structure(fun(jax.random.PRNGKey(0), *args, **kwargs))

    @wraps(fun)
    def wrapped(*args, **kwargs):
        processing_rule = IntervenedProcessingRule(interventions=rvs)
        out = interpret(
            jaxpr.jaxpr,
            jaxpr.consts,
            jaxpr.jaxpr.invars,
            args,
            jaxpr.jaxpr.outvars,
            process_eqn=processing_rule,
        )

        return jax.tree_util.tree_unflatten(tree_out, out)

    setattr(wrapped, "_probjax_interventions", frozenset(rvs.keys()))
    return wrapped


def log_potential_fn(fun: Callable, *args, **kwargs):
    """Computes the log potential of the probabilistic function.
    This does not about normalizing constant.

    Args:
        fun (Callable): Probabilistic function

    Returns:
        Callable: Log potential function
    """
    jaxpr = jax.make_jaxpr(fun)(jax.random.PRNGKey(0), *args, **kwargs)
    interventions = getattr(fun, "_probjax_interventions", None)

    def log_potential(**joint_samples):
        processing_rule = LogPotentialProcessingRule(
            joint_samples=joint_samples,
            interventions=interventions,
        )

        log_potential_result = cast(
            tuple[list, jax.Array],
            interpret(
                jaxpr.jaxpr,
                jaxpr.consts,
                jaxpr.jaxpr.invars,
                (jax.random.PRNGKey(0),) + args,
                jaxpr.jaxpr.outvars,
                process_eqn=processing_rule,
                reducer=log_potential_state_reducer,
                initial_state=jnp.asarray(0.0),
                return_state=True,
            ),
        )
        log_prob = log_potential_result[1]

        return jnp.nan_to_num(log_prob, nan=-jnp.inf, posinf=jnp.inf, neginf=-jnp.inf)

    return log_potential


def trace(fun: Callable, traced_vars=None):
    get_jaxpr = _cached_jaxpr_getter(fun)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        processing_rule = TraceProcessingRule(traced_vars=traced_vars)
        jaxpr = get_jaxpr(*args, **kwargs)
        trace_result = cast(
            tuple[list, dict],
            interpret(
                jaxpr.jaxpr,
                jaxpr.consts,
                jaxpr.jaxpr.invars,
                args,
                jaxpr.jaxpr.outvars,
                process_eqn=processing_rule,
                reducer=trace_state_reducer,
                initial_state={},
                return_state=True,
            ),
        )
        traced_samples = trace_result[1]

        return traced_samples

    return wrapped


def inverse(fun: Callable, static_argnums=(), invertible_arg=None):
    get_jaxpr = _cached_jaxpr_getter(fun, static_argnums=static_argnums)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        processing_rule = InverseProcessingRule()
        jaxpr = get_jaxpr(*args, **kwargs)

        if invertible_arg is not None:
            flatten_args, _ = jax.tree_util.tree_flatten(args)
            if invertible_arg < 0:
                adjusted_invertible_arg = len(flatten_args) + invertible_arg
            else:
                adjusted_invertible_arg = invertible_arg
            out_arg = [flatten_args[adjusted_invertible_arg]]
            flat_args = (
                flatten_args[:adjusted_invertible_arg]
                + flatten_args[adjusted_invertible_arg + 1 :]
                + out_arg
            )
            const_invars = (
                jaxpr.jaxpr.invars[:adjusted_invertible_arg]
                + jaxpr.jaxpr.invars[adjusted_invertible_arg + 1 :]
            )
            out_invar = [jaxpr.jaxpr.invars[adjusted_invertible_arg]]

        else:
            const_invars = []
            out_invar = jaxpr.jaxpr.invars
            flat_args = args
        out = propagate(
            jaxpr.jaxpr,
            jaxpr.consts,
            const_invars + jaxpr.jaxpr.outvars,
            flat_args,
            out_invar,
            process_eqn=processing_rule,
            cost_fn=inverse_cost_fn,
            process_all_eqns=True,
        )

        return out[0]

    return wrapped


def inverse_and_logabsdet(fun: Callable, static_argnums=(), invertible_arg=None):
    get_jaxpr = _cached_jaxpr_getter(fun, static_argnums=static_argnums)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        processing_rule = InverseAndLogAbsDetProcessingRule(
            state_namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE
        )
        jaxpr = get_jaxpr(*args, **kwargs)

        if invertible_arg is not None:
            flatten_args, _ = jax.tree_util.tree_flatten(args)
            if invertible_arg < 0:
                adjusted_invertible_arg = len(flatten_args) + invertible_arg
            else:
                adjusted_invertible_arg = invertible_arg
            out_arg = [flatten_args[adjusted_invertible_arg]]
            flat_args = (
                flatten_args[:adjusted_invertible_arg]
                + flatten_args[adjusted_invertible_arg + 1 :]
                + out_arg
            )
            const_invars = (
                jaxpr.jaxpr.invars[:adjusted_invertible_arg]
                + jaxpr.jaxpr.invars[adjusted_invertible_arg + 1 :]
            )
            out_invar = [jaxpr.jaxpr.invars[adjusted_invertible_arg]]
            invars = const_invars + jaxpr.jaxpr.outvars
            outvars = out_invar
            args_for_propagate = flat_args
        else:
            invars = jaxpr.jaxpr.outvars
            outvars = jaxpr.jaxpr.invars
            args_for_propagate = args

        inverse_result = cast(
            tuple[list, dict],
            propagate(
                jaxpr.jaxpr,
                jaxpr.consts,
                invars,
                args_for_propagate,
                outvars,
                process_eqn=processing_rule,
                cost_fn=inverse_cost_fn,
                process_all_eqns=True,
                reducer=inverse_and_logabsdet_state_reducer,
                initial_state={},
                return_state=True,
                state_namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
            ),
        )
        out, log_dets = inverse_result

        log_det = jnp.asarray(0.0)
        for v in outvars:
            log_det = log_det + jnp.asarray(log_dets.get(v, 0.0))
        return out[0], log_det

    return wrapped
