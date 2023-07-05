from functools import wraps
from typing import Callable, Iterable, Optional

import jax

from probjax.core.jaxpr_propagation.interpret import interpret
from probjax.core.jaxpr_propagation.propagate import propagate

from probjax.core.interpreters.joint_sample import JointSampleProcessingRule
from probjax.core.interpreters.log_potential import (
    LogPotentialProcessingRule,
    potential_cost_fn,
    extract_random_vars_values,
)
from probjax.core.interpreters.inverse import (
    InverseProcessingRule,
    inverse_cost_fn,
    InverseAndLogAbsDetProcessingRule,
)
from probjax.core.interpreters.trace import TraceProcessingRule


def joint_sample(fun: Callable, rvs: Optional[Iterable] = None) -> Callable:
    """Samples all random variables called in the probabilstic function. If rvs is given, it only samples the random variables in rvs.

    Args:
        fun (Callable): Probabilistic function
        rvs (Optional[Iterable], optional): Subset of random variables in the probabilistic program. Defaults to None.

    Returns:
        Callable: Sampling function
    """
    jaxpr_maker = jax.make_jaxpr(fun)
    processing_rule = JointSampleProcessingRule(rvs=rvs)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        jaxpr = jaxpr_maker(*args, **kwargs)
        _ = interpret(
            jaxpr.jaxpr,
            jaxpr.consts,
            jaxpr.jaxpr.invars,
            args,
            jaxpr.jaxpr.outvars,
            process_eqn=processing_rule,
        )

        return processing_rule.joint_samples

    return wrapped


def log_potential_fn(fun: Callable, *args, **kwargs):
    """Computes the log potential of the probabilistic function.
    This does not about normalizing constant.

    Args:
        fun (Callable): Probabilistic function

    Returns:
        Callable: Log potential function
    """
    jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)

    def log_potential(joint_samples):
        processing_rule = LogPotentialProcessingRule(joint_samples=joint_samples)
        rv_vars, rv_values = extract_random_vars_values(jaxpr.jaxpr, joint_samples)
        _ = propagate(
            jaxpr.jaxpr,
            jaxpr.consts,
            [],
            [],
            rv_vars,
            process_eqn=processing_rule,
            cost_fn=potential_cost_fn,
        )
        return processing_rule.log_prob

    return log_potential

def trace(fun: Callable, traced_vars=None):
    jaxpr_maker = jax.make_jaxpr(fun)
    processing_rule = TraceProcessingRule(traced_vars=traced_vars)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        jaxpr = jaxpr_maker(*args, **kwargs)
        _ = interpret(
            jaxpr.jaxpr,
            jaxpr.consts,
            jaxpr.jaxpr.invars,
            args,
            jaxpr.jaxpr.outvars,
            process_eqn=processing_rule,
        )

        return processing_rule.traced_samples

    return wrapped


def inverse(fun: Callable, invertible_arg=None):
    jaxpr_maker = jax.make_jaxpr(fun)
    processing_rule = InverseProcessingRule()

    @wraps(fun)
    def wrapped(*args, **kwargs):
        jaxpr = jaxpr_maker(*args, **kwargs)

        if invertible_arg is not None:
            flatten_args, _ = jax.tree_util.tree_flatten(args)
            if invertible_arg < 0:
                adjusted_invertible_arg = len(flatten_args) + invertible_arg
            else:
                adjusted_invertible_arg = invertible_arg
            out_arg = [flatten_args[adjusted_invertible_arg]]
            flat_args = (
                flatten_args[:adjusted_invertible_arg] + flatten_args[adjusted_invertible_arg + 1:] + out_arg
            )
            const_invars = (
                jaxpr.jaxpr.invars[:adjusted_invertible_arg]
                + jaxpr.jaxpr.invars[adjusted_invertible_arg+1:]
            )
            out_invar = [jaxpr.jaxpr.invars[adjusted_invertible_arg]]
            print(const_invars)

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


def inverse_and_logabsdet(fun: Callable):
    jaxpr_maker = jax.make_jaxpr(fun)
    processing_rule = InverseAndLogAbsDetProcessingRule()

    @wraps(fun)
    def wrapped(*args, **kwargs):
        jaxpr = jaxpr_maker(*args, **kwargs)
        out = propagate(
            jaxpr.jaxpr,
            jaxpr.consts,
            jaxpr.jaxpr.outvars,
            args,
            jaxpr.jaxpr.invars,
            process_eqn=processing_rule,
            cost_fn=inverse_cost_fn,
            process_all_eqns=True,
        )
        log_det = sum([processing_rule.log_dets[v] for v in jaxpr.jaxpr.invars])
        return out[0], log_det

    return wrapped
