import numpy as np
from functools import wraps

import jax
from jax import core
import jax.random as jrandom
from jax import lax
from jax._src.util import safe_map

from typing import Callable, Optional, Iterable

from probjax.core.inverse import InverseInterpreter
from probjax.core.joint_sample import JointSampleInterpreter
from probjax.core.interventions import InterventionInterpreter
from probjax.core.log_potential import LogPotentialInterpreter
from probjax.core.domains import DomainInterpreter 
from probjax.core.trace_all import TraceAllInterpreter

inverse_interpreter = InverseInterpreter()
tace_all = TraceAllInterpreter()

def trace_all(fun: Callable) -> Callable:

    @wraps(fun)
    def wrapped(*args, **kwargs):
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        out = tace_all.eval_jaxpr(
            closed_jaxpr.jaxpr, closed_jaxpr.literals, *args
        )
    
        return dict((str(key), val) for key, val in out.items())
    return wrapped

def inverse(fun: Callable) -> Callable:
    """If a invertible function is given as input, it returns the inverse of the function.


    Args:
        fun (Callable): Invertible function

    Returns:
        Callable: Inverse function
    """

    @wraps(fun)
    def wrapped(*args, **kwargs):
        # We may need to flatten and unflatten args...
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        out = inverse_interpreter.eval_jaxpr(
            closed_jaxpr.jaxpr, closed_jaxpr.literals, *args
        )

        if isinstance(out, list):
            out = out[0]
        else:
            out = tuple(out)

        return out

    return wrapped


def domains(fun: Callable) -> Callable:
    """Returns the domains of the random variables in the probabilistic function.

    Args:
        fun (Callable): Probabilistic function

    Returns:
        Callable: Function that returns the domains of the random variables in the probabilistic function.
    """
    interpreter = DomainInterpreter()

    @wraps(fun)
    def wrapped(*args, **kwargs):
        # We may need to flatten and unflatten args...
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        out = interpreter.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)

        return out

    return wrapped


def joint_sample(fun: Callable, rvs: Optional[Iterable] = None) -> Callable:
    """Samples all random variables called in the probabilstic function. If rvs is given, it only samples the random variables in rvs.

    Args:
        fun (Callable): Probabilistic function
        rvs (Optional[Iterable], optional): Subset of random variables in the probabilistic program. Defaults to None.

    Returns:
        Callable: Sampling function
    """
    interpreter = JointSampleInterpreter(rvs)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        # We may need to flatten and unflatten args...
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        outs = interpreter.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        out, joint_sample = outs
        if isinstance(out, list):
            out = out[0]
        else:
            out = tuple(out)

        return out, joint_sample

    return wrapped


def intervene(fun: Callable, interventions: dict) -> Callable:
    """Intervenes on the random variables in the probabilistic function i.e. instead of sampling from the random variable, it returns the intervention value specified in "interventions".

    Args:
        fun (Callable): Probabilistic function.
        interventions (dict): Interventions to apply. The keys are the names of the random variables and the values are the intervention values.

    Returns:
        Callable: Probabilistic function with interventions applied.
    """
    interpreter = InterventionInterpreter(interventions)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        # We may need to flatten and unflatten args...
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        out = interpreter.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)

        if isinstance(out, list):
            out = out[0]
        else:
            out = tuple(out)

        return out

    return wrapped


def log_potential(fun: Callable) -> Callable:
    """Computes the log potential ( unnormalized log_prob ) of the probabilistic function.

    Args:
        fun (Callable): Probabilistic function.

    Returns:
        Callable: Log potential function.
    """
    interpreter = LogPotentialInterpreter()
    closed_jaxpr = jax.make_jaxpr(fun)(jrandom.PRNGKey(0))
    @wraps(fun)
    def wrapped(**kwargs):
        # We may need to flatten and unflatten args...
        
        out, log_potential = interpreter.eval_jaxpr(
            closed_jaxpr.jaxpr, closed_jaxpr.literals, **kwargs
        )

        return out, log_potential

    return wrapped
