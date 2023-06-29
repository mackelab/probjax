from typing import Any, Callable, Iterable, Optional
from jax import lax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.core import Primitive, Jaxpr, JaxprEqn, eval_jaxpr
from jax._src.util import safe_map

from probjax.core.utils import BaseInterpreter, BaseRules
from probjax.core.custom_primitives.random_variable import rv_p, CallPrimitive
from jax import linear_util as lu


def extract_random_variables(jaxpr: Jaxpr) -> Iterable[JaxprEqn]:
    """Extracts random variables from a Jaxpr.

    Args:
        jaxpr (Jaxpr): Jaxpr

    Returns:
        Iterable[JaxprEqn]: Random variables
    """
    return filter(lambda eqn: eqn.primitive is rv_p, jaxpr.eqns)


class LogPotentialRules(BaseRules):
    def __init__(self, strict=True) -> None:
        self.strict = strict
        self.__dict__[rv_p] = self._rv_rule  # type: ignore

    def _rv_rule(self, prim: Primitive, *args, **kwargs) -> Any:
        name = kwargs["name"]
        log_prob_fn = kwargs["log_prob_fn_jaxpr"]
        log_prob = eval_jaxpr(log_prob_fn.jaxpr, log_prob_fn.literals, *args[:-1], out)

        return out + log_prob


class LogPotentialInterpreter(BaseInterpreter):
    rules = LogPotentialRules()

    def __init__(self, strict=True) -> None:
        super().__init__()
        self.strict = strict

    def _init_environment(
        self,
        jaxpr: Jaxpr,
        consts: Iterable,
        *args,
        **values,
    ) -> Iterable[JaxprEqn]:
        """Initializes the environment for the Jaxpr."""
        self.env = {}
        # Bind args and consts to environment
        safe_map(self.write, jaxpr.constvars, consts)
        return jaxpr.eqns

    def _write_outvals(self, eqn: JaxprEqn, outvals):
        if eqn.primitive is rv_p:
            name = eqn.params["name"]
            self.env["__log_potential__"] += outvals[1]
            super()._write_outvals(eqn, outvals[:1])
        else:
            super()._write_outvals(eqn, outvals)

    def _get_output(self, jaxpr: Jaxpr) -> Any:
        out = super()._get_output(jaxpr)
        log_prob = self.env["__log_potential__"]
        return out, log_prob
