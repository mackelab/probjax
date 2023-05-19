from typing import Any, Callable, Iterable, Optional
from jax import lax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.core import Primitive, Jaxpr, JaxprEqn,eval_jaxpr
from jax._src.util import safe_map

from probjax.core.utils import BaseInterpreter, BaseRules
from probjax.core.random_variable import rv_p, CallPrimitive
from probjax.core.log_prob import log_prob
from jax import linear_util as lu

from probjax.core.random_variable import unzip2


class LogPotentialRules(BaseRules):
    def __init__(self) -> None:
        self.__dict__[rv_p] = self._rv_rule  # type: ignore
        self._vals = {}

    def _rv_rule(self, prim: Primitive, *args, **kwargs) -> Any:
        out = super()._default_rule(prim, *args, **kwargs)
        name = kwargs["name"]
        kwargs["mode"] = "log_prob"
        consts, _ = unzip2(*args, **kwargs)  
    
        log_prob_fn = kwargs["log_prob_fn_jaxpr"]
        log_prob = eval_jaxpr(log_prob_fn.jaxpr, log_prob_fn.literals , *consts, *out)

        return out + log_prob

class LogPotentialInterpreter(BaseInterpreter):
    rules = LogPotentialRules()

    def _init_environment(
        self,
        jaxpr: Jaxpr,
        consts: Iterable,
        *args,
        **kwargs,
    ) -> Iterable[JaxprEqn]:
        """Initializes the environment for the Jaxpr."""
        out = super()._init_environment(jaxpr, consts, *args)
        safe_map(self.write, ["__log_potential__"], [jnp.zeros(1)])
        #print(jaxpr.pretty_print())
        return out

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
