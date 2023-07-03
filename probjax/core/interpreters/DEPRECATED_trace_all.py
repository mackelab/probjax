from typing import Any, Callable, Iterable, Optional
from jax import lax
import jax.numpy as jnp
import jax.random as jrandom
from jax.core import Primitive, Jaxpr, JaxprEqn, ClosedJaxpr
from jax._src.util import safe_map

from probjax.core.jaxpr_propagation.interpret import (
    BaseInterpreter,
    BaseRules,
    jaxpr_returning_const,
    remove_closed_jaxpr_vars_with_suffix,
)
from probjax.core.custom_primitives.random_variable import rv_p
from jax import linear_util as lu
from jax import tree_util
from jax import api_util


class TraceAllInterpreter(BaseInterpreter):
    def _get_output(self, jaxpr: Jaxpr) -> Any:
        """Reads the output of the Jaxpr from the environment."""
        return self.env
