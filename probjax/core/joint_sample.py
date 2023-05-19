from typing import Any, Callable, Iterable, Optional
from jax import lax
import jax.numpy as jnp
from jax.core import Primitive, Jaxpr, JaxprEqn
from jax._src.util import safe_map

from probjax.core.utils import BaseInterpreter, BaseRules
from probjax.core.random_variable import rv_p


class JointSampleInterpreter(BaseInterpreter):
    def __init__(self, rvs: Optional[Iterable] = None) -> None:
        self.rvs = rvs

    def _init_environment(
        self, jaxpr: Jaxpr, consts: Iterable, *args
    ) -> Iterable[JaxprEqn]:
        """Initializes the environment for the Jaxpr."""
        out = super()._init_environment(jaxpr, consts, *args)
        safe_map(self.write, ["__joint_sample__"], [{}])

        return out

    def _write_outvals(self, eqn: JaxprEqn, outvals) -> Iterable:

        if eqn.primitive is rv_p:
            name = eqn.params["name"]
            if self.rvs is None or name in self.rvs:
                self.env["__joint_sample__"][name] = outvals[0]

        return super()._write_outvals(eqn, outvals)

    def _get_output(self, jaxpr: Jaxpr) -> Any:
        out = super()._get_output(jaxpr)
        joint_samples = self.env["__joint_sample__"]
        return out, joint_samples
