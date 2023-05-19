import jax
from functools import wraps

from jax import core
from jax.core import ClosedJaxpr, Jaxpr, Primitive, JaxprEqn
from typing import Iterable
from jax import lax
from jax._src.util import safe_map, curry

from typing import Any, Callable, Iterable, Union


class BaseRules(dict):
    """A dictionary of rules for primitives."""

    def _default_rule(self, prim: Primitive, *args, **kwargs) -> Any:
        """Default implementation of a rule for a primitive.

        Args:
            prim (Primitive): Primitive

        Returns:
            Any: Rule to apply
        """
        return prim.bind(*args, **kwargs)

    def __getitem__(self, key: Primitive) -> Any:
        """Returns a custom rule for a primitive.
            If no rule is found, returns the default rule.

        Args:
            key (str): Name of primitive

        Returns:
            Any: Rule to apply
        """
        return self.__dict__.get(key, self._default_rule)  # type: ignore

    def __setitem__(self, key: Primitive, value: Any) -> None:
        """Sets a custom rule for a primitive.

        Args:
            key (str): Name of primitive
            value (Any): Rule to apply
        """
        self.__dict__[key] = value  # type: ignore


class BaseInterpreter:
    rules: BaseRules = BaseRules()
    env: dict = {}

    def write(self, var: str, val: Any):
        """Writes a value to the environment."""
        self.env[var] = val

    def read(self, var: Union[str, core.Literal]):
        """Reads a value from the environment."""
        if type(var) is core.Literal:
            return var.val
        else:
            return self.env[var]

    def _init_environment(
        self, jaxpr: Jaxpr, consts: Iterable, *args
    ) -> Iterable[JaxprEqn]:
        """Initializes the environment for the Jaxpr.

        Args:
            write (Callable): Function that writes to the environment
            jaxpr (Jaxpr): Jaxpr
            consts (Iterable): Constants required in the Jaxpr

        Returns:
            Iterable[Equation]: Iterable of equations to evaluate.
        """
        self.env = {}
        # Bind args and consts to environment
        safe_map(self.write, jaxpr.invars, args)
        safe_map(self.write, jaxpr.constvars, consts)

        # Equations in some order
        eqns = jaxpr.eqns

        return eqns

    def _get_output(self, jaxpr: Jaxpr) -> Any:
        """Reads the output of the Jaxpr from the environment."""
        out = safe_map(self.read, jaxpr.outvars)
        return out

    def _get_invals(self, eqn: JaxprEqn) -> Iterable:
        """Returns the input variables of an equation."""
        return safe_map(self.read, eqn.invars)

    def _write_outvals(self, eqn: JaxprEqn, outvals) -> Iterable:
        """Returns the output variables of an equation."""
        return safe_map(self.write, eqn.outvars, outvals)

    def eval_jaxpr(self, jaxpr: Jaxpr, consts: Iterable, *args, **kwargs) -> Any:
        # Mapping from variable -> value
        eqns = self._init_environment(jaxpr, consts, *args, **kwargs)

        # Loop through equations and evaluate primitives using `bind`
        for eqn in eqns:
            # Read inputs to equation from environmen
            invals = self._get_invals(eqn)
            prim = eqn.primitive
            subfuns, bind_params = prim.get_bind_params(eqn.params)
            rule = self.rules[prim]
            # `bind` is how a primitive is called
            outvals = rule(prim, *subfuns, *invals, **bind_params)
            # Primitives may return multiple outputs or not
            if not eqn.primitive.multiple_results:
                outvals = [outvals]

            # Write the results of the primitive into the environment
            self._write_outvals(eqn, outvals)
        # Read the final result of the Jaxpr from the environment

        return self._get_output(jaxpr)


def make_jaxpr(fun):
    """Returns the Jaxpr of a function."""

    @wraps(fun)
    def wrapped(*args, **kwargs):
        return fun(*args, **kwargs)

    return jax.make_jaxpr(wrapped)
