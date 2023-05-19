from typing import Any, Callable, Iterable, Optional
from jax import lax
import jax.numpy as jnp
from jax.core import Primitive, Jaxpr, JaxprEqn
from jax._src.util import safe_map

from probjax.core.utils import BaseInterpreter, BaseRules


# Inverse rules base class


class InverseRules(BaseRules):
    def _default_rule(self, prim: Primitive, *args, **kwargs) -> Any:
        raise NotImplementedError(f"Inverse for {prim} not implemented")


rules = InverseRules()


def register_elementwise_inverse_rule(prim: Primitive):
    """Registers a rule for a primitive.

    Args:
        prim (Primitive): Primitive
        rule (Callable): Rule to apply
    """

    def make_rule(fun):
        rules[prim] = fun

    return make_rule


# Inverse interpreter


class InverseInterpreter(BaseInterpreter):
    rules = rules

    def _init_environment(self, jaxpr: Jaxpr, consts: Iterable, *args) -> Iterable:
        # Satrt with outvars instead of invars
        safe_map(self.write, jaxpr.outvars, args)
        safe_map(self.write, jaxpr.constvars, consts)

        # Reverse order of equations!
        return jaxpr.eqns[::-1]

    def _get_output(self, jaxpr: Jaxpr) -> Any:
        """Reads the output of the Jaxpr from the environment."""
        out = safe_map(self.read, jaxpr.invars)
        return out

    def _get_invals(self, eqn: JaxprEqn) -> Iterable:
        """Returns the input variables of an equation."""
        return safe_map(self.read, eqn.outvars)

    def _write_outvals(self, eqn: JaxprEqn, outvals) -> Iterable:
        """Returns the output variables of an equation."""
        return safe_map(self.write, eqn.invars, outvals)


# Inverse rules


HALF_PI = jnp.pi / 2

register_elementwise_inverse_rule(lax.exp_p)(
    lambda _, *args, **kwargs: lax.log(*args, **kwargs)
)
register_elementwise_inverse_rule(lax.log_p)(
    lambda _, *args, **kwargs: lax.exp(*args, **kwargs)
)
register_elementwise_inverse_rule(lax.tanh_p)(
    lambda _, *args, **kwargs: lax.atanh(*args, **kwargs)
)
register_elementwise_inverse_rule(lax.atanh_p)(
    lambda _, *args, **kwargs: lax.tanh(*args, **kwargs)
)
register_elementwise_inverse_rule(lax.logistic_p)(
    lambda _, x, **kwargs: lax.log(x, **kwargs) - lax.log1p(-x, **kwargs)
)


def _check_asin_acos_domain(x):
    if jnp.any(jnp.abs(x) > 1):
        raise ValueError("Inverse sine/cosine only defined on [-1, 1]")


def _check_sin_domain(x):
    if jnp.any(x > 0) & jnp.any(x <= jnp.pi):
        raise ValueError("Inverse sine only defined on [0, pi]")


def _check_cos_domain(x):
    if jnp.any(jnp.abs(x) > HALF_PI):
        raise ValueError("Inverse cosine only defined on [-pi/2, pi/2]")


@register_elementwise_inverse_rule(lax.sin_p)
def invsin(_, x):
    _check_asin_acos_domain(x)
    return lax.asin(x)


@register_elementwise_inverse_rule(lax.cos_p)
def invcos(_, x):
    _check_asin_acos_domain(x)
    return lax.acos(x)


@register_elementwise_inverse_rule(lax.asin_p)
def sin(_, x):
    _check_sin_domain(x)
    return lax.sin(x)


@register_elementwise_inverse_rule(lax.acos_p)
def cos(_, x):
    _check_cos_domain(x)
    return lax.cos(x)
