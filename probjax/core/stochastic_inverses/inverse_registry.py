

from jax import lax

from jax.core import Primitive, Jaxpr, JaxprEqn

from typing import Any, Callable, Optional
import jax.numpy as jnp



class InverseRegistry:

    def __init__(self) -> None:
        self._registry = {}
        super().__init__()

    def register(self, prim: Primitive, factory: Optional[Callable] = None):
        
        # Decorator support
        if factory is None:
            return lambda factory: self.register(prim, factory)

        self._registry[prim] = factory

    
    def __call__(self, prim: Primitive) -> Any:
        try:
            return self._registry[prim]
        except KeyError:
            raise NotImplementedError(f"Inverse for {prim} not implemented")



invert = InverseRegistry()

def _check_univariate(in_known, out_known):
    assert len(in_known) == 1, "More than one input"
    assert len(out_known) == 1, "Mire than one output"

def _check_bivariate_reduction(in_known, out_known):
    assert len(in_known) == 2, "More than two inputs"
    assert len(out_known) == 1, "Mire than one output"


@invert.register(lax.exp_p)
def invert_exp(prim: Primitive, in_known, out_known):
    _check_univariate(in_known, out_known)
    if all(in_known):
        return jnp.exp
    elif all(out_known):
        return jnp.log

invert.register(lax.log_p, lambda prim, in_known, out_known: invert_exp(prim, out_known, in_known))



@invert.register(lax.add_p)
def invert_add(prim: Primitive, in_known, out_known):
    _check_bivariate_reduction(in_known, out_known)
    if all(in_known):
        return lambda a, b: x+y
    elif all(out_known):
        return lambda x, y: x - y