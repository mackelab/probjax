
import jax
from jax import lax

from jax.core import Primitive, Jaxpr, JaxprEqn

from typing import Any, Callable, Optional
import jax.numpy as jnp





_UNIVARITAE_INVERSE_REGISTRY = {
    jax.lax.tanh_p: jax.lax.atanh_p,
    jax.lax.atanh_p: jax.lax.tanh_p,
    jax.lax.sinh_p: jax.lax.asinh_p,
    jax.lax.asinh_p: jax.lax.sinh_p,
    jax.lax.cosh_p: jax.lax.acosh_p,
    jax.lax.acosh_p: jax.lax.cosh_p,
    jax.lax.exp_p: jax.lax.log_p,
    jax.lax.log_p: jax.lax.exp_p,
    jax.lax.sqrt_p: lambda x: jax.lax.pow_p.bind(x, 2.0),
    jax.lax.rsqrt_p: lambda x: 1.0 / jax.lax.pow_p.bind(x, 2.0),
    jax.lax.neg_p: jax.lax.neg_p,
    jax.lax.log1p_p: jax.lax.expm1_p,
    jax.lax.expm1_p: jax.lax.log1p_p,
    jax.lax.erf_p: jax.lax.erf_inv_p,
    jax.lax.erf_inv_p: jax.lax.erf_p,
    jax.lax.conj_p: jax.lax.conj_p,
}

_BIVARIATE_INVERSE_REGISTRY = {
    jax.lax.mul_p: (jax.lax.div_p.bind, lambda x, y: jax.lax.div_p.bind(y, x)),
    jax.lax.div_p: (jax.lax.mul_p.bind, jax.lax.div_p.bind),
    jax.lax.add_p: (jax.lax.sub_p.bind, lambda x, y: jax.lax.sub_p.bind(y, x)),
    jax.lax.sub_p: (jax.lax.add_p.bind, jax.lax.sub_p.bind),
    jax.lax.pow_p: lambda x, y: jax.lax.pow_p.bind(x, 1.0/y),
    jax.lax.integer_pow_p: lambda x, y: jax.lax.pow_p.bind(x, 1.0/y)
}

def is_univariate(eqn) -> bool:
    return len(eqn.invars) == 1 and len(eqn.outvars) == 1

def is_bivariate(eqn) -> bool:
    return len(eqn.invars) == 2 and len(eqn.outvars) == 1

class InverseProcessingRules:

    def __call__(self, eqn, known_invars, known_outvars):
        pass 


    def _default_forward_processing(self, eqn, known_invars, known_outvars):
        return eqn
