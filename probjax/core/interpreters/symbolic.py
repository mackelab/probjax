from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax.core import JaxprEqn
from jaxtyping import ArrayLike
from sympy import MatrixSymbol, Symbol
from sympy.tensor.array.expressions import ArraySymbol

from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule

_lookup = {
    jax.lax.mul_p: sp.Mul,
    jax.lax.add_p: sp.MatAdd,  # Note: jax.lax.add was originally mapped to both sp.Add and sp.MatAdd, keeping one
    jax.lax.div_p: sp.div,
    jax.lax.abs_p: sp.Abs,
    jax.lax.sign_p: sp.sign,
    jax.lax.ceil_p: sp.ceiling,
    jax.lax.floor_p: sp.floor,
    jax.lax.log_p: sp.log,
    jax.lax.exp_p: sp.exp,
    jax.lax.sqrt_p: sp.sqrt,
    jax.lax.cos_p: sp.cos,
    jax.lax.acos_p: sp.acos,
    jax.lax.sin_p: sp.sin,
    jax.lax.asin_p: sp.asin,
    jax.lax.tan_p: sp.tan,
    jax.lax.atan_p: sp.atan,
    jax.lax.atan2_p: sp.atan2,
    jax.lax.cosh_p: sp.cosh,
    jax.lax.acosh_p: sp.acosh,
    jax.lax.sinh_p: sp.sinh,
    jax.lax.asinh_p: sp.asinh,
    jax.lax.tanh_p: sp.tanh,
    jax.lax.atanh_p: sp.atanh,
    jax.lax.pow_p: sp.Pow,
    jax.lax.integer_pow_p: sp.Pow,
    jax.lax.real_p: sp.re,
    jax.lax.imag_p: sp.im,
    jax.lax.erf_p: sp.erf,
    jax.lax.eq_p: sp.Eq,
    jax.lax.ne_p: sp.Ne,
    jax.lax.gt_p: sp.StrictGreaterThan,
    jax.lax.lt_p: sp.StrictLessThan,
    jax.lax.le_p: sp.LessThan,
    jax.lax.ge_p: sp.GreaterThan,
    jax.lax.max_p: sp.Max,
    jax.lax.min_p: sp.Min,
    jax.lax.add_p: sp.MatAdd,  # Duplicate, noted for completeness
}

_constant_lookup = {
    jnp.e: sp.E,
    jnp.pi: sp.pi,
    jnp.euler_gamma: sp.EulerGamma,
    1j: sp.I,
}


def process_constant(val: Any) -> Any:
    # Check if val is a "special" constant
    for k, v in _constant_lookup.items():
        if np.isclose(val, k):
            return v[k]
    return val


def as_symbolic_var(var: ArrayLike, name="x") -> Symbol:
    var = jnp.asarray(var)
    shape = var.shape
    dtype = var.dtype
    if len(shape) == 0 or sum(shape) <= 1:
        if dtype == jnp.bool_:
            return Symbol(name, boolean=True)
        elif dtype == jnp.int_:
            return Symbol(name, integer=True)
        elif dtype == jnp.float_ or dtype == jnp.float32:
            return Symbol(name, real=True)
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
    elif len(shape) == 1:
        if dtype == jnp.float_ or dtype == jnp.float32:
            return MatrixSymbol(name, *shape + (1,))
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
    elif len(shape) == 2:
        if dtype == jnp.float_ or dtype == jnp.float32:
            return MatrixSymbol(name, *shape)
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
    else:
        return ArraySymbol(name, shape)


class SymbolicProcessingRule(ForwardProcessingRule):
    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> Tuple[Sequence[Any | None], Sequence[Any | None]]:
        sympy_eq = _lookup[eqn.primitive]
        print(known_inputs)
        print(eqn.params)
        params = list(map(process_constant, list(eqn.params.values())))
        print(params)
        sym_eq = sympy_eq(*known_inputs, *params)

        outvars = eqn.outvars
        if isinstance(sym_eq, Sequence):
            outvals = sym_eq
        else:
            outvals = [sym_eq]

        return outvars, outvals
