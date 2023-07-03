import jax

from probjax.distributions.constraints import (
    real,
    integer,
    boolean,
    interval,
    positive,
    negative,
    unit_interval,
    unit_square,
)

import jax.numpy as jnp
import jax.random as jrandom

from jax.lax import integer_pow_p, pow_p, dot_general_p
from jax.core import Primitive, JaxprEqn


def inverse_pow_factory(eqn: JaxprEqn, in_constraint, out_constraint):
    assert (
        eqn.primitive == integer_pow_p or eqn.primitive == pow_p
    ), "This factory only support pow like primitives"
    y = eqn.params["y"]
    if in_constraint in positive and y != 0:
        params = {}

        def inverse_fn(x, **kwargs):
            return x ** (1 / y)

        return inverse_fn, params
    elif y == 0:
        raise NotImplementedError("Inverse of 0 is not implemented")
    else:
        params = {"p": 0.5}

        def inverse_fn(key, x, p, **kwargs):
            sign = jrandom.choice(key, jnp.array([-1, 1]), p=jnp.array([p, 1 - p]))
            return sign * jnp.abs(x) ** (1 / y)


def inverse_dot_general(eqn: JaxprEqn, in_constraint, out_constraint):
    assert (
        eqn.primitive == dot_general_p
    ), "This factory only support dot_general primitive"
    dimension_numbers = eqn.params["dimension_numbers"]
    if dimension_numbers[0] == ((), ()) and dimension_numbers[1] == ((), ()):
        raise NotImplementedError(
            "Inverse of dot_general with no reduction is not implemented"
        )
    elif dimension_numbers[0] == ((1,), (0,)) and dimension_numbers[1] == ((), ()):
        # Matrix vector multiplication
        pass


def inverse_matrix_vector_multiplication(A):
    eps = jnp.finfo(A.dtype).eps
    m, n = A.shape
    d = min(m, n)
    # SVD
    U, s, V_T = jnp.linalg.svd(A)
    # Filter out very small singular values
    s_inv = jnp.where(s > eps, 1 / s, 0)
    is_zero = jnp.concatenate([s < eps, jnp.ones(n - d, dtype=bool)])

    # Orhtogonal basis of the null space
    basis = V_T[is_zero, :]
    # Pseudo inverse
    pseudo_inverse = V_T.T[:, :d] @ (jnp.diag(s_inv) @ U.T[:d, :])

    # Produces an independent sample from the solution space
    def sample_solution(key, b):
        x_star = pseudo_inverse @ b
        noise = basis.T @ jrandom.normal(key, (basis.shape[0],))
        return x_star + noise

    return sample_solution



def inverse_add(c, in_constraints, out_constraints):
    pass