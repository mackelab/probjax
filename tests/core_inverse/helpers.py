from typing import Any, Callable

import jax
import jax.numpy as jnp


def logdet_via_autodiff(f: Callable, x: Any) -> Any:
    """Compute the inverse log-absolute-determinant from the forward Jacobian."""
    jacobian = jax.jacobian(lambda value: f(value).flatten())(x)
    square_shape = (x.size, x.size)
    return -jnp.log(
        jnp.abs(jnp.linalg.det(jnp.reshape(jacobian, square_shape)))
    )
