import jax
import jax.numpy as jnp


def masked_mean(x, mask, axis=-1):
    num_elements = jnp.sum(mask, axis=axis)
    num_elements = jnp.maximum(num_elements, 1)  # Avoid division by zero
    return jnp.sum(x * mask, axis=axis) / num_elements
