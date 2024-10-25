from functools import partial
from typing import Callable, Optional

import flax.nnx as nnx
import jax.numpy as jnp
import jax


@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


def nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(key, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class LRU(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        rngs,
        *,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        # Scale and shift parameters
        self.theta_log = nnx.Param(
            theta_init(rngs.params(), (self.hidden_dim,), max_phase=self.max_phase)
        )
        self.nu_log = nnx.Param(nu_init(rngs.next(), (self.hidden_dim,), r_min, r_max))
        self.gamma_log = nnx.Param(
            gamma_log_init(rngs.params(), (self.nu_log, self.theta_log))
        )

        # Projection matrices
        B_re = matrix_init(
            rngs.params(),
            (in_dim, hidden_dim),
            normalization=jnp.sqrt(2 * self.in_dim),
        )
        self.B_re = nnx.Param(B_re)
        B_im = matrix_init(
            rngs.params(),
            (in_dim, hidden_dim),
            normalization=jnp.sqrt(2 * self.in_dim),
        )
        self.B_im = nnx.Param(B_im)
        C_re = matrix_init(
            rngs.params(),
            (out_dim, hidden_dim),
            normalization=jnp.sqrt(self.hidden_dim),
        )
        self.C_re = nnx.Param(C_re)
        C_im = matrix_init(
            rngs.params(),
            (out_dim, hidden_dim),
            normalization=jnp.sqrt(self.hidden_dim),
        )
        self.C_im = nnx.Param(C_im)
        self.D = nnx.Param(matrix_init(rngs.params(), (out_dim,)))

    def __call__(self, inputs):
        # Fetch parameters
        nu_log = self.nu_log.value
        theta_log = self.theta_log.value
        gamma_log = self.gamma_log.value

        # Fetch projection matrices
        B_re = self.B_re.value
        B_im = self.B_im.value
        C_re = self.C_re.value
        C_im = self.C_im.value
        D = self.D.value

        # Diag drift
        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))

        # Input projection
        B_norm = B_re + 1j * B_im
        B_norm = B_norm * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
        # Output projection
        C = C_re + 1j * C_im

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)
        Bu_elements = jnp.einsum("ih,ti->th", B_norm, inputs)

        # Compute hidden states
        _, hidden_states = jax.lax.associative_scan(
            binary_operator_diag, (Lambda_elements, Bu_elements)
        )
        # Use them to compute the output of the module
        outputs = jnp.real(jnp.einsum("th,ih->ti", hidden_states, C)) + D * inputs

        return outputs


class LRULayer(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        lru: LRU,
        output_dim: int,
        rngs,
        *,
        dropout: Optional[float] = None,
        norm: nnx.Module = nnx.LayerNorm,
        activation: Callable = jax.nn.gelu,
    ):
        self.lru = lru
        self.norm = norm(lru.in_dim, rngs=rngs)
        self.activation = activation
        self.dropout = dropout
        input_dim = lru.out_dim
        self.out1 = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.out2 = nnx.Linear(input_dim, output_dim, rngs=rngs)

    def __call__(self, inputs):
        x = self.norm(inputs)
        x = self.lru(x)
        x = self.activation(x)
        if self.dropout is not None:
            raise NotImplementedError("Dropout not implemented yet")
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        if self.dropout is not None:
            raise NotImplementedError("Dropout not implemented yet")
        return x
