from functools import partial
from typing import Callable, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.nn.nets.mlp import MLP


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
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        rngs,
        *,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
    ):
        """Initialize the Linear Recurrent Unit (LRU) layer.
        This layer implements a Linear Recurrent Unit, which is a type of recurrent
        neural network hat uses complex-valued representations of linear dynamics.

        NOTE: Expressivity is limited to linear dynamics. But recurrent dynamics can be
        parallelized!!!
        NOTE: Presumes an initial state of zero.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            hidden_dim (int): Hidden state dimension.
            rngs: Random number generator keys for parameter initialization.
            r_min (float, optional): Minimum value for the decay rate. Defaults to 0.0.
            r_max (float, optional): Maximum value for the decay rate. Defaults to 1.0.
            max_phase (float, optional): Maximum phase value for theta initialization. Defaults to 6.28.
        Attributes:
            theta_log (nnx.Param): Log of theta parameters controlling the phase.
            nu_log (nnx.Param): Log of nu parameters controlling the decay rate.
            gamma_log (nnx.Param): Log of gamma parameters for scaling.
            B_re (nnx.Param): Real part of the input projection matrix.
            B_im (nnx.Param): Imaginary part of the input projection matrix.
            C_re (nnx.Param): Real part of the output projection matrix.
            C_im (nnx.Param): Imaginary part of the output projection matrix.
            D (nnx.Param): Direct input-to-output connection matrix.
        """

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
        self.D = nnx.Param(matrix_init(rngs.params(), (out_dim, in_dim)))

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

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[-2], axis=-2)
        Bu_elements = jnp.einsum("ih,ti->th", B_norm, inputs)

        # Compute hidden states
        _, hidden_states = jax.lax.associative_scan(
            binary_operator_diag, (Lambda_elements, Bu_elements)
        )
        # Use them to compute the output of the module
        outputs = jnp.real(jnp.einsum("th,ih->ti", hidden_states, C))
        outputs += jnp.einsum("ti,io->to", inputs, D)

        return outputs


class LRUBlock(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        model_dim: int,
        rngs,
        *,
        dropout: Optional[float] = None,
        norm: nnx.Module = nnx.LayerNorm,
        activation: Callable = jax.nn.gelu,
    ):
        """Initialize a Linear Recurrent Unit (LRU) block.
        This is a stackable bloc of LRUs with a residual connection and a
        Gated Linear Unit (GLU) output.
        """
        self.lru = LRU(model_dim, model_dim, model_dim, rngs)
        self.norm = norm(model_dim, rngs=rngs)
        self.activation = activation
        self.dropout = dropout
        if dropout is not None:
            self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
            self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
        self.out1 = nnx.Linear(model_dim, model_dim, rngs=rngs)
        self.out2 = nnx.Linear(model_dim, model_dim, rngs=rngs)

    def __call__(self, inputs, deterministic=False):
        x = self.norm(inputs)
        x = jax.vmap(self.lru)(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout1(x, deterministic=deterministic)
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        if self.dropout is not None:
            x = self.dropout2(x, deterministic=deterministic)
        return x


class LRUModel(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        output_dim: int,
        n_layers: int,
        rngs,
        *,
        bidirectional: bool = True,
        dropout: Optional[float] = None,
        norm: nnx.Module = nnx.LayerNorm,
        activation: Callable = jax.nn.gelu,
    ):
        self.bidirectional = bidirectional

        self.in_layer = nnx.Linear(input_dim, model_dim, rngs=rngs)
        self.out_layer = nnx.Linear(model_dim, output_dim, rngs=rngs)
        self.layers = [
            LRUBlock(
                model_dim,
                rngs,
                dropout=dropout,
                norm=norm,
                activation=activation,
            )
            for _ in range(n_layers)
        ]
        self.mlp_layers = [
            MLP([model_dim, 2 * model_dim, model_dim], rngs=rngs)
            for _ in range(n_layers)
        ]

    def __call__(self, inputs, *args, **kwargs):
        h = self.in_layer(inputs)
        for i, (layer, mlp) in enumerate(zip(self.layers, self.mlp_layers)):
            if self.bidirectional:
                # Alternate between forward and backward layers
                h = layer(h) if i % 2 == 0 else layer(h[:, ::-1])[:, ::-1]
            else:
                h = layer(h)
            h_new = mlp(h)
            h = h + h_new

        out = self.out_layer(h)
        return out
