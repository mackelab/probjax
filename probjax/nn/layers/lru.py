from typing import Callable, Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

# Advanced Pallas kernels (Mamba/SSD) user-facing wrappers
from probjax.nn.pallas_kernels.mambda import compute_mamba_scan


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


class LRU(nnx.Module):
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
            max_phase (float, optional): Maximum phase value for theta initialization.
                Defaults to 6.28.

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
        # Support [L, D] or [B, L, D]
        inputs = jnp.asarray(inputs)

        def _single(x_td):
            # x_td: [L, in_dim]
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
            B_norm = B_norm * jnp.expand_dims(jnp.exp(gamma_log), axis=-2)
            # Output projection
            C = C_re + 1j * C_im

            Lambda_elements = jnp.repeat(
                diag_lambda[None, ...], x_td.shape[-2], axis=-2
            )  # [L, H]
            Bu_elements = jnp.einsum("ih,ti->th", B_norm, x_td)  # [L, H]

            # Compute hidden states via associative scan
            _, hidden_states = jax.lax.associative_scan(
                binary_operator_diag, (Lambda_elements, Bu_elements)
            )
            # Project to output
            outputs = jnp.real(jnp.einsum("th,oh->to", hidden_states, C))
            outputs += jnp.einsum("ti,oi->to", x_td, D)
            return outputs

        if inputs.ndim == 3:
            return jax.vmap(_single)(inputs)

        # Fallback to single-example path [L, D]
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
        B_norm = B_norm * jnp.expand_dims(jnp.exp(gamma_log), axis=-2)
        # Output projection
        C = C_re + 1j * C_im

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[-2], axis=-2)
        Bu_elements = jnp.einsum("ih,ti->th", B_norm, inputs)

        # Compute hidden states
        _, hidden_states = jax.lax.associative_scan(
            binary_operator_diag, (Lambda_elements, Bu_elements)
        )
        # Use them to compute the output of the module
        outputs = jnp.real(jnp.einsum("th,oh->to", hidden_states, C))
        outputs += jnp.einsum("ti,oi->to", inputs, D)

        return outputs


# ----------------------------- Mamba LRU ------------------------------------


def mamba_scan(
    x: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    delta: jax.Array,
    d: jax.Array,
    *,
    seq_tile_size: int = 64,
    dim_tile_size: int = 128,
) -> jax.Array:
    """Functional wrapper over the Pallas Mamba scan kernel.

    Shapes follow the kernel contract:
    - x: [B, L, D]
    - a: [S, D]
    - b, c, delta: [B, L, S], [B, L, S], [B, L, D]
    - d: [1, D]
    Returns: y with shape [B, L, D]
    """
    return compute_mamba_scan(
        x, a, b, c, delta, d, seq_tile_size=seq_tile_size, dim_tile_size=dim_tile_size
    )


class MambaLRU(nnx.Module):
    """A simple nnx.Module wrapper around the Pallas Mamba scan.

    This module maps per-token inputs to Mamba parameters (b, c, delta) via linear
    projections, keeps recurrent matrices (a, d) as learnable parameters, runs the
    Pallas scan, and projects to `out_dim`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        state_dim: int,
        rngs,
        *,
        seq_tile_size: int = 64,
        dim_tile_size: int = 128,
        include_out_proj: bool = True,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.seq_tile_size = seq_tile_size
        self.dim_tile_size = dim_tile_size

        # Recurrent parameters
        self.a = nnx.Param(matrix_init(rngs.params(), (state_dim, in_dim), normalization=jnp.sqrt(in_dim)))
        self.d = nnx.Param(matrix_init(rngs.params(), (1, in_dim), normalization=jnp.sqrt(in_dim)))

        # Token-wise generators for b, c, delta
        self.to_b = nnx.Linear(in_dim, state_dim, rngs=rngs)
        self.to_c = nnx.Linear(in_dim, state_dim, rngs=rngs)
        self.to_delta = nnx.Linear(in_dim, in_dim, rngs=rngs)

        # Optional output projection
        self.out = None
        if include_out_proj:
            self.out = nnx.Linear(in_dim, out_dim, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        # Accept [L, D] or [B, L, D]
        added_batch = False
        if inputs.ndim == 2:
            inputs = inputs[None, ...]
            added_batch = True

        b = self.to_b(inputs)  # [B, L, S]
        c = self.to_c(inputs)  # [B, L, S]
        delta = self.to_delta(inputs)  # [B, L, D]

        y = mamba_scan(
            inputs,
            self.a.value,
            b,
            c,
            delta,
            self.d.value,
            seq_tile_size=self.seq_tile_size,
            dim_tile_size=self.dim_tile_size,
        )
        if self.out is not None:
            y = self.out(y)
        if added_batch:
            y = y[0]
        return y


# ------------------------------ SSD (Mamba-2) -------------------------------


def ssd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    h0: Optional[jax.Array] = None,
) -> jax.Array:
    """Functional wrapper for the Pallas SSD kernel.

    Expects shapes matching `ssd_kernels.ssd`:
    - q, k: [B, G, L, dk]
    - v: [B, H, L, dv] with H % G == 0
    - log_alpha: [B, H, L]
    - h0: optional [B, H, dk, dv]
    Returns: [B, H, L, dv]
    """
    # Lazy import to avoid optional dependency (einops) at module import time.
    from probjax.nn.pallas_kernels import ssd as _ssd_mod
    return _ssd_mod.ssd(q, k, v, log_alpha, h0)


class SSDLRU(nnx.Module):
    """A lightweight wrapper that builds SSD parameters from inputs.

    This module uses single group and `num_heads` value pathways by default
    and aggregates heads by summation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None,
        state_dim: int,
        rngs,
        *,
        num_heads: int = 1,
        reduce: str = "sum",
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.state_dim = state_dim
        self.num_heads = int(num_heads)
        if reduce not in ("sum", "mean"):
            raise ValueError("reduce must be 'sum' or 'mean'")
        self.reduce = reduce

        # Projections to q/k/v and per-step decay log_alpha
        self.to_q = nnx.Linear(in_dim, state_dim, rngs=rngs)
        self.to_k = nnx.Linear(in_dim, state_dim, rngs=rngs)
        # Set dv = out_dim // num_heads if divisible, else use out_dim per head and sum
        # choose dv so H*dv matches out_dim if divisible; else keep dv=in_dim and post-project later
        if (self.out_dim % self.num_heads) == 0:
            self.dv = self.out_dim // self.num_heads
        else:
            self.dv = in_dim
        self.to_v = nnx.Linear(in_dim, self.dv, rngs=rngs)
        self.to_alpha = nnx.Linear(in_dim, self.num_heads, rngs=rngs)
        # Optional post-proj if dv*H != out_dim
        self.post = None
        if self.num_heads * self.dv != self.out_dim:
            self.post = nnx.Linear(self.num_heads * self.dv, self.out_dim, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        # Accept [L, D] or [B, L, D]
        added_batch = False
        if inputs.ndim == 2:
            inputs = inputs[None, ...]
            added_batch = True

        B, L, _ = inputs.shape
        G = self.num_heads  # 1:1 mapping to avoid head-group mismatch
        H = self.num_heads

        q = self.to_q(inputs)  # [B, L, dk]
        k = self.to_k(inputs)
        v = self.to_v(inputs)  # [B, L, dv]
        # Arrange shapes for SSD
        q = q.reshape(B, 1, L, self.state_dim).repeat(G, axis=1)
        k = k.reshape(B, 1, L, self.state_dim).repeat(G, axis=1)
        v = v.reshape(B, 1, L, self.dv).repeat(H, axis=1)
        log_alpha = self.to_alpha(inputs)  # [B, L, H]
        log_alpha = jnp.swapaxes(log_alpha, 1, 2)  # [B, H, L]

        out = ssd(q, k, v, log_alpha, h0=None)  # [B, H, L, dv]
        if self.reduce == "sum":
            y = out.sum(axis=1)  # [B, L, dv]
        else:
            y = out.mean(axis=1)
        if self.post is not None:
            y = self.post(y)
        if added_batch:
            y = y[0]
        return y


# Uniform cell-style wrappers returning [B, L, D]


class LRUCell(nnx.Module):
    def __init__(self, model_dim: int, rngs, **kwargs):
        del kwargs
        self.core = LRU(model_dim, model_dim, model_dim, rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.core(inputs)


class MambaCell(nnx.Module):
    def __init__(self, model_dim: int, rngs, *, state_dim: int | None = None, seq_tile_size: int = 64, dim_tile_size: int = 128, **kwargs):
        del kwargs
        sd = state_dim or model_dim
        # No output projection to preserve dimension
        self.core = MambaLRU(model_dim, model_dim, sd, rngs, seq_tile_size=seq_tile_size, dim_tile_size=dim_tile_size, include_out_proj=False)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.core(inputs)


class SSDCell(nnx.Module):
    def __init__(self, model_dim: int, rngs, *, state_dim: int | None = None, num_heads: int = 1, reduce: str = "sum", **kwargs):
        del kwargs
        sd = state_dim or model_dim
        # Set out_dim=None so SSDLRU maps back to in_dim (model_dim)
        self.core = SSDLRU(model_dim, None, sd, rngs, num_heads=num_heads, reduce=reduce)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.core(inputs)


class LRUBlock(nnx.Module):
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

    def __call__(self, inputs, deterministic: bool | None = None):
        x = self.norm(inputs)
        x = jax.vmap(self.lru)(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout1(x, deterministic=deterministic)
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        if self.dropout is not None:
            x = self.dropout2(x, deterministic=deterministic)
        return x
