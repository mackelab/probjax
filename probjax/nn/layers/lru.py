from typing import Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.nn.pallas_kernels import compute_mamba_scan, ssd as pallas_ssd
from probjax.nn.sharding import BATCH, constrain, replicate
from probjax.nn.utils import (
    filter_precision_kwargs,
    get_active_precision_kwargs,
)
from probjax.utils.typing import DTypeLike, PrecisionLike


@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence."""
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


class RecurrentCell(nnx.Module):
    """Abstract recurrent cell mapping [B, L, D] → [B, L, D].

    Subclasses implement sequence transforms with a fixed model dimension.
    """

    def __call__(
        self, inputs: jax.Array, *, rng: jax.Array | None = None
    ) -> jax.Array:  # pragma: no cover - abstract
        del rng
        del inputs
        raise NotImplementedError


class LRUCell(RecurrentCell):
    """Linear Recurrent Unit (LRU) cell mapping [B, L, D] → [B, L, D].

    Implements a linear recurrent unit with complex-valued state dynamics,
    mapping inputs over a sequence to outputs via diagonal linear recurrences.
    Presumes an initial state of zero.

    Reference: Orvieto et al., 2023 — Linear Recurrent Units (LRU).
    """

    def __init__(
        self,
        model_dim: int,
        rngs,
        *,
        state_dim: int | None = None,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
    ):
        state_dim = state_dim or model_dim

        self.model_dim = model_dim
        self.state_dim = state_dim
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        # Scale and shift parameters
        self.theta_log = nnx.Param(
            theta_init(rngs.params(), (state_dim,), max_phase=self.max_phase)
        )
        self.nu_log = nnx.Param(nu_init(rngs.next(), (state_dim,), r_min, r_max))
        self.gamma_log = nnx.Param(
            gamma_log_init(rngs.params(), (self.nu_log, self.theta_log))
        )

        # Projection matrices
        B_re = matrix_init(
            rngs.params(),
            (model_dim, state_dim),
            normalization=jnp.sqrt(2 * model_dim),
        )
        self.B_re = nnx.Param(B_re)
        B_im = matrix_init(
            rngs.params(),
            (model_dim, state_dim),
            normalization=jnp.sqrt(2 * model_dim),
        )
        self.B_im = nnx.Param(B_im)
        C_re = matrix_init(
            rngs.params(),
            (model_dim, state_dim),
            normalization=jnp.sqrt(state_dim),
        )
        self.C_re = nnx.Param(C_re)
        C_im = matrix_init(
            rngs.params(),
            (model_dim, state_dim),
            normalization=jnp.sqrt(state_dim),
        )
        self.C_im = nnx.Param(C_im)
        self.D = nnx.Param(matrix_init(rngs.params(), (model_dim, model_dim)))

    def __call__(self, inputs: jax.Array, *, rng: jax.Array | None = None) -> jax.Array:
        del rng
        inputs = jnp.asarray(inputs)
        # The recurrence requires gathered (unsharded) inputs.
        inputs = replicate(inputs)

        def _single(x_td):
            # Parameters
            nu_log = self.nu_log[...]
            theta_log = self.theta_log[...]
            gamma_log = self.gamma_log[...]

            B_re = self.B_re[...]
            B_im = self.B_im[...]
            C_re = self.C_re[...]
            C_im = self.C_im[...]
            D = self.D[...]

            # Diagonal dynamics
            diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
            B_norm = (B_re + 1j * B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-2)
            C = C_re + 1j * C_im

            Lambda_elements = jnp.repeat(
                diag_lambda[None, ...], x_td.shape[-2], axis=-2
            )
            Bu_elements = jnp.einsum("ih,ti->th", B_norm, x_td)
            _, hidden_states = jax.lax.associative_scan(
                binary_operator_diag, (Lambda_elements, Bu_elements)
            )
            outputs = jnp.real(jnp.einsum("th,oh->to", hidden_states, C))
            outputs += jnp.einsum("ti,oi->to", x_td, D)
            return outputs

        out = jax.vmap(_single)(inputs) if inputs.ndim == 3 else _single(inputs)
        return constrain(out, BATCH)


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


class MambaCell(RecurrentCell):
    """Mamba cell mapping [B, L, D] → [B, L, D].

    Wraps the Pallas Mamba scan kernel with token-wise projections to generate
    parameters and learnable recurrent matrices.

    Reference: Gu & Dao, 2023 — Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces.
    """

    def __init__(
        self,
        model_dim: int,
        rngs,
        *,
        state_dim: int | None = None,
        seq_tile_size: int = 64,
        dim_tile_size: int = 128,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ):
        sd = state_dim or model_dim
        self.model_dim = model_dim
        self.state_dim = sd
        self.seq_tile_size = seq_tile_size
        self.dim_tile_size = dim_tile_size

        # Recurrent parameters
        self.a = nnx.Param(
            matrix_init(
                rngs.params(), (sd, model_dim), normalization=jnp.sqrt(model_dim)
            )
        )
        self.d = nnx.Param(
            matrix_init(
                rngs.params(), (1, model_dim), normalization=jnp.sqrt(model_dim)
            )
        )

        # Precision/dtype kwargs for linear projections
        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(nnx.Linear, **precision_kwargs)

        # Token-wise generators for b, c, delta
        self.to_b = nnx.Linear(model_dim, sd, rngs=rngs, **precision_kwargs)
        self.to_c = nnx.Linear(model_dim, sd, rngs=rngs, **precision_kwargs)
        self.to_delta = nnx.Linear(model_dim, model_dim, rngs=rngs, **precision_kwargs)

    def __call__(self, inputs: jax.Array, *, rng: jax.Array | None = None) -> jax.Array:
        del rng
        added_batch = False
        if inputs.ndim == 2:
            inputs = inputs[None, ...]
            added_batch = True

        b = self.to_b(inputs)  # [B, L, S]
        c = self.to_c(inputs)  # [B, L, S]
        delta = self.to_delta(inputs)  # [B, L, D]

        y = mamba_scan(
            inputs,
            self.a[...],
            b,
            c,
            delta,
            self.d[...],
            seq_tile_size=self.seq_tile_size,
            dim_tile_size=self.dim_tile_size,
        )
        if added_batch:
            y = y[0]
        return constrain(y, BATCH)


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
    return pallas_ssd(q, k, v, log_alpha, h0)


class SSDCell(RecurrentCell):
    """SSD (Mamba-2 style) cell mapping [B, L, D] → [B, L, D].

    Uses the Pallas SSD kernel with single group and `num_heads` value pathways.
    Aggregates heads by summation by default and preserves the input dimension.

    Reference: Dao et al., 2024 — Mamba-2 / Selective SSMs.
    """

    def __init__(
        self,
        model_dim: int,
        rngs,
        *,
        state_dim: int | None = None,
        num_heads: int = 1,
        reduce: str = "sum",
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ):
        sd = state_dim or model_dim
        self.model_dim = model_dim
        self.state_dim = sd
        self.num_heads = int(num_heads)
        if reduce not in ("sum", "mean"):
            raise ValueError("reduce must be 'sum' or 'mean'")
        self.reduce = reduce

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(nnx.Linear, **precision_kwargs)

        self.to_q = nnx.Linear(model_dim, sd, rngs=rngs, **precision_kwargs)
        self.to_k = nnx.Linear(model_dim, sd, rngs=rngs, **precision_kwargs)
        if (model_dim % self.num_heads) == 0:
            self.dv = model_dim // self.num_heads
        else:
            self.dv = model_dim
        self.to_v = nnx.Linear(model_dim, self.dv, rngs=rngs, **precision_kwargs)
        self.to_alpha = nnx.Linear(
            model_dim, self.num_heads, rngs=rngs, **precision_kwargs
        )
        self.post = None  # by construction preserves model_dim

    def __call__(self, inputs: jax.Array, *, rng: jax.Array | None = None) -> jax.Array:
        del rng
        added_batch = False
        if inputs.ndim == 2:
            inputs = inputs[None, ...]
            added_batch = True

        B, L, _ = inputs.shape
        G = self.num_heads
        H = self.num_heads

        q = self.to_q(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        q = q.reshape(B, 1, L, self.state_dim).repeat(G, axis=1)
        k = k.reshape(B, 1, L, self.state_dim).repeat(G, axis=1)
        v = v.reshape(B, 1, L, self.dv).repeat(H, axis=1)
        log_alpha = self.to_alpha(inputs)
        log_alpha = jnp.swapaxes(log_alpha, 1, 2)

        out = ssd(q, k, v, log_alpha, h0=None)
        y = out.sum(axis=1) if self.reduce == "sum" else out.mean(axis=1)
        if added_batch:
            y = y[0]
        return constrain(y, BATCH)


# Uniform cell-style wrappers returning [B, L, D]


# Canonical public API: LRUCell, MambaCell, SSDCell
