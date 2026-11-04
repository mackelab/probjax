from typing import Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.nn.pallas_kernels import compute_mamba_scan
from probjax.nn.pallas_kernels import ssd as pallas_ssd
from probjax.nn.sharding import BATCH, constrain, replicate
from probjax.nn.utils import (
    filter_precision_kwargs,
)
from probjax.utils.typing import ArrayLike, DTypeLike, PrecisionLike


@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence."""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


@jax.custom_jvp
def linear_recurrence_scan(decay: jax.Array, inputs: jax.Array) -> jax.Array:
    """All states of ``h_t = decay * h_{t-1} + inputs_t`` with ``h_{-1} = 0``.

    ``decay`` has shape ``[S]`` (time-invariant, diagonal; real or complex)
    and ``inputs`` ``[..., T, S]``; returns ``[..., T, S]``.

    Same odd/even recursion as ``lax.associative_scan`` with
    :func:`binary_operator_diag`, but the decay of a block of ``2**k`` steps
    is the constant ``decay**(2**k)`` instead of a scanned ``[T, S]`` array.
    Its derivative is the same scan of the tangent inputs (custom JVP), so
    the backward pass is one reversed scan. Together this keeps the HLO
    small: the gradient compiles ~10x faster than the generic associative
    scan and runs ~2x faster.
    """
    return _linear_recurrence(decay, inputs)


@linear_recurrence_scan.defjvp
def _linear_recurrence_scan_jvp(primals, tangents):
    decay, inputs = primals
    d_decay, d_inputs = tangents
    states = _linear_recurrence(decay, inputs)
    # d h_t = decay * d h_{t-1} + (d inputs_t + d decay * h_{t-1}).
    previous = jnp.concatenate(
        [jnp.zeros_like(states[..., :1, :]), states[..., :-1, :]], axis=-2
    )
    return states, _linear_recurrence(decay, d_inputs + d_decay * previous)


def _linear_recurrence(decay: jax.Array, inputs: jax.Array) -> jax.Array:
    length = inputs.shape[-2]
    if length == 1:
        return inputs
    batch, width = inputs.shape[:-2], inputs.shape[-1]
    half = length // 2
    pairs = inputs[..., : 2 * half, :].reshape(batch + (half, 2, width))
    # States at odd positions 1, 3, 5, ... from the pair-combined inputs.
    odd = _linear_recurrence(decay * decay, decay * pairs[..., 0, :] + pairs[..., 1, :])
    # States at even positions 0, 2, 4, ...: one more step from the odd ones.
    rest = inputs[..., 2::2, :]
    even = jnp.concatenate(
        [inputs[..., :1, :], decay * odd[..., : rest.shape[-2], :] + rest], axis=-2
    )
    out = jnp.stack([even[..., :half, :], odd], axis=-2)
    out = out.reshape(batch + (2 * half, width))
    if length % 2:
        out = jnp.concatenate([out, even[..., -1:, :]], axis=-2)
    return out


def matrix_init(rng, shape, dtype=jnp.float32, normalization: ArrayLike = 1.0):
    return jax.random.normal(key=rng, shape=shape, dtype=dtype) / normalization


def nu_init(rng, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=rng, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(rng, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(rng, shape=shape, dtype=dtype)
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

        nu_log = self.nu_log[...]
        theta_log = self.theta_log[...]
        gain = jnp.exp(self.gamma_log[...])

        # Diagonal dynamics
        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        # The input is real: project with the real and imaginary parts of
        # B separately (two real matmuls instead of one complex one).
        Bu_elements = jax.lax.complex(
            jnp.einsum("...ti,ih->...th", inputs, self.B_re[...] * gain),
            jnp.einsum("...ti,ih->...th", inputs, self.B_im[...] * gain),
        )
        hidden_states = linear_recurrence_scan(diag_lambda, Bu_elements)
        # Only the real part of C h is used: Re(C h) = C_re h_re - C_im h_im.
        out = jnp.einsum(
            "...th,oh->...to", jnp.real(hidden_states), self.C_re[...]
        ) - jnp.einsum("...th,oh->...to", jnp.imag(hidden_states), self.C_im[...])
        out = out + jnp.einsum("...ti,oi->...to", inputs, self.D[...])
        return constrain(out, BATCH)


# -------------------------------- Mamba --------------------------------------


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

    On CPU (no Pallas backend) this runs the pure-JAX associative-scan
    recurrence.

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

    Parameterization and initialization follow the official implementation
    (``mamba_ssm/modules/mamba_simple.py``): the decay matrix is stored as
    ``a_log`` and used as ``a = -exp(a_log)`` so it stays negative (S4D
    init), the step size is ``delta = softplus(to_delta(x))`` with the
    projection bias set so ``softplus(bias)`` starts in
    ``[dt_min, dt_max]``, and the skip parameter ``d`` starts at one.

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
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ):
        precision_kwargs = dict(
            dtype=dtype,
            precision=precision,
            param_dtype=param_dtype,
            preferred_element_type=preferred_element_type,
        )
        sd = state_dim or model_dim
        self.model_dim = model_dim
        self.state_dim = sd
        self.seq_tile_size = seq_tile_size
        self.dim_tile_size = dim_tile_size

        # S4D real init: a = -(1..S), stored in log space to stay negative.
        a = jnp.tile(jnp.arange(1, sd + 1, dtype=jnp.float32)[:, None], (1, model_dim))
        self.a_log = nnx.Param(jnp.log(a))

        # Skip parameter starts at one.
        self.d = nnx.Param(jnp.ones((1, model_dim), dtype=jnp.float32))

        # Precision/dtype kwargs for linear projections
        precision_kwargs = filter_precision_kwargs(nnx.Linear, **precision_kwargs)

        # Token-wise generators for b, c, delta
        self.to_b = nnx.Linear(model_dim, sd, rngs=rngs, **precision_kwargs)
        self.to_c = nnx.Linear(model_dim, sd, rngs=rngs, **precision_kwargs)
        self.to_delta = nnx.Linear(model_dim, model_dim, rngs=rngs, **precision_kwargs)

        # Init the delta bias so softplus(bias) lands in [dt_min, dt_max].
        key = rngs.params()
        dt = jnp.exp(
            jax.random.uniform(key, (model_dim,), dtype=jnp.float32)
            * (jnp.log(dt_max) - jnp.log(dt_min))
            + jnp.log(dt_min)
        )
        dt = jnp.maximum(dt, dt_init_floor)
        bias = self.to_delta.bias
        assert bias is not None, "to_delta must be built with a bias."
        bias[...] = dt + jnp.log(-jnp.expm1(-dt))

    def __call__(self, inputs: jax.Array, *, rng: jax.Array | None = None) -> jax.Array:
        del rng
        added_batch = False
        if inputs.ndim == 2:
            inputs = inputs[None, ...]
            added_batch = True

        a = -jnp.exp(self.a_log[...])  # [S, D], always negative
        b = self.to_b(inputs)  # [B, L, S]
        c = self.to_c(inputs)  # [B, L, S]
        delta = jax.nn.softplus(self.to_delta(inputs))  # [B, L, D], positive

        y = mamba_scan(
            inputs,
            a,
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
        precision_kwargs = dict(
            dtype=dtype,
            precision=precision,
            param_dtype=param_dtype,
            preferred_element_type=preferred_element_type,
        )
        sd = state_dim or model_dim
        self.model_dim = model_dim
        self.state_dim = sd
        self.num_heads = int(num_heads)
        if reduce not in ("sum", "mean"):
            raise ValueError("reduce must be 'sum' or 'mean'")
        self.reduce = reduce

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
