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
    **kwargs,
) -> jax.Array:
    """Functional wrapper for the SSD kernel (:func:`~probjax.nn.pallas_kernels.ssd`).

    Expects shapes matching `ssd_kernels.ssd`:
    - q, k: [B, G, L, dk]
    - v: [B, H, L, dv] with H % G == 0 (head ``h`` reads group ``h // (H/G)``)
    - log_alpha: [B, H, L]
    - h0: optional [B, H, dk, dv]
    Returns: [B, H, L, dv]

    Keyword arguments (``implementation``, ``chunk_size``, ``remat``,
    ``precision``, ``return_final_state``) are forwarded to the kernel.
    """
    return pallas_ssd(q, k, v, log_alpha, h0, **kwargs)


class ShortCausalConv(nnx.Module):
    """Depthwise causal 1D convolution over the sequence axis of ``[B, L, C]``."""

    def __init__(self, channels: int, kernel_size: int, rngs: nnx.Rngs):
        self.kernel_size = int(kernel_size)
        bound = 1.0 / jnp.sqrt(self.kernel_size)
        self.kernel = nnx.Param(
            jax.random.uniform(
                rngs.params(), (self.kernel_size, channels), minval=-bound, maxval=bound
            )
        )
        self.bias = nnx.Param(jnp.zeros((channels,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        kernel = self.kernel[...].astype(x.dtype)
        k = self.kernel_size
        xp = jnp.pad(x, ((0, 0), (k - 1, 0), (0, 0)))
        length = x.shape[1]
        # Short kernels (k ~ 4): a sum of shifted products beats a generic conv.
        y = sum(xp[:, i : i + length] * kernel[i] for i in range(k))
        return y + self.bias[...].astype(x.dtype)


def _dt_bias_init(key, num_heads, dt_min, dt_max, dt_init_floor):
    """Bias with ``softplus(bias)`` log-uniform in ``[dt_min, dt_max]``."""
    dt = jnp.exp(
        jax.random.uniform(key, (num_heads,), dtype=jnp.float32)
        * (jnp.log(dt_max) - jnp.log(dt_min))
        + jnp.log(dt_min)
    )
    dt = jnp.maximum(dt, dt_init_floor)
    return dt + jnp.log(-jnp.expm1(-dt))  # inverse softplus


def _a_log_init(key, num_heads, a_init_range):
    lo, hi = a_init_range
    if not 0 < lo <= hi:
        raise ValueError(f"a_init_range must satisfy 0 < lo <= hi, got {a_init_range}")
    a = jax.random.uniform(key, (num_heads,), minval=lo, maxval=hi)
    return jnp.log(a)


def _check_ssd_heads(num_heads, num_groups):
    if num_heads <= 0 or num_groups <= 0:
        raise ValueError(
            f"num_heads and num_groups must be positive, got {num_heads}, {num_groups}"
        )
    if num_heads % num_groups:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_groups={num_groups}"
        )


def ssd_mixer(
    x: jax.Array,
    b: jax.Array,
    c: jax.Array,
    dt: jax.Array,
    a_log: jax.Array,
    d: Optional[jax.Array],
    *,
    num_heads: int,
    num_groups: int,
    **ssd_kwargs,
) -> jax.Array:
    """Discretized multi-head selective SSM (the Mamba-2 SSM core).

    Per head ``h`` (group ``g = h // (H / G)``) and step ``t``::

        a_t = exp(-exp(a_log_h) * dt_{t,h})                # in (0, 1)
        S_t = a_t S_{t-1} + dt_{t,h} b_{t,g} x_{t,h}^T     # [N, P]
        y_{t,h} = S_t^T c_{t,g} + d_h x_{t,h}

    Args:
        x: ``[B, L, H * P]`` head inputs.
        b, c: ``[B, L, G * N]`` input/output state projections.
        dt: ``[B, L, H]`` positive step sizes.
        a_log: ``[H]``; the continuous decay is ``-exp(a_log) < 0``.
        d: ``[H]`` skip weights or ``None``.
        num_heads, num_groups: ``H`` and ``G`` (``H % G == 0``).
        **ssd_kwargs: forwarded to :func:`ssd`.

    Returns:
        ``[B, L, H * P]``.
    """
    batch, length, width = x.shape
    head_dim = width // num_heads
    state_dim = b.shape[-1] // num_groups
    dt = dt.astype(jnp.float32)
    log_alpha = (-jnp.exp(a_log.astype(jnp.float32)) * dt).transpose(0, 2, 1)
    # The SSM itself runs in float32 (the kernel's contract, and what the
    # reference implementation does for low-precision inputs).
    xh = x.reshape(batch, length, num_heads, head_dim).astype(jnp.float32)
    v = (xh * dt[..., None]).transpose(0, 2, 1, 3)
    k = b.reshape(batch, length, num_groups, state_dim).transpose(0, 2, 1, 3)
    q = c.reshape(batch, length, num_groups, state_dim).transpose(0, 2, 1, 3)
    y = ssd(q.astype(jnp.float32), k.astype(jnp.float32), v, log_alpha, **ssd_kwargs)
    y = y.transpose(0, 2, 1, 3)
    if d is not None:
        y = y + xh * d.astype(jnp.float32)[:, None]
    return y.reshape(batch, length, width).astype(x.dtype)


class SSDCell(RecurrentCell):
    """Multi-head SSD (Mamba-2 SSM) cell mapping ``[B, L, D] -> [B, L, D]``.

    The bare selective SSM of Mamba-2 between linear projections (see
    :func:`ssd_mixer`)::

        x = W_x u,  b = W_b u,  c = W_c u,  dt = softplus(W_dt u + dt_bias)
        y = W_o SSD(x, b, c, dt; A = -exp(a_log), d)

    ``num_heads`` heads of size ``head_dim`` share ``num_groups`` groups of
    ``b``/``c`` (``state_dim`` each), like grouped-query attention. The decay
    is ``-exp(a_log) * dt < 0`` by construction, so the recurrence is stable
    at any length. For the full Mamba-2 block (expansion, short convolution,
    gated RMSNorm) use :class:`Mamba2Cell`.

    Reference: Dao & Gu, 2024 -- Transformers are SSMs (Mamba-2).
    """

    def __init__(
        self,
        model_dim: int,
        rngs,
        *,
        state_dim: int = 64,
        num_heads: int = 1,
        num_groups: int = 1,
        head_dim: int | None = None,
        use_skip: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        a_init_range: tuple[float, float] = (1.0, 16.0),
        implementation: str | None = None,
        chunk_size: int | None = None,
        remat: str = "full",
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ):
        num_heads, num_groups = int(num_heads), int(num_groups)
        _check_ssd_heads(num_heads, num_groups)
        if head_dim is None:
            if model_dim % num_heads:
                raise ValueError(
                    f"model_dim={model_dim} is not divisible by num_heads={num_heads}; "
                    "pass head_dim explicitly."
                )
            head_dim = model_dim // num_heads
        self.model_dim = model_dim
        self.state_dim = int(state_dim)
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = int(head_dim)
        self.inner_dim = num_heads * self.head_dim
        self.ssd_kwargs = dict(
            implementation=implementation, chunk_size=chunk_size, remat=remat
        )

        precision_kwargs = filter_precision_kwargs(
            nnx.Linear,
            dtype=dtype,
            precision=precision,
            param_dtype=param_dtype,
            preferred_element_type=preferred_element_type,
        )

        def lin(i, o):
            return nnx.Linear(i, o, use_bias=False, rngs=rngs, **precision_kwargs)

        gn = num_groups * self.state_dim
        self.to_x = lin(model_dim, self.inner_dim)
        self.to_b = lin(model_dim, gn)
        self.to_c = lin(model_dim, gn)
        self.to_dt = lin(model_dim, num_heads)
        self.to_out = lin(self.inner_dim, model_dim)
        key_a, key_dt = jax.random.split(rngs.params())
        self.a_log = nnx.Param(_a_log_init(key_a, num_heads, a_init_range))
        self.dt_bias = nnx.Param(
            _dt_bias_init(key_dt, num_heads, dt_min, dt_max, dt_init_floor)
        )
        self.d = nnx.Param(jnp.ones((num_heads,), jnp.float32)) if use_skip else None

    def __call__(self, inputs: jax.Array, *, rng: jax.Array | None = None) -> jax.Array:
        del rng
        added_batch = inputs.ndim == 2
        if added_batch:
            inputs = inputs[None]
        dt = jax.nn.softplus(self.to_dt(inputs).astype(jnp.float32) + self.dt_bias[...])
        y = ssd_mixer(
            self.to_x(inputs),
            self.to_b(inputs),
            self.to_c(inputs),
            dt,
            self.a_log[...],
            None if self.d is None else self.d[...],
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            **self.ssd_kwargs,
        )
        y = self.to_out(y)
        if added_batch:
            y = y[0]
        return constrain(y, BATCH)


class Mamba2Cell(RecurrentCell):
    """Mamba-2 block mapping ``[B, L, D] -> [B, L, D]``.

    Follows the reference ``Mamba2`` module (``mamba_ssm/modules/mamba2.py``)::

        z, xbc, dt_raw = split(W_in u)                 # E = expand * D
        x, b, c = split(SiLU(ShortConv(xbc)))          # conv over x, b, c
        dt = softplus(dt_raw + dt_bias)
        y  = SSD(x, b, c, dt; A = -exp(a_log), d)      # see ssd_mixer
        out = W_out GroupRMSNorm(y * SiLU(z))          # num_groups norm groups

    ``head_dim`` defaults to 64 (``num_heads = E / head_dim``), falling back
    to a single head when ``E`` is not a multiple of 64. Initialization
    matches the reference: ``A ~ U[1, 16]``, ``softplus(dt_bias)``
    log-uniform in ``[dt_min, dt_max]``, ``d = 1``. Can be used as the
    ``recurrent_cls`` of :class:`probjax.nn.nets.ssm.SSMModel`.

    Reference: Dao & Gu, 2024 -- Transformers are SSMs: Generalized Models
    and Efficient Algorithms Through Structured State Space Duality.
    """

    def __init__(
        self,
        model_dim: int,
        rngs,
        *,
        expand: int = 2,
        state_dim: int = 64,
        head_dim: int | None = None,
        num_groups: int = 1,
        conv_size: int = 4,
        use_skip: bool = True,
        norm_eps: float = 1e-5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        a_init_range: tuple[float, float] = (1.0, 16.0),
        implementation: str | None = None,
        chunk_size: int | None = None,
        remat: str = "full",
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ):
        inner = int(expand) * model_dim
        if head_dim is None:
            head_dim = 64 if inner % 64 == 0 else inner
        if inner % head_dim:
            raise ValueError(
                f"expand * model_dim = {inner} is not divisible by head_dim={head_dim}"
            )
        num_heads, num_groups = inner // head_dim, int(num_groups)
        _check_ssd_heads(num_heads, num_groups)
        self.model_dim = model_dim
        self.inner_dim = inner
        self.state_dim = int(state_dim)
        self.head_dim = int(head_dim)
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.norm_eps = norm_eps
        self.ssd_kwargs = dict(
            implementation=implementation, chunk_size=chunk_size, remat=remat
        )

        precision_kwargs = filter_precision_kwargs(
            nnx.Linear,
            dtype=dtype,
            precision=precision,
            param_dtype=param_dtype,
            preferred_element_type=preferred_element_type,
        )
        gn = num_groups * self.state_dim
        self.conv_dim = inner + 2 * gn
        self.in_proj = nnx.Linear(
            model_dim,
            inner + self.conv_dim + num_heads,
            use_bias=False,
            rngs=rngs,
            **precision_kwargs,
        )
        self.conv = ShortCausalConv(self.conv_dim, conv_size, rngs)
        self.out_proj = nnx.Linear(
            inner, model_dim, use_bias=False, rngs=rngs, **precision_kwargs
        )
        key_a, key_dt = jax.random.split(rngs.params())
        self.a_log = nnx.Param(_a_log_init(key_a, num_heads, a_init_range))
        self.dt_bias = nnx.Param(
            _dt_bias_init(key_dt, num_heads, dt_min, dt_max, dt_init_floor)
        )
        self.d = nnx.Param(jnp.ones((num_heads,), jnp.float32)) if use_skip else None
        self.norm_scale = nnx.Param(jnp.ones((inner,), jnp.float32))

    def __call__(self, inputs: jax.Array, *, rng: jax.Array | None = None) -> jax.Array:
        del rng
        added_batch = inputs.ndim == 2
        if added_batch:
            inputs = inputs[None]
        batch, length, _ = inputs.shape
        inner, gn = self.inner_dim, self.num_groups * self.state_dim

        zxbcdt = self.in_proj(inputs)
        z = zxbcdt[..., :inner]
        xbc = zxbcdt[..., inner : inner + self.conv_dim]
        dt_raw = zxbcdt[..., inner + self.conv_dim :]
        xbc = jax.nn.silu(self.conv(xbc))
        x, b, c = xbc[..., :inner], xbc[..., inner : inner + gn], xbc[..., inner + gn :]
        dt = jax.nn.softplus(dt_raw.astype(jnp.float32) + self.dt_bias[...])
        y = ssd_mixer(
            x,
            b,
            c,
            dt,
            self.a_log[...],
            None if self.d is None else self.d[...],
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            **self.ssd_kwargs,
        )

        # Gated RMSNorm (norm after gating), one normalization group per
        # b/c group as in the reference implementation.
        g = y.astype(jnp.float32) * jax.nn.silu(z.astype(jnp.float32))
        g = g.reshape(batch, length, self.num_groups, inner // self.num_groups)
        g = g * jax.lax.rsqrt(jnp.mean(g * g, axis=-1, keepdims=True) + self.norm_eps)
        g = g.reshape(batch, length, inner) * self.norm_scale[...]
        out = self.out_proj(g.astype(inputs.dtype))
        if added_batch:
            out = out[0]
        return constrain(out, BATCH)


# Canonical public API: LRUCell, MambaCell, SSDCell, Mamba2Cell
