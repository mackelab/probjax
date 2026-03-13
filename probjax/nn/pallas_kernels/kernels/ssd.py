# Some of the code in this file is adapted from:
#
# Apple/axlearn
# https://github.com/apple/axlearn/blob/main/axlearn/common/ssm_kernels/ssd_kernels.py

"""Pallas kernels for Mamba2

High-level idea: this kernel implements a two-level chunking algorithm to
balance memory consumption and running speed. To reduce forward HBM traffic,
chunk-level states are recomputed in backward rather than stored from forward.


Notations:
    nb: number of chunks
    ns: number of subchunks
    bl: subchunk size
    dkn: number of tiles in the dk dim
    dvn: number of tiles in the dv dim
    dk: state_dim (corresponds to dim of qk heads)
    dv: head_dim (corresponds to dim of v heads)

q/k/v is used as it's more intuitive than b/c/x of SSD in the orginal implementation,
see section 7.2 https://arxiv.org/pdf/2405.21060. Accordingly, dk/dv is used instead
of state_dim/head_dim.  This notation is also used in linear attention models.
However, state_dim/head_dim is used in the model file to be consistent with Mamba1
and the original implementation.

"""

import functools
import os
import warnings
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from jax import lax
from jax.experimental import pallas as pl

from ..kernel_utils import get_dot_precision, use_interpret_mode


def _validate_ssd_sharding(sharding, name: str):
    """Validate that *sharding* (a NamedSharding) does not shard unsupported dims.

    Only batch (dim 0) and heads/groups (dim 1) sharding are supported for SSD.
    Sharding on the sequence (dim 2) or dk/dv (dim 3) is rejected with an
    informative error.
    """
    spec = getattr(sharding, "spec", None)
    if spec is None:
        return
    for dim_idx, axis in enumerate(spec):
        if axis is None:
            continue
        if dim_idx == 2:
            raise ValueError(
                f"Pallas SSD kernel does not support sharding on the "
                f"sequence dimension (dim 2) of '{name}'. "
                f"Got PartitionSpec{tuple(spec)} which shards dim 2 over "
                f"mesh axis '{axis}'. "
                f"Only batch (dim 0) and heads (dim 1) sharding are supported."
            )
        if dim_idx == 3:
            raise ValueError(
                f"Pallas SSD kernel does not support sharding on the "
                f"dk/dv dimension (dim 3) of '{name}'. "
                f"Got PartitionSpec{tuple(spec)} which shards dim 3 over "
                f"mesh axis '{axis}'. "
                f"Only batch (dim 0) and heads (dim 1) sharding are supported."
            )


def _bs(index_map, block_shape):
    """Compatibility wrapper for BlockSpec(index_map, block_shape) call sites."""
    return pl.BlockSpec(block_shape=block_shape, index_map=index_map)


def _tpu_compiler_params(*, dimension_semantics):
    """Returns TPU compiler params when running on TPU, else None."""
    if use_interpret_mode() or jax.default_backend() != "tpu":
        return None
    from jax.experimental.pallas import tpu as pltpu

    return pltpu.CompilerParams(dimension_semantics=dimension_semantics)


def _is_hopper_or_newer_gpu() -> bool:
    if jax.default_backend() != "gpu":
        return False
    kind = jax.devices()[0].device_kind.lower()
    return any(
        tag in kind for tag in ("h100", "h200", "b100", "b200", "hopper", "blackwell")
    )


def _prefer_mosaic_gpu() -> bool:
    val = os.environ.get("PROBJAX_PALLAS_PREFER_MOSAIC_GPU", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def _pallas_backend(prefer_mosaic_gpu: bool | None = None) -> str | None:
    """Selects the explicit Pallas backend for the current JAX platform."""
    if prefer_mosaic_gpu is None:
        prefer_mosaic_gpu = _prefer_mosaic_gpu()
    backend = jax.default_backend()
    if backend == "gpu":
        if prefer_mosaic_gpu and _is_hopper_or_newer_gpu():
            return "mosaic_gpu"
        return "triton"
    if backend == "tpu":
        return "mosaic_tpu"
    return None


def _ssd_tiling_config() -> tuple[int, int, int]:
    """Returns (singleton_dim, chunk_size, subchunk_size) for the current backend."""
    if jax.default_backend() == "gpu":
        # Triton path: reduce tile sizes to satisfy shared-memory limits on common GPUs.
        return 64, 256, 64
    # TPU-tuned defaults.
    return 128, 512, 64


@functools.lru_cache(maxsize=64)
def _gpu_supports_ssd_pallas_for_shape(
    *,
    seq_len: int,
    num_groups: int,
    num_heads: int,
    dk: int,
    dv: int,
    pallas_backend: str | None,
) -> bool:
    """Cheap/static capability check for SSD Pallas on GPU."""
    if jax.default_backend() != "gpu":
        return False
    if use_interpret_mode():
        return True
    if pallas_backend not in ("triton", "mosaic_gpu"):
        return False
    if pallas_backend == "mosaic_gpu" and not _is_hopper_or_newer_gpu():
        return False
    if pallas_backend == "triton" and num_heads < 2:
        return False
    if seq_len <= 0 or num_groups <= 0 or num_heads <= 0 or dk <= 0 or dv <= 0:
        return False
    singleton_dim, chunk_size, _ = _ssd_tiling_config()
    if seq_len % chunk_size != 0:
        return False
    if dk % singleton_dim != 0 or dv % singleton_dim != 0:
        return False
    if num_heads % num_groups != 0:
        return False
    return True


_FAILED_SSD_PALLAS_CONFIGS: set[tuple] = set()


def _ssd_pallas_failure_key(
    *,
    pallas_backend: str | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    h0: jax.Array,
) -> tuple:
    return (
        pallas_backend,
        q.shape,
        k.shape,
        v.shape,
        log_alpha.shape,
        h0.shape,
        str(q.dtype),
        str(k.dtype),
        str(v.dtype),
        str(log_alpha.dtype),
        str(h0.dtype),
    )


def _matmul_fp32(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """A wrapper around jax.lax.dot to conduct float32 matmul"""
    precision = get_dot_precision(jax.default_backend(), lhs.dtype)
    return jax.lax.dot(
        lhs, rhs, precision=precision, preferred_element_type=jnp.float32
    )


def _validate_ssd_runtime_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    h0: Optional[jax.Array],
) -> None:
    if q.ndim != 4:
        raise ValueError(f"`q` must be rank-4 [B, G, L, Dk], got shape {q.shape}.")
    if k.ndim != 4:
        raise ValueError(f"`k` must be rank-4 [B, G, L, Dk], got shape {k.shape}.")
    if v.ndim != 4:
        raise ValueError(f"`v` must be rank-4 [B, H, L, Dv], got shape {v.shape}.")
    if log_alpha.ndim != 3:
        raise ValueError(
            f"`log_alpha` must be rank-3 [B, H, L], got shape {log_alpha.shape}."
        )

    if q.shape != k.shape:
        raise ValueError(
            f"`q` and `k` must have the same shape, got {q.shape} and {k.shape}."
        )

    bs, ng, seq_len, dk = q.shape
    bs_v, nh, seq_len_v, dv = v.shape
    bs_a, nh_a, seq_len_a = log_alpha.shape

    if bs_v != bs or seq_len_v != seq_len:
        raise ValueError(
            f"`v` shape mismatch: expected batch/seq {(bs, seq_len)}, got {(bs_v, seq_len_v)}."
        )
    if bs_a != bs or nh_a != nh or seq_len_a != seq_len:
        raise ValueError(
            f"`log_alpha` shape mismatch: expected {(bs, nh, seq_len)}, got {log_alpha.shape}."
        )
    if ng <= 0 or nh <= 0:
        raise ValueError(
            f"`num_groups` and `num_heads` must be > 0, got ng={ng}, nh={nh}."
        )
    if nh % ng != 0:
        raise ValueError(
            f"`num_heads` must be divisible by `num_groups`, got nh={nh}, ng={ng}."
        )

    if v.dtype != jnp.float32:
        raise ValueError(f"`v` must be float32, got dtype {v.dtype}.")
    if log_alpha.dtype != jnp.float32:
        raise ValueError(f"`log_alpha` must be float32, got dtype {log_alpha.dtype}.")

    if h0 is not None:
        expected_h0_shape = (bs, nh, dk, dv)
        if h0.ndim != 4:
            raise ValueError(
                f"`h0` must be rank-4 [B, H, Dk, Dv], got shape {h0.shape}."
            )
        if h0.shape != expected_h0_shape:
            raise ValueError(
                f"`h0` shape mismatch: expected {expected_h0_shape}, got {h0.shape}."
            )


def _ssd_forward_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    initial_state: jax.Array,
) -> jax.Array:
    """Forward pass body for SSD (no @jax.jit — called from custom_partitioning).

    Args:
        q, k: [bs, num_groups, seq_len, dk]
        v: [bs, num_heads, seq_len, dv]
        log_alpha: [bs, num_heads, seq_len]
        initial_state: [bs, num_heads, dk, dv]

    Returns:
        o: [bs, num_heads, seq_len, dv]
    """
    o, _ = _ssd_forward(q, k, v, log_alpha, initial_state)
    return o


def _ssd_forward_kernel(
    q_ref: jax.Array,
    k_ref: jax.Array,
    v_ref: jax.Array,
    cum_log_alpha_ref: jax.Array,
    initial_state_ref: jax.Array,
    gamma_ref: jax.Array,
    causal_mask_ref: jax.Array,
    mutable_final_state_ref: jax.Array,
    mutable_o_ref: jax.Array,
):
    """Forward kernel for SSD.

    Args:
        q_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        k_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        v_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        cum_log_alpha_ref: jax.Array reference of shape [ns, bl]
        initial_state_ref: jax.Array reference of shape [singleton_dim, singleton_dim]
        gamma_ref: jax.Array reference of shape [ns, bl, singleton_dim]

    Output via mutable jax.Arrays:
        mutable_final_state_ref: jax.Array reference of shape [singleton_dim, singleton_dim]
        mutable_o_ref: jax.Array reference of shape [ns, bl, singleton_dim]

    Note on intial_state and final_state:
        * initial_state is at seq-level and not updated during the forward pass
        * final_state is used to pass chunk-level states across different chunks
            - it will be initialized to initial_state at the beginning of each chunk
            - it will be updated after processing each chunk
            - in the end, it will return as the seq-level final state
    """
    subchunk_dim, _ = cum_log_alpha_ref.shape[0], cum_log_alpha_ref.shape[1]
    causal_mask = causal_mask_ref[:]

    # In our grid definition, axis 4 is the chunk index.
    @pl.when(pl.program_id(axis=4) == 0)
    def init_carry():
        mutable_final_state_ref[:, :] = initial_state_ref[:, :]

    def _ssd_forward_chunk_loop_body(t: int, h_carry: jax.Array):
        subchunk_idx = t
        prev_state = h_carry

        q_block = q_ref[subchunk_idx, :].astype(jnp.float32)
        k_block = k_ref[subchunk_idx, :].astype(jnp.float32)
        v_block = v_ref[subchunk_idx, :].astype(jnp.float32)

        # Notation mapping wrt. the paper: lambda -> Lambda, gamma -> gamma, beta -> Gamma.
        lambda_block = cum_log_alpha_ref[subchunk_idx, :]
        gamma_block = gamma_ref[subchunk_idx]

        lambda_block = jnp.expand_dims(lambda_block, axis=-1)  # [bl, 1]
        beta_block = (
            jnp.expand_dims(gamma_block, axis=0) - lambda_block
        )  # [bl, singleton_dim] after broadcasting
        ssd_mask_block = lambda_block - jnp.transpose(lambda_block, [1, 0])
        ssd_mask_block = ssd_mask_block * causal_mask

        lambda_block = jnp.exp(lambda_block)
        beta_block = jnp.exp(beta_block)
        gamma_block = jnp.exp(gamma_block)
        ssd_mask_block = jnp.exp(ssd_mask_block)

        q_tilde_block = q_block * lambda_block
        k_tilde_block = k_block * beta_block

        o_block_inter = _matmul_fp32(q_tilde_block, prev_state)
        intra_att = _matmul_fp32(q_block, k_block.T)
        attn_mask = causal_mask * ssd_mask_block
        o_block_intra = _matmul_fp32((intra_att * attn_mask), v_block)
        o_block = o_block_inter + o_block_intra

        cur_state = prev_state * jnp.expand_dims(gamma_block, axis=-1) + _matmul_fp32(
            k_tilde_block.T, v_block
        )  # [d_k, d_v]
        mutable_o_ref[subchunk_idx, :] = o_block.astype(mutable_o_ref.dtype)
        return cur_state

    h_carry = mutable_final_state_ref[:, :]
    final_state = lax.fori_loop(0, subchunk_dim, _ssd_forward_chunk_loop_body, h_carry)
    mutable_final_state_ref[:, :] = final_state


def _ssd_chunk_states_kernel(
    k_ref: jax.Array,
    v_ref: jax.Array,
    cum_log_alpha_ref: jax.Array,
    initial_state_ref: jax.Array,
    gamma_ref: jax.Array,
    mutable_ch_ref: jax.Array,
    mutable_final_state_ref: jax.Array,
):
    """Kernel that recomputes chunk-level states for SSD backward."""
    subchunk_dim, _ = cum_log_alpha_ref.shape[0], cum_log_alpha_ref.shape[1]

    @pl.when(pl.program_id(axis=4) == 0)
    def init_carry():
        mutable_final_state_ref[:, :] = initial_state_ref[:, :]

    def _ssd_chunk_state_loop_body(t: int, h_carry: jax.Array):
        subchunk_idx = t
        prev_state = h_carry

        k_block = k_ref[subchunk_idx, :].astype(jnp.float32)
        v_block = v_ref[subchunk_idx, :].astype(jnp.float32)

        lambda_block = cum_log_alpha_ref[subchunk_idx, :]
        gamma_block = gamma_ref[subchunk_idx]

        lambda_block = jnp.expand_dims(lambda_block, axis=-1)
        beta_block = gamma_block - lambda_block

        beta_block = jnp.exp(beta_block)
        gamma_block = jnp.exp(gamma_block)

        k_tilde_block = k_block * beta_block
        cur_state = prev_state * jnp.expand_dims(gamma_block, axis=-1) + _matmul_fp32(
            k_tilde_block.T, v_block
        )
        return cur_state

    h_carry = mutable_final_state_ref[:, :]
    mutable_ch_ref[:, :] = h_carry
    final_state = lax.fori_loop(0, subchunk_dim, _ssd_chunk_state_loop_body, h_carry)
    mutable_final_state_ref[:, :] = final_state


@jax.jit
def _ssd_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    initial_state: jax.Array,
) -> Tuple:
    """Forward pass for SSD.

    Args:
        q, k: [bs, num_heads, seq_len, dk]
        v: [bs, num_heads, seq_len, dv]
        log_alpha: [bs, num_heads, seq_len]
        initial_state: [singleton_dim, singleton_dim]

    Returns:
        o: [bs, num_heads, seq_len, dv]
        residuals: Tuple of jax.Arrays to be used in the backward
    """
    bs, num_qk_heads, seq_len, k_head_dim = q.shape
    _, num_v_heads, _, v_head_dim = v.shape
    singleton_dim, chunk_size, subchunk_size = _ssd_tiling_config()
    acc_dtype, orig_dtype = jnp.float32, q.dtype

    assert seq_len % chunk_size == 0 and chunk_size % subchunk_size == 0

    assert num_v_heads % num_qk_heads == 0
    num_heads = num_v_heads
    num_head_per_group = num_v_heads // num_qk_heads

    assert k_head_dim % singleton_dim == 0
    assert v_head_dim % singleton_dim == 0
    num_k_tiles = k_head_dim // singleton_dim
    num_v_tiles = v_head_dim // singleton_dim

    # Add two extra dims for chunk-wise computation.
    chunk_dim = seq_len // chunk_size
    subchunk_dim = chunk_size // subchunk_size

    grid = (bs, num_heads, num_k_tiles, num_v_tiles, chunk_dim)

    # q/k/v jax.Arrays are kept in bf16 and converted later to fp32 in VMEM.
    log_alpha = log_alpha.astype(jnp.float32)
    initial_state = initial_state.astype(jnp.float32)

    # None is effectively 1, but the dim will be squeezed out.
    qk_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    qk_spec = _bs(
        lambda b, h, k, v, m: (b, lax.div(h, num_head_per_group), m, 0, k), qk_tiling
    )
    v_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    v_spec = _bs(lambda b, h, k, v, m: (b, h, m, 0, v), v_tiling)

    alpha_tiling = (None, None, None, subchunk_dim, subchunk_size)
    alpha_spec = _bs(lambda b, h, k, v, m: (b, h, m, 0, 0), alpha_tiling)

    # Initial hidden states.
    is_tiling = (None, None, singleton_dim, singleton_dim)
    is_spec = _bs(lambda b, h, k, v, m: (b, h, k, v), is_tiling)

    # Chunk-wise final states help pass states from the previous chunk to the next.
    fs_spec = is_spec

    fs_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32
    )

    # Pre-compute the cumulative sum of log_alpha.
    log_alpha = rearrange(
        log_alpha, "b h (nb ns bl) -> b h nb ns bl", nb=chunk_dim, ns=subchunk_dim
    )
    cum_log_alpha = jnp.cumsum(log_alpha, axis=-1)

    q = rearrange(q, "b h (nb bl) dk -> b h nb bl dk", bl=subchunk_size)
    k = rearrange(k, "b h (nb bl) dk -> b h nb bl dk", bl=subchunk_size)
    v = rearrange(v, "b h (nb bl) dv -> b h nb bl dv", bl=subchunk_size)

    gamma = cum_log_alpha[:, :, :, :, subchunk_size - 1 :]  # [b, h, nb, ns, 1]
    gamma_expanded = jnp.repeat(
        gamma, singleton_dim, axis=-1
    )  # [b, h, nb, ns, singleton_dim]
    gamma_tiling = (None, None, None, subchunk_dim, singleton_dim)
    gamma_spec = _bs(lambda b, h, k, v, m: (b, h, m, 0, 0), gamma_tiling)
    causal_mask_spec = _bs(lambda b, h, k, v, m: (0, 0), (subchunk_size, subchunk_size))
    causal_mask = jnp.tril(
        jnp.ones((subchunk_size, subchunk_size), dtype=jnp.float32), k=0
    )

    o_tiling = (None, None, None, subchunk_dim, subchunk_size, singleton_dim)
    o_spec = _bs(lambda b, h, k, v, m: (b, h, k, m, 0, v), o_tiling)
    o_shape = jax.ShapeDtypeStruct(
        shape=(
            bs,
            num_heads,
            num_k_tiles,
            chunk_dim * subchunk_dim,
            subchunk_size,
            v_head_dim,
        ),
        dtype=orig_dtype,
    )

    _, o = pl.pallas_call(
        _ssd_forward_kernel,
        in_specs=(
            qk_spec,
            qk_spec,
            v_spec,
            alpha_spec,
            is_spec,
            gamma_spec,
            causal_mask_spec,
        ),
        out_specs=(fs_spec, o_spec),
        out_shape=(fs_shape, o_shape),
        grid=grid,
        compiler_params=_tpu_compiler_params(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            )
        ),
        backend=_pallas_backend(),
        interpret=use_interpret_mode(),
    )(q, k, v, cum_log_alpha, initial_state, gamma_expanded, causal_mask)

    o = jnp.sum(o, axis=2)  # sum over dkn dim
    o = rearrange(o, "b h nb bl dv -> b h (nb bl) dv")

    # Store minimal residuals; chunk_states/gamma are recomputed in backward to
    # reduce forward HBM traffic.
    return o, (q, k, v, cum_log_alpha, initial_state)


@jax.jit
def _ssd_recompute_chunk_states(
    k: jax.Array,
    v: jax.Array,
    cum_log_alpha: jax.Array,
    gamma_expanded: jax.Array,
    initial_state: jax.Array,
) -> jax.Array:
    """Recomputes per-chunk start states for SSD backward."""
    singleton_dim, _, _ = _ssd_tiling_config()
    bs, num_heads, chunk_dim, subchunk_dim, subchunk_size = cum_log_alpha.shape
    k_dim, v_dim = k.shape[-1], v.shape[-1]
    num_qk_heads = k.shape[1]
    num_head_per_group = num_heads // num_qk_heads
    num_k_tiles, num_v_tiles = k_dim // singleton_dim, v_dim // singleton_dim

    grid = (bs, num_heads, num_k_tiles, num_v_tiles, chunk_dim)

    qk_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    qk_spec = _bs(
        lambda b, h, k_, v_, m: (
            b,
            lax.div(h, num_head_per_group),
            m,
            0,
            k_,
        ),
        qk_tiling,
    )
    v_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    v_spec = _bs(lambda b, h, k_, v_, m: (b, h, m, 0, v_), v_tiling)

    alpha_tiling = (None, None, None, subchunk_dim, subchunk_size)
    alpha_spec = _bs(lambda b, h, k_, v_, m: (b, h, m, 0, 0), alpha_tiling)
    gamma_tiling = (None, None, None, subchunk_dim, singleton_dim)
    gamma_spec = _bs(lambda b, h, k_, v_, m: (b, h, m, 0, 0), gamma_tiling)

    is_tiling = (None, None, singleton_dim, singleton_dim)
    is_spec = _bs(lambda b, h, k_, v_, m: (b, h, k_, v_), is_tiling)

    ch_tiling = (None, None, None, singleton_dim, singleton_dim)
    ch_spec = _bs(lambda b, h, k_, v_, m: (b, h, m, k_, v_), ch_tiling)
    fs_spec = is_spec

    ch_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, chunk_dim, k_dim, v_dim), dtype=jnp.float32
    )
    fs_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, k_dim, v_dim), dtype=jnp.float32
    )

    chunk_states, _ = pl.pallas_call(
        _ssd_chunk_states_kernel,
        in_specs=(qk_spec, v_spec, alpha_spec, is_spec, gamma_spec),
        out_specs=(ch_spec, fs_spec),
        out_shape=(ch_shape, fs_shape),
        grid=grid,
        compiler_params=_tpu_compiler_params(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            )
        ),
        backend=_pallas_backend(),
        interpret=use_interpret_mode(),
    )(k, v, cum_log_alpha, initial_state.astype(jnp.float32), gamma_expanded)
    return chunk_states


def _ssd_backward_kernel(
    q_ref: jax.Array,
    k_ref: jax.Array,
    v_ref: jax.Array,
    cum_log_alpha_ref: jax.Array,
    gamma_ref: jax.Array,
    ch_ref: jax.Array,
    causal_mask_ref: jax.Array,
    mutable_do_ref: jax.Array,
    mutable_dq_ref: jax.Array,
    mutable_dk_ref: jax.Array,
    mutable_dv_ref: jax.Array,
    mutable_dh_carry_ref: jax.Array,
):
    """Backward kernel for SSD.

    Args:
        q_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        k_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        v_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        cum_log_alpha_ref: jax.Array reference of shape [ns, bl]
        gamma_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        ch_ref: jax.Array reference of shape [ns, singleton_dim, singleton_dim]

    Output via mutable jax.Arrays:
        mutable_do_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        mutable_dq_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        mutable_dk_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        mutable_dv_ref: jax.Array reference of shape [ns, bl, singleton_dim]
        mutable_dh_carry_ref: jax.Array reference of shape [ns, singleton_dim, singleton_dim]

    Note: similar to final_state in the forward pass, dh_carry is used to pass gradients wrt.
    hidden states across different chunks. It will be initalized to zero at the last chunk.
    The final gradient wrt. hidden states will be returned as the gradient wrt. initial_state.
    """
    subchunk_dim, _ = cum_log_alpha_ref.shape[0], cum_log_alpha_ref.shape[1]
    causal_mask = causal_mask_ref[:]

    # In our grid definition, axis 4 is the chunk index.
    @pl.when(pl.program_id(axis=4) == 0)
    def init_carry():
        mutable_dh_carry_ref[:, :] = jnp.zeros_like(
            mutable_dh_carry_ref, dtype=jnp.float32
        )

    def _ssd_backward_dq_chunk_loop_body(t: int, h_carry: jax.Array):
        subchunk_idx = t
        h_block = h_carry  # final states from previous chunk
        k_block = k_ref[subchunk_idx, :].astype(jnp.float32)
        v_block = v_ref[subchunk_idx, :].astype(jnp.float32)
        do_block = mutable_do_ref[subchunk_idx, :].astype(jnp.float32)

        lambda_block = cum_log_alpha_ref[subchunk_idx, :]
        gamma_block = gamma_ref[subchunk_idx]

        lambda_block = jnp.expand_dims(lambda_block, axis=-1)  # [nb, 1]
        beta_block = gamma_block - lambda_block  # [nb, d_k]
        ssd_mask_block = lambda_block - jnp.transpose(lambda_block, [1, 0])
        ssd_mask_block = ssd_mask_block * causal_mask

        lambda_block = jnp.exp(lambda_block)
        beta_block = jnp.exp(beta_block)
        gamma_block = jnp.exp(gamma_block)
        ssd_mask_block = jnp.exp(ssd_mask_block)

        k_tilde_block = k_block * beta_block

        attn_mask = causal_mask * ssd_mask_block
        d_intra_att = _matmul_fp32(do_block, v_block.T) * attn_mask

        dq_tilde_block = _matmul_fp32(do_block, h_block.T)
        dq_block_1 = dq_tilde_block * lambda_block
        dq_block_2 = _matmul_fp32(d_intra_att, k_block)
        dq_block = dq_block_1 + dq_block_2
        mutable_dq_ref[subchunk_idx, :] = dq_block

        next_h_block = h_block * jnp.expand_dims(gamma_block, axis=-1) + _matmul_fp32(
            k_tilde_block.T, v_block
        )
        return next_h_block

    def _ssd_backward_dkv_chunk_loop_body(t: int, dh_carry: jax.Array):
        subchunk_idx = t
        dh_block = dh_carry
        q_block = q_ref[subchunk_idx, :].astype(jnp.float32)
        k_block = k_ref[subchunk_idx, :].astype(jnp.float32)
        v_block = v_ref[subchunk_idx, :].astype(jnp.float32)
        do_block = mutable_do_ref[subchunk_idx, :].astype(jnp.float32)
        lambda_block = cum_log_alpha_ref[subchunk_idx, :]
        gamma_block = gamma_ref[subchunk_idx]

        lambda_block = jnp.expand_dims(lambda_block, axis=-1)  # [nb, 1]
        beta_block = gamma_block - lambda_block  # [nb, d_k]
        ssd_mask_block = lambda_block - jnp.transpose(lambda_block, [1, 0])
        ssd_mask_block = ssd_mask_block * causal_mask

        lambda_block = jnp.exp(lambda_block)
        beta_block = jnp.exp(beta_block)
        gamma_block = jnp.exp(gamma_block)
        ssd_mask_block = jnp.exp(ssd_mask_block)

        q_tilde_block = q_block * lambda_block
        k_tilde_block = k_block * beta_block

        intra_att = _matmul_fp32(q_block, k_block.T)
        attn_mask = causal_mask * ssd_mask_block
        d_intra_att = _matmul_fp32(do_block, v_block.T) * attn_mask

        dk_block_1 = _matmul_fp32(d_intra_att.T, q_block)
        dk_tilde_block = _matmul_fp32(v_block, dh_block.T)
        dk_block_2 = dk_tilde_block * beta_block
        dk_block = dk_block_1 + dk_block_2
        mutable_dk_ref[subchunk_idx, :] = dk_block

        dv_block_1 = _matmul_fp32((intra_att * attn_mask).T, do_block)
        dv_block_2 = _matmul_fp32(k_tilde_block, dh_block)
        dv_block = dv_block_1 + dv_block_2
        mutable_dv_ref[subchunk_idx, :] = dv_block

        prev_dh_block = dh_block * jnp.expand_dims(gamma_block, axis=-1) + _matmul_fp32(
            q_tilde_block.T, do_block
        )
        return prev_dh_block

    h_carry = ch_ref[:, :]
    _ = lax.fori_loop(0, subchunk_dim, _ssd_backward_dq_chunk_loop_body, h_carry)

    def _ssd_backward_dkv_chunk_loop_body_reverse(t: int, dh_carry_ref: jax.Array):
        return _ssd_backward_dkv_chunk_loop_body(subchunk_dim - 1 - t, dh_carry_ref)

    dh_carry = mutable_dh_carry_ref[:, :]
    dinitial_state = lax.fori_loop(
        0, subchunk_dim, _ssd_backward_dkv_chunk_loop_body_reverse, dh_carry
    )
    mutable_dh_carry_ref[:, :] = dinitial_state


@jax.jit
def _ssd_backward(residuals: Tuple, do: jax.Array) -> Tuple:
    """Backward pass for SSD.

    Args:
        residuals: Tuple of jax.Arrays returned from the forward pass
        do: [bs, num_heads, seq_len, dv]

    Returns:
        dq: [bs, num_heads, seq_len, dk]
        dk: [bs, num_heads, seq_len, dk]
        dv: [bs, num_heads, seq_len, dv]
        dlog_alpha: [bs, num_heads, seq_len]
        dinitial_state: [bs, num_heads, dk, dv]
    """
    q, k, v, cum_log_alpha, initial_state = residuals
    orig_dtype = q.dtype

    singleton_dim, _, _ = _ssd_tiling_config()
    bs, num_heads, chunk_dim, subchunk_dim, subchunk_size = cum_log_alpha.shape
    k_dim, v_dim = q.shape[-1], v.shape[-1]
    num_k_tiles, num_v_tiles = k_dim // singleton_dim, v_dim // singleton_dim
    num_qk_heads = q.shape[1]
    num_head_per_group = num_heads // num_qk_heads

    gamma = cum_log_alpha[:, :, :, :, subchunk_size - 1 :]  # [b, h, nb, ns, 1]
    gamma_expanded = jnp.repeat(gamma, singleton_dim, axis=-1)
    chunk_states = _ssd_recompute_chunk_states(
        k, v, cum_log_alpha, gamma_expanded, initial_state
    )

    grid = (bs, num_heads, num_k_tiles, num_v_tiles, chunk_dim)

    qk_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    qk_spec = _bs(
        lambda b, h, k, v, m: (
            b,
            lax.div(h, num_head_per_group),
            chunk_dim - 1 - m,
            0,
            k,
        ),
        qk_tiling,
    )
    v_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    v_spec = _bs(lambda b, h, k, v, m: (b, h, chunk_dim - 1 - m, 0, v), v_tiling)

    alpha_tiling = (None, None, None, subchunk_dim, subchunk_size)
    alpha_spec = _bs(
        lambda b, h, k, v, m: (b, h, chunk_dim - 1 - m, 0, 0), alpha_tiling
    )
    gamma_tiling = (None, None, None, subchunk_dim, singleton_dim)
    gamma_spec = _bs(
        lambda b, h, k, v, m: (b, h, chunk_dim - 1 - m, 0, 0), gamma_tiling
    )

    ch_tiling = (None, None, None, singleton_dim, singleton_dim)
    ch_spec = _bs(lambda b, h, k, v, m: (b, h, chunk_dim - 1 - m, k, v), ch_tiling)

    do_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    do_spec = _bs(lambda b, h, k, v, m: (b, h, chunk_dim - 1 - m, 0, v), do_tiling)
    causal_mask_spec = _bs(lambda b, h, k, v, m: (0, 0), (subchunk_size, subchunk_size))
    causal_mask = jnp.tril(
        jnp.ones((subchunk_size, subchunk_size), dtype=jnp.float32), k=0
    )

    dqk_tiling = (None, None, None, None, subchunk_dim, subchunk_size, singleton_dim)
    dqk_spec = _bs(
        lambda b, h, k, v, m: (
            b,
            lax.div(h, num_head_per_group),
            lax.rem(h, num_head_per_group),
            v,
            chunk_dim - 1 - m,
            0,
            k,
        ),
        dqk_tiling,
    )
    dqk_shape = jax.ShapeDtypeStruct(
        shape=(
            bs,
            num_qk_heads,
            num_head_per_group,
            num_v_tiles,
            chunk_dim * subchunk_dim,
            subchunk_size,
            k_dim,
        ),
        dtype=jnp.float32,
    )

    dv_tiling = (None, None, None, subchunk_dim, subchunk_size, singleton_dim)
    dv_spec = _bs(lambda b, h, k, v, m: (b, h, k, chunk_dim - 1 - m, 0, v), dv_tiling)
    dv_shape = jax.ShapeDtypeStruct(
        shape=(
            bs,
            num_heads,
            num_k_tiles,
            chunk_dim * subchunk_dim,
            subchunk_size,
            v_dim,
        ),
        dtype=jnp.float32,
    )

    dh_carry_tiling = (None, None, singleton_dim, singleton_dim)
    dh_carry_spec = _bs(lambda b, h, k, v, m: (b, h, k, v), dh_carry_tiling)
    dh_carry_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, k_dim, v_dim), dtype=jnp.float32
    )

    do = rearrange(do, "b h (nb bl) dv -> b h nb bl dv", bl=subchunk_size)

    dq, dk, dv, dinitial_state = pl.pallas_call(
        _ssd_backward_kernel,
        in_specs=(
            qk_spec,
            qk_spec,
            v_spec,
            alpha_spec,
            gamma_spec,
            ch_spec,
            causal_mask_spec,
            do_spec,
        ),
        out_specs=(dqk_spec, dqk_spec, dv_spec, dh_carry_spec),
        out_shape=(dqk_shape, dqk_shape, dv_shape, dh_carry_shape),
        grid=grid,
        compiler_params=_tpu_compiler_params(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            )
        ),
        backend=_pallas_backend(),
        interpret=use_interpret_mode(),
    )(q, k, v, cum_log_alpha, gamma_expanded, chunk_states, causal_mask, do)

    # Sum over dvn dim.
    dq = jnp.sum(dq, axis=3)
    dk = jnp.sum(dk, axis=3)
    dq = rearrange(dq, "b ng nhg nb bl dk -> b ng nhg (nb bl) dk")
    dk = rearrange(dk, "b ng nhg nb bl dk -> b ng nhg (nb bl) dk")

    # Compute dlog_alpha via `q * dq - k * dk`.
    dq_ = rearrange(dq, "b ng nhg l dk -> b (ng nhg) l dk")
    dk_ = rearrange(dk, "b ng nhg l dk -> b (ng nhg) l dk")

    q_ = repeat(q, "b ng nb bl dk -> b (ng nhg) nb bl dk", nhg=num_head_per_group)
    k_ = repeat(k, "b ng nb bl dk -> b (ng nhg) nb bl dk", nhg=num_head_per_group)
    q_ = rearrange(q_, "b h nb bl dk -> b h (nb bl) dk")
    k_ = rearrange(k_, "b h nb bl dk -> b h (nb bl) dk")

    dlog_alpha_ = jnp.sum(dq_ * q_ - dk_ * k_, axis=-1)
    dlog_alpha = lax.cumsum(dlog_alpha_, axis=2, reverse=True)

    # Sum over dkn dim.
    dv = jnp.sum(dv, axis=2)
    dv = rearrange(dv, "b h nb bl dv -> b h (nb bl) dv")

    # Sum over nhg dim
    dq = jnp.sum(dq, axis=2)
    dk = jnp.sum(dk, axis=2)
    # `dlog_alpha` is always in float32, `dv` is also in float32.
    dq, dk = dq.astype(orig_dtype), dk.astype(orig_dtype)

    dinitial_state = dinitial_state.astype(orig_dtype)
    return dq, dk, dv, dlog_alpha, dinitial_state


def _ssd_backward_impl(
    do: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    initial_state: jax.Array,
) -> Tuple:
    """Backward pass body for SSD (no @jax.jit — called from custom_partitioning).

    Takes the *original* input shapes and redoes the rearrangements that were
    done in the forward so that ``_ssd_backward`` receives the residual shapes
    it expects.

    Args:
        do: [bs, num_heads, seq_len, dv]
        q, k: [bs, num_groups, seq_len, dk]
        v: [bs, num_heads, seq_len, dv]
        log_alpha: [bs, num_heads, seq_len]
        initial_state: [bs, num_heads, dk, dv]

    Returns:
        (dq, dk, dv, dlog_alpha, dinitial_state)
    """
    singleton_dim, chunk_size, subchunk_size = _ssd_tiling_config()
    seq_len = q.shape[2]
    chunk_dim = seq_len // chunk_size
    subchunk_dim = chunk_size // subchunk_size

    # Redo the rearrangements that _ssd_forward performed before saving residuals.
    log_alpha_r = rearrange(
        log_alpha, "b h (nb ns bl) -> b h nb ns bl", nb=chunk_dim, ns=subchunk_dim
    )
    cum_log_alpha = jnp.cumsum(log_alpha_r, axis=-1)
    q_r = rearrange(q, "b h (nb bl) dk -> b h nb bl dk", bl=subchunk_size)
    k_r = rearrange(k, "b h (nb bl) dk -> b h nb bl dk", bl=subchunk_size)
    v_r = rearrange(v, "b h (nb bl) dv -> b h nb bl dv", bl=subchunk_size)

    residuals = (q_r, k_r, v_r, cum_log_alpha, initial_state)
    return _ssd_backward(residuals, do)


def _make_ssd():
    """Build a differentiable, SPMD-partitionable SSD op.

    Returns a function ``f(q, k, v, log_alpha, h0) -> o`` that:
      - Uses ``custom_partitioning`` on the forward and backward Pallas kernels
        so that GSPMD runs them per-shard without inserting all-gathers.
      - Uses ``custom_vjp`` so that ``jax.grad`` works through the op.
    """
    from jax.experimental.custom_partitioning import custom_partitioning

    # -- Forward kernel with custom_partitioning --------------------------
    @custom_partitioning
    def _fwd(q, k, v, log_alpha, h0):
        return _ssd_forward_impl(q, k, v, log_alpha, h0)

    def _fwd_partition(mesh, arg_shapes, result_shape):
        names = ("q", "k", "v", "log_alpha", "h0")
        for shape, name in zip(arg_shapes, names):
            _validate_ssd_sharding(shape.sharding, name)

        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(q, k, v, log_alpha, h0):
            return _ssd_forward_impl(q, k, v, log_alpha, h0)

        return mesh, lower_fn, result_shardings, arg_shardings

    _fwd.def_partition(
        partition=_fwd_partition,
        # q(B,G,L,Dk) k(B,G,L,Dk) v(B,H,L,Dv) log_alpha(B,H,L) h0(B,H,Dk,Dv)
        # -> o(B,H,L,Dv)
        # batch and heads/groups are shardable; seq, dk, dv must be replicated.
        sharding_rule=(
            "batch groups seq dk, batch groups seq dk, "
            "batch heads seq dv, batch heads seq, batch heads dk dv "
            "-> batch heads seq dv"
        ),
        need_replication_factors=("seq", "dk", "dv"),
    )

    # -- Backward kernel with custom_partitioning -------------------------
    @custom_partitioning
    def _bwd(do, q, k, v, log_alpha, h0):
        return _ssd_backward_impl(do, q, k, v, log_alpha, h0)

    def _bwd_partition(mesh, arg_shapes, result_shape):
        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(do, q, k, v, log_alpha, h0):
            return _ssd_backward_impl(do, q, k, v, log_alpha, h0)

        return mesh, lower_fn, result_shardings, arg_shardings

    _bwd.def_partition(
        partition=_bwd_partition,
        # do(B,H,L,Dv) q(B,G,L,Dk) k(B,G,L,Dk) v(B,H,L,Dv)
        #   log_alpha(B,H,L) h0(B,H,Dk,Dv)
        # -> dq(B,G,L,Dk) dk(B,G,L,Dk) dv(B,H,L,Dv)
        #    dlog_alpha(B,H,L) dh0(B,H,Dk,Dv)
        sharding_rule=(
            "batch heads seq dv, batch groups seq dk, batch groups seq dk, "
            "batch heads seq dv, batch heads seq, batch heads dk dv "
            "-> batch groups seq dk, batch groups seq dk, "
            "batch heads seq dv, batch heads seq, batch heads dk dv"
        ),
        need_replication_factors=("seq", "dk", "dv"),
    )

    # -- Differentiable wrapper using custom_vjp --------------------------
    @jax.custom_vjp
    def _op(q, k, v, log_alpha, h0):
        return _fwd(q, k, v, log_alpha, h0)

    def _op_fwd(q, k, v, log_alpha, h0):
        o = _fwd(q, k, v, log_alpha, h0)
        return o, (q, k, v, log_alpha, h0)

    def _op_bwd(res, do):
        q, k, v, log_alpha, h0 = res
        return _bwd(do, q, k, v, log_alpha, h0)

    _op.defvjp(_op_fwd, _op_bwd)
    return _op


@jax.named_call  # `named_call` ensures the name is used in tracing, which is useful for profiling.
def ssd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    h0: Optional[jax.Array] = None,
) -> jax.Array:
    """Differentiable function that computes the output of SSD.

    When inputs carry ``NamedSharding`` (e.g. batch/heads-sharded across a
    device mesh), the forward and backward Pallas kernels are automatically
    run per-shard via ``custom_partitioning`` — no ``shard_map`` wrapping is
    needed at the call site.

    Args:
        q: [batch_size, num_groups, seq_len, dk]
        k: [batch_size, num_groups, seq_len, dk]
        v: [batch_size, num_groups, seq_len, dv]
        log_alpha: [batch_size, num_heads, seq_len]
        h0: [batch_size, num_heads, dk, dv]

    Returns:
        output: [batch_size, num_heads, seq_len, dv]

    The notion of groups is similar to the group in multi-group attention (or more preciesly
    multi-value attention) -- one group of q/k corresponds to multiple v heads.
    """
    _validate_ssd_runtime_inputs(q, k, v, log_alpha, h0)

    bs, ng, _, dk = q.shape
    _, nh, _, dv = v.shape
    backend = jax.default_backend()
    pallas_backend = _pallas_backend()
    if backend == "cpu":
        raise RuntimeError("ssd requires an accelerator backend.")

    if h0 is None:
        h0 = jnp.zeros((bs, nh, dk, dv), dtype=jnp.float32)

    if backend not in ("tpu", "gpu"):
        output, _ = ssd_linear_scan(q, k, v, log_alpha, h0)
        return output

    if backend == "gpu" and not _gpu_supports_ssd_pallas_for_shape(
        seq_len=q.shape[2],
        num_groups=ng,
        num_heads=nh,
        dk=dk,
        dv=dv,
        pallas_backend=pallas_backend,
    ):
        output, _ = ssd_linear_scan(q, k, v, log_alpha, h0)
        return output
    if backend == "gpu":
        failure_key = _ssd_pallas_failure_key(
            pallas_backend=pallas_backend,
            q=q,
            k=k,
            v=v,
            log_alpha=log_alpha,
            h0=h0,
        )
        if failure_key in _FAILED_SSD_PALLAS_CONFIGS:
            output, _ = ssd_linear_scan(q, k, v, log_alpha, h0)
            return output

    # Build a custom_partitioning-aware SSD op.
    # When inputs are NamedSharded, GSPMD will call our partition()
    # callback and run the Pallas kernel per-shard without all-gathers.
    _ssd_op = _make_ssd()

    try:
        return _ssd_op(q, k, v, log_alpha, h0)
    except (
        AssertionError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as err:
        if backend != "gpu":
            raise
        _FAILED_SSD_PALLAS_CONFIGS.add(failure_key)
        warnings.warn(
            f"Falling back to ssd_linear_scan on GPU because Pallas/Triton kernel failed: {err}",
            RuntimeWarning,
            stacklevel=2,
        )
        output, _ = ssd_linear_scan(q, k, v, log_alpha, h0)
        return output


def ssd_linear_scan(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    h0: Union[jax.Array, None] = None,
) -> jax.Array:
    """LinearScan based reference implementations for testing SSD kernels.

    Args:
        q, k: [batch_size, num_groups, seq_len, dk]
        k: [batch_size, num_groups, seqlen, dk]
        v: [batch_size, num_heads, seq_len, dv]
        log_alpha: [batch_size, num_heads, seq_len]
        h0: [batch_size, num_heads, dk, dv] or None

     Returns:
        output: [batch_size, num_heads, seq_len, dv]
        hidden_state: [batch_size, num_heads, dk, dv]
    """
    bs, ng, _, dk = q.shape
    bs, nh, _, dv = v.shape
    assert nh % ng == 0

    # The linearscan kernel assumes that nh == ng, so we need to repeat q/k.
    num_head_per_group = nh // ng
    q = repeat(q, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)
    k = repeat(k, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)

    # ITt's more convenient for vmap to have internal states of size [dv, dk]
    if h0 is None:
        h0 = jnp.zeros((bs, nh, dv, dk), dtype=jnp.float32)
    else:
        # to be consistent with pallas api, h0 is in dk x dv as input
        h0 = rearrange(h0, "b h dk dv -> b h dv dk")

    # All inputs are upcasted to float32, making this function a good reference funciton to
    # test pallas kernel's numerical precision in the case of bf16 inputs.
    dtype = q.dtype
    if dtype == jnp.bfloat16:
        q, k, v, h0 = map(lambda x: x.astype(jnp.float32), (q, k, v, h0))

    def scan_body_fn(h_prev, current_inputs):
        acc_dtype = h_prev.dtype
        q_t, k_t, v_t, log_a_t = current_inputs
        a_t = jnp.exp(log_a_t).astype(acc_dtype)
        h_next = a_t * h_prev + jnp.einsum(
            "i,j->ij", v_t, k_t, preferred_element_type=jnp.float32
        )
        o_t = jnp.einsum("ij,j->i", h_next, q_t, preferred_element_type=jnp.float32)
        return h_next, o_t.astype(q_t.dtype)

    def single_head_scan(q_head, k_head, v_head, alpha_head, h0_head):
        return jax.lax.scan(scan_body_fn, h0_head, (q_head, k_head, v_head, alpha_head))

    multi_head_scan = jax.vmap(
        single_head_scan, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0)
    )
    batched_scan = jax.vmap(multi_head_scan, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0))

    # Note: if dk > 128 (e.g., 256), somehow jax jvp would fail; a work-around
    # is to add another dim to ensure that minor dk is always 128.
    q = rearrange(q, "b h l (dkn dks) -> dkn b h l dks", dks=128)
    k = rearrange(k, "b h l (dkn dks) -> dkn b h l dks", dks=128)
    h0 = rearrange(h0, "b h dv (dkn dks) -> dkn b h dv dks", dks=128)

    batched_scan = jax.vmap(
        batched_scan, in_axes=(0, 0, None, None, 0), out_axes=(0, 0)
    )
    final_state, output = batched_scan(q, k, v, log_alpha, h0)
    final_state = rearrange(final_state, "dkn b h dv dks -> b h dv (dkn dks)")
    output = jnp.sum(output, axis=0)

    final_state = rearrange(final_state, "b h dv dk -> b h dk dv")

    if dtype == jnp.bfloat16:
        output = output.astype(jnp.bfloat16)
        final_state = final_state.astype(jnp.bfloat16)

    return output, final_state


def ssd_linear_scan_w_hidden_states(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    h0: Union[jax.Array, None] = None,
) -> jax.Array:
    """LinearScan based reference implementations for testing SSD kernels.

    This version additionally returns the hidden states of all tokens.

    Args:
        q: [batch_size, num_groups, seqlen, dk]
        k: [batch_size, num_groups, seqlen, dk]
        v: [batch_size, num_heads, seqlen, dv]
        log_alpha: [batch_size, num_heads, seqlen]
        h0: [batch_size, num_heads, dk, dv] or None

     Returns:
        output: [batch_size, num_heads, seq_len, dv]
        hidden_states: [batch_size, num_heads, seq_len, dk, dv]
    """
    bs, ng, _, dk = q.shape
    bs, nh, _, dv = v.shape
    assert nh % ng == 0

    num_head_per_group = nh // ng
    q = repeat(q, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)
    k = repeat(k, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)

    if h0 is None:
        h0 = jnp.zeros((bs, nh, dv, dk), dtype=jnp.float32)
    else:
        # to be consistent with pallas api, h0 is in dk x dv as input
        h0 = rearrange(h0, "b h dk dv -> b h dv dk")

    dtype = q.dtype
    if dtype == jnp.bfloat16:
        q, k, v, h0 = map(lambda x: x.astype(jnp.float32), (q, k, v, h0))

    def scan_body_fn(h_prev, current_inputs):
        acc_dtype = h_prev.dtype
        k_t, v_t, log_a_t = current_inputs
        a_t = jnp.exp(log_a_t).astype(acc_dtype)
        h_next = a_t * h_prev + jnp.einsum("i,j->ij", v_t, k_t)
        return h_next, h_next

    def single_head_scan(k_head, v_head, alpha_head, h0_head):
        return jax.lax.scan(scan_body_fn, h0_head, (k_head, v_head, alpha_head))

    multi_head_scan = jax.vmap(single_head_scan, in_axes=(0, 0, 0, 0), out_axes=(0, 0))
    batched_scan = jax.vmap(multi_head_scan, in_axes=(0, 0, 0, 0), out_axes=(0, 0))

    k = rearrange(k, "b h l (dkn dks) -> dkn b h l dks", dks=128)
    h0 = rearrange(h0, "b h dv (dkn dks) -> dkn b h dv dks", dks=128)

    batched_scan = jax.vmap(batched_scan, in_axes=(0, None, None, 0), out_axes=(0, 0))
    final_state, hidden_states = batched_scan(k, v, log_alpha, h0)
    assert final_state is not None

    hidden_states = rearrange(hidden_states, "dkn b h l dv dks -> b h l (dkn dks) dv")
    output = jnp.einsum(
        "b h l s, b h l s d -> b h l d",
        q,
        hidden_states,
        preferred_element_type=jnp.float32,
    )

    if dtype == jnp.bfloat16:
        output = output.astype(jnp.bfloat16)
    return output, hidden_states


def ssd_linear_scan_w_timestep(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    timestep: jax.Array,
    h0=None,
) -> jax.Array:
    """LinearScan that takes timestep as input and masks useless k/v based on timestep.

    This function is used during inference where decoding might start from different timesteps.

    Args:
        q: [batch_size, num_groups, seqlen, dk]
        k: [batch_size, num_groups, seqlen, dk]
        v: [batch_size, num_heads, seqlen, dv]
        log_alpha: [batch_size, num_heads, seqlen]
        h0: [batch_size, num_heads, dk, dv] or None
        timestep: [batch_size, seqlen] or None

     Returns:
        output: [batch_size, num_heads, seq_len, dv]
        hidden_states: [batch_size, num_heads, dk, dv]

    """
    bs, ng, l, dk = q.shape
    bs, nh, l, dv = v.shape
    assert nh % ng == 0

    num_head_per_group = nh // ng
    q = repeat(q, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)
    k = repeat(k, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)

    timestep_mask = jnp.arange(l)[None, :] >= timestep[:, None]
    k = jnp.where(timestep_mask[:, None, :, None], 0.0, k)
    v = jnp.where(timestep_mask[:, None, :, None], 0.0, v)
    log_alpha = jnp.where(timestep_mask[:, None, :], 0.0, log_alpha)

    if h0 is None:
        h0 = jnp.zeros((bs, nh, dv, dk), dtype=jnp.float32)
    else:
        # to be consistent with pallas api, h0 is in dk x dv as input
        h0 = rearrange(h0, "b h dk dv -> b h dv dk")

    dtype = q.dtype
    if dtype == jnp.bfloat16:
        q, k, v, h0 = map(lambda x: x.astype(jnp.float32), (q, k, v, h0))

    def scan_body_fn(h_prev, current_inputs):
        acc_dtype = h_prev.dtype
        q_t, k_t, v_t, log_a_t = current_inputs
        a_t = jnp.exp(log_a_t).astype(acc_dtype)
        h_next = a_t * h_prev + jnp.einsum(
            "i,j->ij", v_t, k_t, preferred_element_type=jnp.float32
        )
        o_t = jnp.einsum("ij,j->i", h_next, q_t, preferred_element_type=jnp.float32)
        return h_next, o_t.astype(q_t.dtype)

    def single_head_scan(q_head, k_head, v_head, alpha_head, h0_head):
        return jax.lax.scan(scan_body_fn, h0_head, (q_head, k_head, v_head, alpha_head))

    multi_head_scan = jax.vmap(
        single_head_scan, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0)
    )
    batched_scan = jax.vmap(multi_head_scan, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0))

    q = rearrange(q, "b h l (dkn dks) -> dkn b h l dks", dks=128)
    k = rearrange(k, "b h l (dkn dks) -> dkn b h l dks", dks=128)
    h0 = rearrange(h0, "b h dv (dkn dks) -> dkn b h dv dks", dks=128)

    batched_scan = jax.vmap(
        batched_scan, in_axes=(0, 0, None, None, 0), out_axes=(0, 0)
    )
    final_state, output = batched_scan(q, k, v, log_alpha, h0)
    final_state = rearrange(final_state, "dkn b h dv dks -> b h dv (dkn dks)")
    output = jnp.sum(output, axis=0)

    final_state = rearrange(final_state, "b h dv dk -> b h dk dv")

    if dtype == jnp.bfloat16:
        output = output.astype(jnp.bfloat16)

    return output, final_state
