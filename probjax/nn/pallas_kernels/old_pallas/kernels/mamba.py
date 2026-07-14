# Some of the code in this file is adapted from:
#
# Apple/axlearn
# https://github.com/apple/axlearn/blob/main/axlearn/common/ssm_kernels/mamba_kernels.py
#
#
# google-deepmind/recurrentgemma
# Copyright 2024 The recurrentgemma authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Pallas kernels for use with Mamba models."""

import functools
import os
from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from ..kernel_utils import (
    def_partition_compat,
    get_dot_precision,
    pallas_call_compat,
    use_interpret_mode,
)


def _validate_mamba_sharding(sharding, name: str):
    """Validate that *sharding* (a NamedSharding) does not shard unsupported dims.

    Only batch (dim 0) sharding is supported for Mamba.  Sharding on the
    sequence (dim 1) or inner_dim (dim 2) is rejected with an informative error.
    """
    spec = getattr(sharding, "spec", None)
    if spec is None:
        return
    for dim_idx, axis in enumerate(spec):
        if axis is None:
            continue
        if dim_idx == 1:
            raise ValueError(
                f"Pallas Mamba kernel does not support sharding on the "
                f"sequence dimension (dim 1) of '{name}'. "
                f"Got PartitionSpec{tuple(spec)} which shards dim 1 over "
                f"mesh axis '{axis}'. Only batch (dim 0) sharding is supported."
            )
        if dim_idx == 2:
            raise ValueError(
                f"Pallas Mamba kernel does not support sharding on the "
                f"inner_dim dimension (dim 2) of '{name}'. "
                f"Got PartitionSpec{tuple(spec)} which shards dim 2 over "
                f"mesh axis '{axis}'. Only batch (dim 0) sharding is supported."
            )


def _bs(index_map, block_shape):
    """Compatibility wrapper for BlockSpec(index_map, block_shape) call sites."""
    return pl.BlockSpec(block_shape=block_shape, index_map=index_map)


def _validate_mamba_runtime_inputs(
    x: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    delta: jax.Array,
    d: jax.Array,
    *,
    seq_tile_size: int,
    dim_tile_size: int,
) -> None:
    if seq_tile_size <= 0 or seq_tile_size % 8 != 0:
        raise ValueError("`seq_tile_size` must be a positive multiple of 8.")
    if dim_tile_size <= 0 or dim_tile_size % 128 != 0:
        raise ValueError("`dim_tile_size` must be a positive multiple of 128.")

    if x.ndim != 3:
        raise ValueError(f"`x` must be rank-3 [B, L, D], got shape {x.shape}.")
    if a.ndim != 2:
        raise ValueError(f"`a` must be rank-2 [S, D], got shape {a.shape}.")
    if b.ndim != 3:
        raise ValueError(f"`b` must be rank-3 [B, L, S], got shape {b.shape}.")
    if c.ndim != 3:
        raise ValueError(f"`c` must be rank-3 [B, L, S], got shape {c.shape}.")
    if delta.ndim != 3:
        raise ValueError(f"`delta` must be rank-3 [B, L, D], got shape {delta.shape}.")
    if d.ndim != 2:
        raise ValueError(f"`d` must be rank-2 [1, D], got shape {d.shape}.")

    batch_size, seq_len, inner_dim = x.shape
    state_dim, a_inner_dim = a.shape

    if b.shape != (batch_size, seq_len, state_dim):
        raise ValueError(
            f"`b` shape mismatch: expected {(batch_size, seq_len, state_dim)}, got {b.shape}."
        )
    if c.shape != (batch_size, seq_len, state_dim):
        raise ValueError(
            f"`c` shape mismatch: expected {(batch_size, seq_len, state_dim)}, got {c.shape}."
        )
    if delta.shape != (batch_size, seq_len, inner_dim):
        raise ValueError(
            f"`delta` shape mismatch: expected {(batch_size, seq_len, inner_dim)}, got {delta.shape}."
        )
    if a_inner_dim != inner_dim:
        raise ValueError(
            f"`a` inner dimension mismatch: expected {inner_dim}, got {a_inner_dim}."
        )
    if d.shape != (1, inner_dim):
        raise ValueError(
            f"`d` shape mismatch: expected {(1, inner_dim)}, got {d.shape}."
        )


class MambaArgumentBlockSpecs(NamedTuple):
    """A NamedTuple for storing the pl.BlockSpecs associated with arguments to the Mamba
    Pallas kernels."""

    # A BlockSpec for arguments with shape [batch_size, seq_len, inner_dim].
    x_spec: pl.BlockSpec
    # A BlockSpec for arguments with shape [state_dim, inner_dim].
    a_spec: pl.BlockSpec
    # A BlockSpec for arguments with shape [batch_size, seq_len, state_dim].
    b_spec: pl.BlockSpec
    # A BlockSpec for arguments with shape [1, inner_dim].
    d_spec: pl.BlockSpec
    # A BlockSpec for arguments with shape [batch_size, state_dim, inner_dim].
    carry_spec: pl.BlockSpec


def _parameter_blockspecs(
    state_dim: int, seq_tile_size: int, dim_tile_size: int
) -> MambaArgumentBlockSpecs:
    """Returns `pl.BlockSpec`s for the parameters of the forward and backward Mamba kernels.

    Args:
        state_dim: The Mamba state_dim.
        seq_tile_size: The size of the tiles into which the sequence dimension will be divided.
        dim_tile_size: The size of the tiles into which the "inner" dimension will be divided.

    Returns:
        An instance of MambaArgumentBlockSpecs.
    """

    # Note: `None` is equivalent to 1, but the dimension is squeezed away in the kernel.
    x_tiling = (None, seq_tile_size, dim_tile_size)
    x_spec = _bs(lambda b, d, s: (b, s, d), x_tiling)

    # Note: we do not tile over state_dim.
    a_tiling = (state_dim, dim_tile_size)
    a_spec = _bs(lambda b, d, s: (0, d), a_tiling)

    b_tiling = (None, seq_tile_size, state_dim)
    b_spec = _bs(lambda b, d, s: (b, s, 0), b_tiling)

    d_tiling = (1, dim_tile_size)
    d_spec = _bs(lambda b, d, s: (0, d), d_tiling)

    carry_tiling = (None, state_dim, dim_tile_size)
    carry_spec = _bs(lambda b, d, s: (b, 0, 0), carry_tiling)

    return MambaArgumentBlockSpecs(
        x_spec=x_spec,
        a_spec=a_spec,
        b_spec=b_spec,
        d_spec=d_spec,
        carry_spec=carry_spec,
    )


def _in_kernel_dtype() -> jnp.dtype:
    """Returns the dtype to use within kernel computations."""
    return jnp.float32


def _matmul_precision(dtype: jnp.dtype):
    return get_dot_precision(jax.default_backend(), dtype)


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


def _unsafe_enable_mosaic_gpu() -> bool:
    """Explicit opt-in for the unstable Mosaic GPU path for Mamba."""
    val = os.environ.get("PROBJAX_PALLAS_UNSAFE_ENABLE_MOSAIC_GPU", "0")
    return val.strip().lower() in ("1", "true", "yes", "on")


def _pallas_backend(prefer_mosaic_gpu: bool | None = None) -> str | None:
    """Selects the explicit Pallas backend for the current JAX platform."""
    if prefer_mosaic_gpu is None:
        prefer_mosaic_gpu = _prefer_mosaic_gpu()
    backend = jax.default_backend()
    if backend == "gpu":
        if (
            prefer_mosaic_gpu
            and _unsafe_enable_mosaic_gpu()
            and _is_hopper_or_newer_gpu()
        ):
            return "mosaic_gpu"
        return "triton"
    if backend == "tpu":
        return "mosaic_tpu"
    return None


@functools.lru_cache(maxsize=128)
def _gpu_supports_mamba_pallas_for_shape(
    *,
    seq_len: int,
    inner_dim: int,
    state_dim: int,
    seq_tile_size: int,
    dim_tile_size: int,
    pallas_backend: str | None,
) -> bool:
    """Cheap/static capability check for Mamba Pallas on GPU."""
    if jax.default_backend() != "gpu":
        return False
    if use_interpret_mode():
        return True
    if pallas_backend not in ("triton", "mosaic_gpu"):
        return False
    if pallas_backend == "mosaic_gpu" and not _is_hopper_or_newer_gpu():
        return False
    if pallas_backend == "triton" and state_dim < 16:
        return False
    if seq_len <= 0 or inner_dim <= 0 or state_dim <= 0:
        return False
    if seq_tile_size <= 0 or seq_tile_size % 8 != 0:
        return False
    if dim_tile_size <= 0 or dim_tile_size % 128 != 0:
        return False
    return True


# pylint: disable=invalid-name


def _update_carry(
    t: int,
    carry: jax.Array,
    *,
    dtype: jnp.dtype,
    x_ref: jax.Array,
    a: jax.Array,
    b_ref: jax.Array,
    delta_ref: jax.Array,
):
    """Computes h_{t+1} = a_bar_t * h_t + b_bar_t * x_t, where:
            a_bar_t = exp(delta_t * a), b_bar_t = delta_t * b_t.

    Args:
        t: the current time-step.
        mutable_carry_ref: A [state_dim, dim_tile_size] jax.Array reference to h_t,
            modified by the function.
        x_ref: A [seq_len, dim_tile_size] jax.Array reference.
        a: A [state_dim, dim_tile_size] jax.Array.
        b_ref: A [seq_len, state_dim] jax.Array reference.
        delta_ref: A [seq_len, dim_tile_size] jax.Array reference.
    """
    delta = jnp.expand_dims(delta_ref[t].astype(dtype), axis=0)  # [1, inner_dim]
    a_bar = jnp.exp(delta * a)  # [state_sim, inner_dim]
    b = jnp.expand_dims(b_ref[t].astype(dtype), axis=-1)  # [state_dim, 1]
    b_bar = delta * b * jnp.expand_dims(x_ref[t].astype(dtype), axis=0)
    # `carry` holds h_t.
    a_bar *= carry
    # Put h_{t+1} in the a_bar register.
    a_bar += b_bar
    # Return h_{t+1}.
    return a_bar


def _forward_kernel_boundary_hs(
    x_ref: jax.Array,
    a_ref: jax.Array,
    b_ref: jax.Array,
    delta_ref: jax.Array,
    mutable_carry_ref: jax.Array,
    mutable_boundary_hs_ref: jax.Array,
):
    """Computes the Mamba forward states, but not outputs.

    Args:
        x_ref: A [seq_len, dim_tile_size] jax.Array reference.
        a_ref: A [state_dim, dim_tile_size] jax.Array reference.
        b_ref: A [seq_len, state_dim] jax.Array reference.
        delta_ref: A [seq_len, dim_tile_size] jax.Array reference.
        mutable_carry_ref: A [state_dim, dim_tile_size] jax.Array reference to h_t,
            modified by the function.
        mutable_boundary_hs_ref: A [state_dim, dim_tile_size] jax.Array reference,
            modified by the function.
    """
    dtype = _in_kernel_dtype()
    a = a_ref[:].astype(dtype)
    seq_len = x_ref.shape[0]

    # Zero h_carry_ref, which holds h_t, only when in the first tile along the sequence dimension.
    @pl.when(pl.program_id(axis=2) == 0)
    def zero_carry():
        mutable_carry_ref[:] = jnp.zeros_like(mutable_carry_ref)

    h_carry = mutable_carry_ref[:]
    h_T = lax.fori_loop(  # fills h_ref
        0,
        seq_len,
        functools.partial(
            _update_carry,
            dtype=dtype,
            x_ref=x_ref,
            a=a,
            b_ref=b_ref,
            delta_ref=delta_ref,
        ),
        h_carry,
    )
    # Store carry as a reference.
    mutable_carry_ref[:] = h_T
    # Store final state in boundary_hs_ref, so we can use it later for gradients.
    mutable_boundary_hs_ref[:] = h_T


def _backward_kernel(
    dy_ref: jax.Array,
    x_ref: jax.Array,
    a_ref: jax.Array,
    b_ref: jax.Array,
    c_ref: jax.Array,
    delta_ref: jax.Array,
    d_ref: jax.Array,
    boundary_hs_ref: jax.Array,
    mutable_dx_ref: jax.Array,
    mutable_da_ref: jax.Array,
    mutable_db_ref: jax.Array,
    mutable_dc_ref: jax.Array,
    mutable_ddelta_ref: jax.Array,
    mutable_dd_ref: jax.Array,
    mutable_dcarry_ref: jax.Array,
    mutable_forward_hs_ref: jax.Array,
):
    """Computes the Mamba backward pass.

    Args:
        dy_ref: A [seq_len, dim_tile_size] jax.Array reference.
        x_ref: A [seq_len, dim_tile_size] jax.Array reference.
        a_ref: A [state_dim, dim_tile_size] jax.Array reference.
        b_ref: A [seq_len, state_dim] jax.Array reference.
        c_ref: A [seq_len, state_dim] jax.Array reference.
        delta_ref: A [seq_len, dim_tile_size] jax.Array reference.
        d_ref: A [1, dim_tile_size] jax.Array reference.
        boundary_hs_ref: a [state_dim, dim_tile_size] jax.Array reference.
        mutable_dx_ref: A [seq_len, dim_tile_size] jax.Array reference, modified by the function.
        mutable_da_ref: A [state_dim, dim_tile_size] jax.Array reference, modified by the function.
        mutable_db_ref: A [seq_len, state_dim] jax.Array reference, modified by the function.
        mutable_dc_ref: A [seq_len, state_dim] jax.Array reference, modified by the function.
        mutable_ddelta_ref: A [seq_len, dim_tile_size] jax.Array reference, modified by the function.
        mutable_dd_ref: A [1, dim_tile_size] jax.Array reference, modified by the function.
        mutable_dcarry_ref: A [state_dim, dim_tile_size] jax.Array reference, modified by the function.
        mutable_forward_hs_ref: A [seq_len, state_dim, dim_tile_size] jax.Array reference,
            modified by the function.
    """
    seq_len = dy_ref.shape[0]
    dtype = _in_kernel_dtype()
    matmul_prec = _matmul_precision(dtype)
    a = a_ref[:].astype(dtype)

    # If it's the final backward block, which corresponds to the first forward block,
    # then the boundary state is 0.
    @pl.when(pl.program_id(axis=2) == (pl.num_programs(axis=2) - 1))
    def zero_boundary():
        boundary_hs_ref[:] = jnp.zeros_like(boundary_hs_ref)

    def _forward_kernel_save_hs_body(t: int, carry: jax.Array):
        """Computes h_{t+1} = a_bar_t * h_t + b_bar_t * x_t and saves it in
        `mutable_forward_hs_ref`."""
        next_carry = _update_carry(
            t,
            carry,
            dtype=dtype,
            x_ref=x_ref,
            a=a,
            b_ref=b_ref,
            delta_ref=delta_ref,
        )
        mutable_forward_hs_ref[t] = next_carry
        return next_carry

    # Compute all the forward states in the block and store in forward_hs_ref.
    # TODO(swiseman): investigate computing dc as part of this loop (h/t bailin-wang);
    # currently we run into scoped vmem errors.
    h_carry = boundary_hs_ref[:]

    _ = lax.fori_loop(
        0,
        seq_len,
        _forward_kernel_save_hs_body,
        h_carry,
    )

    d = jnp.squeeze(d_ref[:], axis=0).astype(dtype)
    ones_row = jnp.ones((1, a_ref.shape[1]))[:]

    def _backward_kernel_body(i: int, carry: jax.Array):
        """Computes gradient contributions wrt parameters from time-step t, and
        partially computes d/dh_{t-1}."""
        t = seq_len - 1 - i
        dh_t = carry.astype(dtype)
        delta = jnp.expand_dims(delta_ref[t].astype(dtype), axis=0)  # [1, inner]
        a_bar = jnp.exp(delta * a)
        dL_da_bar = (
            dh_t * mutable_forward_hs_ref[t - 1].astype(dtype) * a_bar
        )  # [state, inner]
        mutable_da_ref[:] += dL_da_bar * delta

        x = jnp.expand_dims(x_ref[t].astype(dtype), axis=0)  # [1, inner]
        b = jnp.expand_dims(b_ref[t].astype(dtype), axis=-1)  # [state, 1]
        ddelta = jnp.sum(dL_da_bar * a, axis=0) + jnp.sum(dh_t * b * x, axis=0)
        mutable_ddelta_ref[t] = ddelta.astype(mutable_ddelta_ref.dtype)

        dh_t_delta = dh_t * delta
        # Sum over the inner dimension with a matmul arranged to satisfy Triton
        # dot operand constraints (second operand dims >= 16).
        db_t = jax.lax.dot_general(
            ones_row,
            (dh_t_delta * x).T,
            (
                ((1,), (0,)),
                ((), ()),
            ),
            preferred_element_type=jnp.float32,
            precision=matmul_prec,
        )
        mutable_db_ref[t] = db_t.T
        dy = dy_ref[t]
        mutable_dd_ref[:] += x.astype(dtype) * dy
        mutable_dx_ref[t] = (jnp.sum(dh_t_delta * b, axis=0) + dy * d).astype(
            mutable_dx_ref.dtype
        )
        dc_t = jax.lax.dot_general(
            jnp.expand_dims(dy, axis=0),  # [1, inner]
            mutable_forward_hs_ref[t].astype(dtype).T,  # [inner, state]
            (
                ((1,), (0,)),
                ((), ()),
            ),
            preferred_element_type=jnp.float32,
            precision=matmul_prec,
        )
        mutable_dc_ref[t] = dc_t.T
        # Compute d/dh_{t-1} and store it.
        dh_prev = jnp.expand_dims(dy_ref[t - 1], axis=0) * jnp.expand_dims(
            c_ref[t - 1], axis=-1
        )
        dh_prev += dh_t * a_bar
        return dh_prev

    # Initialize da and dd to zero if this is the first backward tile.
    @pl.when(pl.program_id(axis=2) == 0)
    def zero_da_dd():
        mutable_da_ref[:] = jnp.zeros_like(mutable_da_ref)
        mutable_dd_ref[:] = jnp.zeros_like(mutable_dd_ref)

    # Finish computing d/dh_T. If it's the final timestep, dcarry_ref gets 0s. Otherwise it contains
    # d/dh_T as computed in the subsequent tile, minus the contribution from the current step.
    @pl.when(pl.program_id(axis=2) == 0)
    def zero_dcarry():
        mutable_dcarry_ref[:] = jnp.zeros_like(mutable_dcarry_ref)

    dcarry = mutable_dcarry_ref[:] + (
        jnp.expand_dims(dy_ref[seq_len - 1].astype(dtype), axis=0)
        * jnp.expand_dims(c_ref[seq_len - 1].astype(dtype), axis=-1)
    )

    # Compute param grads for steps T thru 2 and d/dh_t for steps T-1 thru 1.
    dh_1 = lax.fori_loop(
        0,
        seq_len - 1,
        _backward_kernel_body,
        dcarry,
    )
    # Compute param grads for step 1. Note: difficult to refactor this without adding
    # control-flow, which may slow things down.
    delta1 = jnp.expand_dims(delta_ref[0].astype(dtype), axis=0)
    a_bar1 = jnp.exp(delta1 * a)
    dL_da1_bar = dh_1 * boundary_hs_ref[:].astype(dtype) * a_bar1
    mutable_da_ref[:] += dL_da1_bar * delta1

    x1 = jnp.expand_dims(x_ref[0].astype(dtype), axis=0)
    b1 = jnp.expand_dims(b_ref[0].astype(dtype), axis=-1)
    mutable_ddelta_ref[0] = (
        jnp.sum(dL_da1_bar * a, axis=0) + jnp.sum(dh_1 * b1 * x1, axis=0)
    ).astype(mutable_ddelta_ref.dtype)
    # Compute d/db1 and d/dx1
    dh_1_delta = dh_1 * delta1
    db_1 = jax.lax.dot_general(
        ones_row,
        (dh_1_delta * x1).T,
        (
            ((1,), (0,)),
            ((), ()),
        ),
        preferred_element_type=jnp.float32,
        precision=matmul_prec,
    )
    mutable_db_ref[0] = db_1.T
    dy1 = dy_ref[0]
    mutable_dd_ref[:] += dy1 * x1.astype(dtype)
    mutable_dx_ref[0] = (jnp.sum(dh_1_delta * b1, axis=0) + dy1 * d).astype(
        mutable_dx_ref.dtype
    )
    dc_1 = jax.lax.dot_general(
        jnp.expand_dims(dy1, axis=0),
        mutable_forward_hs_ref[0].astype(dtype).T,
        (
            ((1,), (0,)),
            ((), ()),
        ),
        preferred_element_type=jnp.float32,
        precision=matmul_prec,
    )
    mutable_dc_ref[0] = dc_1.T
    # Compute part of d/d_h0 (i.e., grad wrt final step of previous tile). If necessary,
    # the contribution from previous timestep will be added above.
    dh_0 = dh_1 * a_bar1
    mutable_dcarry_ref[:] = dh_0


def _loop_backward_pallas(
    dy: jax.Array,
    x: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    delta: jax.Array,
    d: jax.Array,
    seq_tile_size: int,
    dim_tile_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute the Mamba backward pass using a Pallas kernel.

    Args:
        dy: [batch_size, seq_len, inner_dim]
        x: [batch_size, seq_len, inner_dim]
        a: [state_dim, inner_dim]
        b: [batch_size, seq_len, state_dim]
        c: [batch_size, seq_len, state_dim]
        delta: [batch_size, seq_len, inner_dim]
        d: [1, inner_dim]
        seq_tile_size: The size of the tiles into which the sequence dimension will be divided.
        dim_tile_size: The size of the tiles into which the "inner" dimension will be divided.

    Returns:
        A 6-tuple of gradients wrt x, a, b, c, delta, and d (resp.) each having the same shape
        as the original argument.
    """
    # Calculating each time-step's contribution to the Mamba parameter gradients requires
    # access to forward states from both the current and previous time-steps. We want to avoid
    # saving all forward states in HBM, and so we use the following strategy:
    # - Recompute all forward states before the backward pass in order to obtain "boundary" states.
    #     - "Boundary" states correspond to the last time-step of a block.
    # - Save boundary states to HBM.
    # - Pass the boundary state from the i-th block to the i+1-th block, so each block has the state
    #   from before its first time-step.
    # - Recompute the states within a block from the boundary state, and update gradients.
    #
    # A faster and simpler solution would involve simply reversing the Mamba forward pass, from
    # the last time-step. But this seems to be difficult to accomplish in a numerically stable way.

    state_dim = b.shape[2]
    x_dtype = x.dtype
    # All parameters except `a` need to be in float32 in order for Pallas not to complain.
    dy, x, b, c, delta, d = (arg.astype(jnp.float32) for arg in [dy, x, b, c, delta, d])
    batch_tile_size = 1
    x_spec, a_spec, b_spec, d_spec, carry_spec = _parameter_blockspecs(
        state_dim,
        seq_tile_size,
        dim_tile_size,
    )
    hcarry_shape = (x.shape[0], a.shape[0], dim_tile_size)
    carry_dtype = _in_kernel_dtype()
    # Create a batch_size x inner_dim x seq_len grid, which allows us to carry state
    # between sequence blocks.
    grid = (
        x.shape[0] // batch_tile_size,
        x.shape[2] // dim_tile_size,
        x.shape[1] // seq_tile_size,
    )
    # Prepare to save the seq_len / seq_tile_size boundary states in HBM.
    seq_grid_size = x.shape[1] // seq_tile_size
    boundary_hs_shape = (
        x.shape[0],
        seq_grid_size + 1,
    ) + a.shape  # [batch_size, seq_len / seq_tile_size + 1, state_dim, inner_dim]
    boundary_hs_tiling = (None, None, state_dim, dim_tile_size)
    # In the forward pass, i-th block writes to i+1-th block for use in the backward pass.
    bdry_spec = _bs(lambda b, d, s: (b, s + 1, 0, d), boundary_hs_tiling)
    bw_bdry_spec = _bs(
        lambda b, d, s: (b, seq_grid_size - 1 - s, 0, d), boundary_hs_tiling
    )
    # Write boundary states to hbm.
    _, boundary_hs = pallas_call_compat(
        _forward_kernel_boundary_hs,
        grid=grid,
        in_specs=[x_spec, a_spec, b_spec, x_spec],
        out_shape=[
            jax.ShapeDtypeStruct(hcarry_shape, carry_dtype),
            jax.ShapeDtypeStruct(boundary_hs_shape, carry_dtype),
        ],
        out_specs=[carry_spec, bdry_spec],
        compiler_params=_tpu_compiler_params(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        backend=_pallas_backend(),
        interpret=use_interpret_mode(),
    )(x, a, b, delta)

    # Create BlockSpecs for gradient jax.Arrays with different shapes from their forward
    # counterparts.
    # We will not have contiguous writes along the batch_size axis, so we cannot
    # accumulate da or dd gradients over examples. We therefore add a batch dimension.
    da_shape = (x.shape[0],) + a.shape
    da_tiling = (None, state_dim, dim_tile_size)
    da_spec = _bs(lambda b, d, s: (b, 0, d), da_tiling)

    dd_shape = (x.shape[0],) + d.shape
    dd_tiling = (None, 1, dim_tile_size)
    dd_spec = _bs(lambda b, d, s: (b, 0, d), dd_tiling)

    # Similarly, db and dc gradients cannot accumulate over the inner_dim blocks,
    # since they are not accessed contiguously. We therefore add a seq_tile_size
    # dimension.
    db_shape = (grid[1],) + b.shape + (1,)
    db_tiling = (None, None, seq_tile_size, state_dim, 1)
    db_spec = _bs(lambda b, d, s: (d, b, s, 0, 0), db_tiling)

    # Reverse BlockSpecs along the sequence axis, since gradients are computed from
    # last time-step to first.
    def _reversed_blockspec(spec):
        return _bs(
            lambda b, d, s: spec.index_map(b, d, seq_grid_size - 1 - s),
            spec.block_shape,
        )

    bw_x_spec = _reversed_blockspec(x_spec)
    bw_b_spec = _reversed_blockspec(b_spec)
    db_spec = _reversed_blockspec(db_spec)

    # Allocate scratch space to recompute forward states within a block. Ideally
    # this could be done directly in VMEM, but it does not appear to be possible.
    forward_hs_shape = (batch_tile_size, seq_tile_size, state_dim, dim_tile_size)
    forward_hs_tiling = (None, seq_tile_size, state_dim, dim_tile_size)
    forward_hs_spec = _bs(lambda b, d, s: (0, 0, 0, 0), forward_hs_tiling)

    dx, da, db, dc, ddelta, dd, _, _ = pallas_call_compat(
        _backward_kernel,
        grid=grid,
        in_specs=[
            bw_x_spec,
            bw_x_spec,
            a_spec,
            bw_b_spec,
            bw_b_spec,
            bw_x_spec,
            d_spec,
            bw_bdry_spec,
        ],
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, carry_dtype),
            jax.ShapeDtypeStruct(da_shape, carry_dtype),
            jax.ShapeDtypeStruct(db_shape, carry_dtype),
            jax.ShapeDtypeStruct(db_shape, carry_dtype),
            jax.ShapeDtypeStruct(delta.shape, carry_dtype),
            jax.ShapeDtypeStruct(dd_shape, carry_dtype),
            jax.ShapeDtypeStruct(hcarry_shape, carry_dtype),
            jax.ShapeDtypeStruct(forward_hs_shape, carry_dtype),
        ],
        out_specs=[
            bw_x_spec,
            da_spec,
            db_spec,
            db_spec,
            bw_x_spec,
            dd_spec,
            carry_spec,
            forward_hs_spec,
        ],
        compiler_params=_tpu_compiler_params(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        backend=_pallas_backend(),
        interpret=use_interpret_mode(),
    )(
        dy,
        x,
        a,
        b,
        c,
        delta,
        d,
        boundary_hs,
    )
    dx = dx.astype(x_dtype)
    # Accumulate gradients over any additional dimensions we've added,
    # and cast to the right type.
    da = jnp.sum(da, axis=0).astype(x_dtype)
    db = jnp.squeeze(jnp.sum(db, axis=0), axis=-1).astype(x_dtype)
    dc = jnp.squeeze(jnp.sum(dc, axis=0), axis=-1).astype(x_dtype)
    ddelta = ddelta.astype(x_dtype)
    dd = jnp.sum(dd, axis=0).astype(x_dtype)
    return dx, da, db, dc, ddelta, dd


def _forward_kernel(
    x_ref: jax.Array,
    a_ref: jax.Array,
    b_ref: jax.Array,
    c_ref: jax.Array,
    delta_ref: jax.Array,
    d_ref: jax.Array,
    mutable_y_ref: jax.Array,
    mutable_carry_ref: jax.Array,
):
    """Computes the Mamba forward pass.

    Args:
        x_ref: A [seq_len, dim_tile_size] jax.Array reference.
        a_ref: A [state_dim, dim_tile_size] jax.Array reference.
        b_ref: A [seq_len, state_dim] jax.Array reference.
        c_ref: A [seq_len, state_dim] jax.Array reference.
        delta_ref: A [seq_len, dim_tile_size] jax.Array reference.
        d_ref: A [1, dim_tile_size] jax.Array reference.
        mutable_y_ref: A [seq_len, dim_tile_size] jax.Array reference, modified by the function.
        mutable_carry_ref: A [state_dim, dim_tile_size] jax.Array reference to ht,
            modified by the function.
    """
    dtype = _in_kernel_dtype()
    matmul_prec = _matmul_precision(dtype)
    a = a_ref[:].astype(dtype)
    d_vec = d_ref[:].astype(dtype)

    def _forward_kernel_body(t: int, carry: jax.Array):
        """Computes h_{t+1}, stores it in mutable_carry_ref, computes output,
        and stores it in y_ref."""
        # Compute h_{t+1} = h_t * a_bar_t + b_bar_x_t, where:
        # a_bar_t = exp(delta_t * a), and b_bar_x_t = delta_t * b_t * x_t.
        # The calculation of the carry below can be replaced with `update_carry`, but this
        # decreases speed by about 5%, presumably because `a_bar` cannot then stay in a register.
        delta = jnp.expand_dims(delta_ref[t].astype(dtype), axis=0)  # [1, inner_dim]
        a_bar = jnp.exp(delta * a)  # [state_dim, inner_dim]
        x = jnp.expand_dims(x_ref[t].astype(dtype), axis=0)  # [1, inner_dim]
        b_bar = delta * jnp.expand_dims(b_ref[t].astype(dtype), axis=-1) * x
        a_bar *= carry
        # Put h_{t+1} in the a_bar register.
        a_bar += b_bar
        ct = jnp.expand_dims(c_ref[t].astype(dtype), axis=0)  # [1, state_dim]
        yt = jax.lax.dot_general(  # Equivalent to ct @ a_bar: [1, inner_dim]
            ct,
            a_bar,
            (
                ((1,), (0,)),
                ((), ()),
            ),
            preferred_element_type=jnp.float32,
            precision=matmul_prec,
        )
        x *= d_vec
        yt += x
        mutable_y_ref[t] = jnp.squeeze(yt, axis=0).astype(mutable_y_ref.dtype)
        return a_bar

    seq_len = x_ref.shape[0]

    # Initialize h_carry_ref to 0 iff it's the first time-step. Otherwise do nothing.
    @pl.when(pl.program_id(axis=2) == 0)
    def zero_carry():
        mutable_carry_ref[:] = jnp.zeros_like(mutable_carry_ref)

    # Loop variables need to be "concrete," not references.
    h_carry = mutable_carry_ref[:]
    # Fill y_ref inside the loop.
    h_T = lax.fori_loop(
        0,
        seq_len,
        _forward_kernel_body,
        h_carry,
    )
    # Store the carry as a reference, for possible use later.
    mutable_carry_ref[:] = h_T


def _loop_forward_pallas(
    x: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    delta: jax.Array,
    d: jax.Array,
    seq_tile_size: int,
    dim_tile_size: int,
) -> jax.Array:
    """Compute the Mamba scan using a Pallas kernel.

    Args:
        x: [batch_size, seq_len, inner_dim]
        a: [state_dim, inner_dim]
        b: [batch_size, seq_len, state_dim]
        c: [batch_size, seq_len, state_dim]
        delta: [batch_size, seq_len, inner_dim]
        d: [1, inner_dim]
        seq_tile_size: The size of the tiles into which the sequence dimension will be divided.
        dim_tile_size: The size of the tiles into which the "inner" dimension will be divided.

    Returns:
        A [batch_size, seq_len, inner_dim] jax.Array representing the output of Mamba's forward scan.
    """
    state_dim = b.shape[2]
    x_dtype = x.dtype
    carry_dtype = _in_kernel_dtype()

    batch_tile_size = 1  # We always consider batch examples individually.
    # Create a batch_size x inner_dim x seq_len grid, which allows us to carry state
    # between sequence blocks.
    grid = (
        x.shape[0] // batch_tile_size,
        x.shape[2] // dim_tile_size,
        x.shape[1] // seq_tile_size,
    )
    x_spec, a_spec, b_spec, d_spec, carry_spec = _parameter_blockspecs(
        state_dim, seq_tile_size, dim_tile_size
    )
    # Create the output shape for the carry. We just need its last dimension to be
    # the size of the block, since we will run through the entire sequence before
    # moving on to the next inner_dim-block.
    hcarry_shape = (x.shape[0], a.shape[0], dim_tile_size)

    outputs = pallas_call_compat(
        _forward_kernel,
        grid=grid,
        in_specs=[x_spec, a_spec, b_spec, b_spec, x_spec, d_spec],
        out_shape=[
            # Note: using bfloat16 as y's return type results in each y_t being 0
            # for all t s.t. t % seq_tile_size == 1.
            jax.ShapeDtypeStruct(x.shape, carry_dtype),  # Output 1: y.
            jax.ShapeDtypeStruct(hcarry_shape, carry_dtype),  # Output 2: hcarry.
        ],
        out_specs=[x_spec, carry_spec],
        compiler_params=_tpu_compiler_params(
            # Below we tell the compiler which dimensions can be parallelized.
            # All but the sequence dimension are embarassingly parallel.
            dimension_semantics=(
                "parallel",
                "parallel",
                "arbitrary",
            )
        ),
        backend=_pallas_backend(),
        interpret=use_interpret_mode(),
    )(
        x.astype(jnp.float32),
        a,
        b.astype(jnp.float32),
        c.astype(jnp.float32),
        delta.astype(jnp.float32),
        d,
    )
    y = outputs[0].astype(x_dtype)
    return y


def _make_mamba_scan(seq_tile_size: int, dim_tile_size: int):
    """Build a differentiable, SPMD-partitionable Mamba scan for the given tile sizes.

    Returns a function ``f(x, a, b, c, delta, d) -> y`` that:
      - Uses ``custom_partitioning`` on the forward and backward Pallas kernels
        so that GSPMD runs them per-shard without inserting all-gathers.
      - Uses ``custom_vjp`` so that ``jax.grad`` works through the scan.

    The tile sizes are captured in the closure so that the returned function
    accepts only array arguments (required by ``custom_partitioning``).
    """
    from jax.experimental.custom_partitioning import custom_partitioning

    # -- Forward kernel with custom_partitioning --------------------------
    @custom_partitioning
    def _fwd(x, a, b, c, delta, d):
        return _loop_forward_pallas(x, a, b, c, delta, d, seq_tile_size, dim_tile_size)

    def _fwd_partition(mesh, arg_shapes, result_shape):
        # Validate shardings: reject seq/inner_dim sharding.
        names = ("x", "a", "b", "c", "delta", "d")
        for shape, name in zip(arg_shapes, names):
            _validate_mamba_sharding(shape.sharding, name)

        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(x, a, b, c, delta, d):
            return _loop_forward_pallas(
                x, a, b, c, delta, d, seq_tile_size, dim_tile_size
            )

        return mesh, lower_fn, result_shardings, arg_shardings

    _fwd_partition._cp_raw_fn = lambda x, a, b, c, delta, d: _loop_forward_pallas(
        x, a, b, c, delta, d, seq_tile_size, dim_tile_size
    )  # type: ignore[attr-defined]

    def_partition_compat(
        _fwd.def_partition,
        partition=_fwd_partition,
        # x(B,L,D) a(S,D) b(B,L,S) c(B,L,S) delta(B,L,D) d(O,D) -> y(B,L,D)
        # Only batch dim is shardable; seq/dim/state/one must be replicated.
        sharding_rule=(
            "batch seq dim, state dim, batch seq state, "
            "batch seq state, batch seq dim, one dim "
            "-> batch seq dim"
        ),
        need_replication_factors=("seq", "dim", "state", "one"),
    )

    # -- Backward kernel with custom_partitioning -------------------------
    @custom_partitioning
    def _bwd(dy, x, a, b, c, delta, d):
        return _loop_backward_pallas(
            dy, x, a, b, c, delta, d, seq_tile_size, dim_tile_size
        )

    def _bwd_partition(mesh, arg_shapes, result_shape):
        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(dy, x, a, b, c, delta, d):
            return _loop_backward_pallas(
                dy, x, a, b, c, delta, d, seq_tile_size, dim_tile_size
            )

        return mesh, lower_fn, result_shardings, arg_shardings

    _bwd_partition._cp_raw_fn = lambda dy, x, a, b, c, delta, d: _loop_backward_pallas(
        dy, x, a, b, c, delta, d, seq_tile_size, dim_tile_size
    )  # type: ignore[attr-defined]

    def_partition_compat(
        _bwd.def_partition,
        partition=_bwd_partition,
        # dy(B,L,D) x(B,L,D) a(S,D) b(B,L,S) c(B,L,S) delta(B,L,D) d(O,D)
        # -> dx(B,L,D) da(S,D) db(B,L,S) dc(B,L,S) ddelta(B,L,D) dd(O,D)
        sharding_rule=(
            "batch seq dim, batch seq dim, state dim, "
            "batch seq state, batch seq state, batch seq dim, one dim "
            "-> batch seq dim, state dim, batch seq state, "
            "batch seq state, batch seq dim, one dim"
        ),
        need_replication_factors=("seq", "dim", "state", "one"),
    )

    # -- Differentiable wrapper using custom_vjp --------------------------
    @jax.custom_vjp
    def _scan(x, a, b, c, delta, d):
        return _fwd(x, a, b, c, delta, d)

    def _scan_fwd(x, a, b, c, delta, d):
        y = _fwd(x, a, b, c, delta, d)
        return y, (x, a, b, c, delta, d)

    def _scan_bwd(res, dy):
        x, a, b, c, delta, d = res
        return _bwd(dy, x, a, b, c, delta, d)

    _scan.defvjp(_scan_fwd, _scan_bwd)
    return _scan


def _pad_to_multiple(x: jax.Array, *, divisor: int, axis: int) -> jax.Array:
    """Pads the variable `x` to have size along `axis` divisible by `divisor`."""
    if x.shape[axis] % divisor == 0:
        return x
    n = divisor - x.shape[axis] % divisor
    pad_shape = list(x.shape)
    pad_shape[axis] = n
    zeros = jnp.zeros(pad_shape, dtype=x.dtype)
    return jnp.concatenate([x, zeros], axis=axis)


def _mamba_scan_reference(
    x: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    delta: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Pure-JAX reference implementation of the Mamba recurrence."""
    dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    a_f32 = a.astype(jnp.float32)
    b_f32 = b.astype(jnp.float32)
    c_f32 = c.astype(jnp.float32)
    delta_f32 = delta.astype(jnp.float32)
    d_f32 = d.astype(jnp.float32)

    batch_size, _, inner_dim = x_f32.shape
    state_dim = a_f32.shape[0]
    h0 = jnp.zeros((batch_size, state_dim, inner_dim), dtype=jnp.float32)

    def step(carry, inputs):
        x_t, b_t, c_t, delta_t = inputs  # [B,D], [B,S], [B,S], [B,D]
        a_bar = jnp.exp(delta_t[:, None, :] * a_f32[None, :, :])  # [B,S,D]
        b_bar = delta_t[:, None, :] * b_t[:, :, None] * x_t[:, None, :]  # [B,S,D]
        h = a_bar * carry + b_bar
        y_t = jnp.einsum("bs,bsd->bd", c_t, h, preferred_element_type=jnp.float32)
        y_t = y_t + x_t * d_f32
        return h, y_t.astype(dtype)

    _, y = jax.lax.scan(
        step,
        h0,
        (
            jnp.swapaxes(x_f32, 0, 1),
            jnp.swapaxes(b_f32, 0, 1),
            jnp.swapaxes(c_f32, 0, 1),
            jnp.swapaxes(delta_f32, 0, 1),
        ),
    )
    return jnp.swapaxes(y, 0, 1)


def compute_mamba_scan(
    x: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    delta: jax.Array,
    d: jax.Array,
    *,
    seq_tile_size: int,
    dim_tile_size: int,
) -> jax.Array:
    """Computes a Mamba scan given inputs. This function is the external interface
    to using Pallas kernels for Mamba scans.

    When inputs carry ``NamedSharding`` (e.g. batch-sharded across a device
    mesh), the forward and backward Pallas kernels are automatically run
    per-shard via ``custom_partitioning`` — no ``shard_map`` wrapping is needed
    at the call site.

    Args:
        x: [batch_size, seq_len, inner_dim]
        a: [state_dim, inner_dim]
        b: [batch_size, seq_len, state_dim]
        c: [batch_size, seq_len, state_dim]
        delta: [batch_size, seq_len, inner_dim]
        d: [1, inner_dim]
        seq_tile_size: The size of the tiles into which the sequence dimension will be divided.
        dim_tile_size: The size of the tiles into which the "inner" dimension will be divided.

    Returns:
        A [batch_size, seqlen, inner_dim] jax.Array representing the Mamba scan's output.
    """
    _validate_mamba_runtime_inputs(
        x,
        a,
        b,
        c,
        delta,
        d,
        seq_tile_size=seq_tile_size,
        dim_tile_size=dim_tile_size,
    )

    backend = jax.default_backend()
    pallas_backend = _pallas_backend()
    if backend == "cpu":
        raise RuntimeError("compute_mamba_scan requires an accelerator backend.")
    if backend not in ("tpu", "gpu"):
        raise RuntimeError(
            f"compute_mamba_scan only supports TPU and GPU backends, got {backend!r}."
        )
    if backend == "gpu" and not _gpu_supports_mamba_pallas_for_shape(
        seq_len=x.shape[1],
        inner_dim=x.shape[2],
        state_dim=a.shape[0],
        seq_tile_size=seq_tile_size,
        dim_tile_size=dim_tile_size,
        pallas_backend=pallas_backend,
    ):
        raise RuntimeError(
            "compute_mamba_scan does not support this GPU Pallas configuration. "
            f"backend={pallas_backend!r}, seq_len={x.shape[1]}, "
            f"inner_dim={x.shape[2]}, state_dim={a.shape[0]}, "
            f"seq_tile_size={seq_tile_size}, dim_tile_size={dim_tile_size}. "
            "Reference fallback on GPU is disabled."
        )

    _, seqlen, inner = x.shape

    x, b, c, delta = (
        _pad_to_multiple(arg, divisor=seq_tile_size, axis=1) for arg in [x, b, c, delta]
    )
    x, delta = (
        _pad_to_multiple(arg, divisor=dim_tile_size, axis=2) for arg in [x, delta]
    )
    a, d = (_pad_to_multiple(arg, divisor=dim_tile_size, axis=1) for arg in [a, d])

    # Build a custom_partitioning-aware scan for these tile sizes.
    # When inputs are NamedSharded, GSPMD will call our partition()
    # callback and run the Pallas kernel per-shard without all-gathers.
    _scan = _make_mamba_scan(seq_tile_size, dim_tile_size)

    y = _scan(x, a, b, c, delta, d)
    # Remove zero-padding if any.
    return y[:, :seqlen, :inner]
