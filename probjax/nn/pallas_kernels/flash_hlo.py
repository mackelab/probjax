from functools import partial, wraps

import numpy as np
import jax
import jax.numpy as jnp
from jax.extend import core
from jax.core import ShapedArray
from jax import dtypes
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters.mlir import ir
from jax.ffi import ffi_call


import einops

import flash_attn_jax_lib.flash_api as flash_api

# ==== Register primitives ====

_flash_mha_fwd_hlo_p = core.Primitive("flash_mha_fwd_hlo")
_flash_mha_fwd_hlo_p.multiple_results = True
_flash_mha_fwd_hlo_p.def_impl(partial(xla.apply_primitive, _flash_mha_fwd_hlo_p))

_flash_mha_bwd_hlo_p = core.Primitive("flash_mha_bwd_hlo")
_flash_mha_bwd_hlo_p.multiple_results = True
_flash_mha_bwd_hlo_p.def_impl(partial(xla.apply_primitive, _flash_mha_bwd_hlo_p))

# ==== Primitive wrapper ====


def _flash_mha_fwd_hlo(q, k, v, softmax_scale, is_causal, window_size):
    out, lse = _flash_mha_fwd_hlo_p.bind(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size=window_size,
    )
    return out, lse


def _flash_mha_bwd_hlo(dout, q, k, v, out, lse, softmax_scale, is_causal, window_size):
    dq, dk, dv = _flash_mha_bwd_hlo_p.bind(
        dout,
        q,
        k,
        v,
        out,
        lse,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size=window_size,
    )
    return dq, dk, dv


# ==== HLO lowerings ====

#Register functions defined in gpu_ops using the new FFI mechanism
for _name, _value in flash_api.get_registrations().items():
    print(_name, _value)
    jax.ffi.register_ffi_target(_name, _value, platform="gpu")


def default_layouts(*shapes):
    def row_major(shape):
        return range(len(shape) - 1, -1, -1)

    return [row_major(shape) for shape in shapes]


def value_layouts(*values):
    return default_layouts(*[ir.RankedTensorType(x.type).shape for x in values])


def ir_type_to_dtype(ty):
    for dtype in [np.dtype('bfloat16'), np.dtype('float16'), np.dtype('float32')]:
        if ty == mlir.dtype_to_ir_type(dtype):
            return dtype


def _flash_mha_fwd_hlo_lowering(
    ctx, q, k, v, softmax_scale=None, is_causal=False, window_size=None
):
    q_type = ir.RankedTensorType(q.type)
    q_shape = q_type.shape
    k_type = ir.RankedTensorType(k.type)
    k_shape = k_type.shape
    v_type = ir.RankedTensorType(v.type)
    v_shape = v_type.shape

    assert q_type.element_type == k_type.element_type, (
        "Q and K must have the same dtype"
    )
    assert q_type.element_type == v_type.element_type, (
        "Q and V must have the same dtype"
    )
    element_type = q_type.element_type
    assert type(element_type) in [ir.F16Type, ir.BF16Type], (
        "Only support fp16 and bf16 data type"
    )
    [n, l, h, d] = q_shape
    [nk, lk, hk, dk] = k_shape
    assert k_shape == v_shape, "K and V must have the same shape"
    assert [n, d] == [nk, dk], "Q and K must have the same batch size and head size"

    opaque = flash_api.make_flash_mha_fwd_args(
        0.0,  # p_dropout
        softmax_scale,
        is_causal,  # is_causal
        window_size[0],  # window_size_left
        window_size[1],  # window_size_right
        False,  # return_softmax
        n,
        l,
        h,
        d,
        lk,
        hk,
        flash_api.BF16 if type(element_type) == ir.BF16Type else flash_api.FP16,
        0,
    )

    def fwd(q, k, v):
        dpad = (8 - d % 8) % 8
        if dpad > 0:
            # We need padding. It's better to let xla's allocator handle it here than directly call cudaMalloc.
            q = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, dpad)), 'constant')
            k = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, dpad)), 'constant')
            v = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, dpad)), 'constant')

        q_shape = [n, l, h, d + dpad]
        k_shape = [n, lk, hk, d + dpad]
        v_shape = [n, lk, hk, d + dpad]
        o_shape = [n, l, h, d + dpad]
        lse_shape = [n, h, l]

        lse_type = ir.RankedTensorType.get(
            [n, h, l], mlir.dtype_to_ir_type(jnp.float32.dtype)
        )
        out_types = [ir.RankedTensorType.get(o_shape, element_type), lse_type]
        operand_layouts = default_layouts(q_shape, k_shape, v_shape)
        result_layouts = default_layouts(o_shape, lse_shape)

        o, lse = ffi_call(
            "flash_mha_fwd",
            [q, k, v],
            out_types,
            opaque,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts,
        )

        if dpad > 0:
            o = o[:, :, :, :d]
        return o, lse

    return mlir.lower_fun(fwd, multiple_results=True)(ctx, q, k, v)


mlir.register_lowering(
    _flash_mha_fwd_hlo_p,
    _flash_mha_fwd_hlo_lowering,
    platform="gpu",
)


def _flash_mha_bwd_hlo_lowering(
    ctx, dout, q, k, v, out, lse, softmax_scale=None, is_causal=None, window_size=None
):
    dout_type = ir.RankedTensorType(dout.type).element_type
    q_type = ir.RankedTensorType(q.type).element_type
    k_type = ir.RankedTensorType(k.type).element_type
    v_type = ir.RankedTensorType(v.type).element_type
    out_type = ir.RankedTensorType(out.type).element_type
    lse_type = ir.RankedTensorType(lse.type).element_type

    assert type(q_type) in [ir.F16Type, ir.BF16Type]
    assert q_type == dout_type
    assert q_type == k_type
    assert q_type == v_type
    assert q_type == out_type
    assert type(lse_type) in [ir.F32Type]
    dtype = q_type

    dout_shape = ir.RankedTensorType(dout.type).shape
    q_shape = ir.RankedTensorType(q.type).shape
    k_shape = ir.RankedTensorType(k.type).shape
    v_shape = ir.RankedTensorType(v.type).shape
    out_shape = ir.RankedTensorType(out.type).shape
    lse_shape = ir.RankedTensorType(lse.type).shape
    [n, lq, hq, d] = q_shape
    [nk, lk, hk, dk] = k_shape
    assert n == nk
    assert d == dk

    assert list(
        map(list, [dout_shape, q_shape, k_shape, v_shape, out_shape, lse_shape])
    ) == [
        [n, lq, hq, d],
        [n, lq, hq, d],
        [n, lk, hk, d],
        [n, lk, hk, d],
        [n, lq, hq, d],
        [n, hq, lq],
    ]

    opaque = flash_api.make_flash_mha_bwd_args(
        0.0,  # p_dropout
        softmax_scale,
        is_causal,  # is_causal
        window_size[0],  # window_size_left
        window_size[1],  # window_size_right
        False,  # deterministic
        n,
        lq,
        hq,
        d,
        lk,
        hk,
        flash_api.BF16 if type(q_type) == ir.BF16Type else flash_api.FP16,
        0,
    )

    def fwd(dout, q, k, v, out, lse):
        dpad = (8 - d % 8) % 8
        if dpad > 0:
            # We need padding. It's better to let xla's allocator handle it here than directly call cudaMalloc.
            q = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, dpad)), 'constant')
            k = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, dpad)), 'constant')
            v = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, dpad)), 'constant')
            out = jnp.pad(out, ((0, 0), (0, 0), (0, 0), (0, dpad)), 'constant')
            dout = jnp.pad(dout, ((0, 0), (0, 0), (0, 0), (0, dpad)), 'constant')

        # For MQA/GQA, hq != hk, but we pass a hq sized output tensor to the kernel and sum over it afterwards to reduce the size.
        out_types = [
            ir.RankedTensorType.get([n, lq, hq, d + dpad], dtype),
            ir.RankedTensorType.get([n, lk, hq, d + dpad], dtype),
            ir.RankedTensorType.get([n, lk, hq, d + dpad], dtype),
        ]
        out_layouts = default_layouts(
            [n, lq, hq, d + dpad], [n, lk, hq, d + dpad], [n, lk, hq, d + dpad]
        )

        dq, dk, dv = ffi_call(
            "flash_mha_bwd",
            [dout, q, k, v, out, lse],
            out_types,
            opaque,
            operand_layouts=default_layouts(
                dout.shape, q.shape, k.shape, v.shape, out.shape, lse.shape
            ),
            result_layouts=out_layouts,
        )

        if hq != hk:
            assert hq > hk and hq % hk == 0
            m = hq // hk
            dk = einops.reduce(dk, 'n l (h m) d -> n l h d', reduction='sum', h=hk)
            dv = einops.reduce(dv, 'n l (h m) d -> n l h d', reduction='sum', h=hk)

        if dpad > 0:
            dq = dq[:, :, :, :d]
            dk = dk[:, :, :, :d]
            dv = dv[:, :, :, :d]

        return dq, dk, dv

    return mlir.lower_fun(fwd, multiple_results=True)(ctx, dout, q, k, v, out, lse)


mlir.register_lowering(
    _flash_mha_bwd_hlo_p,
    _flash_mha_bwd_hlo_lowering,
    platform="gpu",
)

# ==== Abstract evaluation rules ====


def _flash_mha_fwd_abstract(
    q, k, v, softmax_scale=None, is_causal=None, window_size=None
):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [n, l, h, d] = q.shape
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    return (
        ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),
        ShapedArray([n, h, l], jnp.float32),
    )


_flash_mha_fwd_hlo_p.def_abstract_eval(_flash_mha_fwd_abstract)


def _flash_mha_bwd_abstract(
    dout, q, k, v, out, lse, softmax_scale=None, is_causal=None, window_size=None
):
    dout_dtype = dtypes.canonicalize_dtype(dout.dtype)
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    out_dtype = dtypes.canonicalize_dtype(out.dtype)
    lse_dtype = dtypes.canonicalize_dtype(lse.dtype)
    [n, lq, hq, d] = q.shape
    assert len(set([dout_dtype, q_dtype, k_dtype, v_dtype, out_dtype])) == 1
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    return (
        ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),
        ShapedArray(k.shape, k_dtype, named_shape=k.named_shape),
        ShapedArray(v.shape, v_dtype, named_shape=v.named_shape),
    )


_flash_mha_bwd_hlo_p.def_abstract_eval(_flash_mha_bwd_abstract)


# from jaxlib.hlo_helpers import custom_call
# from jax.experimental.custom_partitioning import custom_partitioning

# from jax.sharding import PartitionSpec as P
# from jax.sharding import Mesh
# from jax.sharding import NamedSharding
# from jax.sharding import PositionalSharding

# from einops import rearrange
# import math


# # ==== Sharding ====

# _flash_mha_fwd_hlo_sharded = custom_partitioning(
#     _flash_mha_fwd_hlo, static_argnums=(3, 4, 5)
# )
# _flash_mha_bwd_hlo_sharded = custom_partitioning(
#     _flash_mha_bwd_hlo, static_argnums=(6, 7, 8)
# )

# from jax._src.ad_checkpoint import _optimization_barrier


# def is_replicated(sharding):
#     return (isinstance(sharding, PositionalSharding) and sharding.shape == (1,)) or (
#         isinstance(sharding, NamedSharding) and len(sharding.spec) == 0
#     )


# def partition_fwd(
#     softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape
# ):
#     result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
#     arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)

#     q_sharding = arg_shardings[0]
#     k_sharding = arg_shardings[1]
#     v_sharding = arg_shardings[2]
#     assert q_sharding == k_sharding and q_sharding == v_sharding, (
#         "Only support q, k, v sharing the same sharding."
#     )
#     if is_replicated(q_sharding):
#         result_sharding = (q_sharding, q_sharding)
#     elif isinstance(q_sharding, PositionalSharding):
#         (n, l, h, d) = q_sharding.shape
#         assert d == 1, "Sharding across `d` won't be efficient, so it's not supported."
#         assert l == 1, (
#             "For ring attention, use `with Mesh(...) as mesh` and NamedSharding."
#         )
#         result_shardings = q_sharding, q_sharding.reshape((n, h, 1))  # n h l
#         arg_shardings = q_sharding, q_sharding, q_sharding
#     elif isinstance(q_sharding, NamedSharding):
#         mesh = q_sharding.mesh
#         [n, l, h, d] = q_sharding.spec
#         assert d == None, (
#             "Sharding across `d` won't be efficient, so it's not supported."
#         )
#         if l != None:
#             # assert not is_causal and window_size == (-1,-1), "Ring attention doesn't support causal or local masking yet."
#             assert window_size == (-1, -1), (
#                 "Ring attention doesn't support local masking yet."
#             )
#             result_shardings = q_sharding, NamedSharding(mesh, P(n, h, l))
#             arg_shardings = q_sharding, q_sharding, q_sharding
#             axis_name = l
#             axis_size = mesh.shape[axis_name]
#             # ring attention
#             return (
#                 mesh,
#                 partial(
#                     ring_fwd,
#                     softmax_scale=softmax_scale,
#                     is_causal=is_causal,
#                     axis_name=axis_name,
#                     axis_size=axis_size,
#                     mha_fwd=_flash_mha_fwd_hlo,
#                 ),
#                 result_shardings,
#                 arg_shardings,
#             )
#         else:
#             result_shardings = q_sharding, NamedSharding(mesh, P(n, h, l))
#             arg_shardings = q_sharding, q_sharding, q_sharding

#     def fwd(q, k, v):
#         return _flash_mha_fwd_hlo(
#             q,
#             k,
#             v,
#             softmax_scale=softmax_scale,
#             is_causal=is_causal,
#             window_size=window_size,
#         )

#     return mesh, fwd, result_shardings, arg_shardings


# def infer_sharding_fwd(
#     softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape
# ):
#     arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
#     q_sharding = arg_shardings[0]
#     k_sharding = arg_shardings[1]
#     v_sharding = arg_shardings[2]
#     assert q_sharding == k_sharding and q_sharding == v_sharding, (
#         "Only support q, k, v sharing the same sharding."
#     )
#     if is_replicated(q_sharding):
#         result_sharding = (q_sharding, q_sharding)
#     elif isinstance(q_sharding, PositionalSharding):
#         [n, l, h, d] = q_sharding.shape
#         result_sharding = (
#             q_sharding,  # [n,l,h,d]
#             q_sharding.replicate(3).reshape(n, l, h).transpose((0, 2, 1)),  # [n,h,l]
#         )
#     elif isinstance(q_sharding, NamedSharding):
#         [n, l, h, d] = q_sharding.spec
#         result_sharding = (q_sharding, NamedSharding(q_sharding.mesh, P(n, h, l)))
#     else:
#         raise ValueError("Unsupported sharding type.", type(q_sharding))
#     return result_sharding


# _flash_mha_fwd_hlo_sharded.def_partition(
#     infer_sharding_from_operands=infer_sharding_fwd, partition=partition_fwd
# )


# def infer_sharding_bwd(
#     softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape
# ):
#     # args: dout, q, k, v, out, lse
#     # outs: dq, dk, dv
#     # i think generally we want the output sharding for dq,dk,dv to be the same as q,k,v?
#     arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
#     q_sharding = arg_shardings[1]
#     k_sharding = arg_shardings[2]
#     v_sharding = arg_shardings[3]
#     return q_sharding, k_sharding, v_sharding


# def partition_bwd(
#     softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape
# ):
#     result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
#     arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)

#     do_sharding = arg_shardings[0]
#     q_sharding = arg_shardings[1]
#     k_sharding = arg_shardings[2]
#     v_sharding = arg_shardings[3]
#     o_sharding = arg_shardings[4]
#     lse_sharding = arg_shardings[5]
#     assert q_sharding == k_sharding and q_sharding == v_sharding, (
#         "Only support q, k, v sharing the same sharding."
#     )
#     if is_replicated(q_sharding):
#         result_shardings = (q_sharding,) * 3
#     elif isinstance(q_sharding, PositionalSharding):
#         assert q_sharding == k_sharding, "Expect q and k sharding to match"
#         assert q_sharding == v_sharding, "Expect q and v sharding to match"
#         [n, l, h, d] = q_sharding.shape
#         assert d == 1, "Sharding across `d` won't be efficient, so it's not supported."
#         assert l == 1, (
#             "For ring attention, use `with Mesh(...) as mesh` and NamedSharding."
#         )
#         lse_sharding = q_sharding.reshape(n, h, 1)  # n h l
#         result_shardings = (q_sharding,) * 3
#         arg_shardings = (q_sharding,) * 5 + (lse_sharding,)
#     elif isinstance(q_sharding, NamedSharding):
#         mesh = q_sharding.mesh
#         [n, l, h, d] = q_sharding.spec
#         assert d == None, (
#             "Sharding across `d` won't be efficient, so it's not supported."
#         )
#         if l != None:
#             # assert not is_causal and window_size == (-1,-1), "Ring attention doesn't support causal or local masking yet."
#             assert window_size == (-1, -1), (
#                 "Ring attention doesn't support local masking yet."
#             )
#             result_shardings = q_sharding, q_sharding, q_sharding
#             lse_sharding = NamedSharding(mesh, P(n, h, l))
#             arg_shardings = (q_sharding,) * 5 + (lse_sharding,)
#             axis_name = l
#             axis_size = mesh.shape[axis_name]
#             # ring attention
#             return (
#                 mesh,
#                 partial(
#                     ring_bwd,
#                     softmax_scale=softmax_scale,
#                     is_causal=is_causal,
#                     axis_name=axis_name,
#                     axis_size=axis_size,
#                     mha_bwd=_flash_mha_bwd_hlo,
#                 ),
#                 result_shardings,
#                 arg_shardings,
#             )
#         else:
#             result_shardings = q_sharding, q_sharding, q_sharding
#             lse_sharding = NamedSharding(mesh, P(n, h, l))
#             arg_shardings = (q_sharding,) * 5 + (lse_sharding,)

#     def fwd(*args):
#         return _flash_mha_bwd_hlo(
#             *args,
#             softmax_scale=softmax_scale,
#             is_causal=is_causal,
#             window_size=window_size,
#         )

#     return mesh, fwd, result_shardings, arg_shardings


# _flash_mha_bwd_hlo_sharded.def_partition(
#     infer_sharding_from_operands=infer_sharding_bwd, partition=partition_bwd
# )
