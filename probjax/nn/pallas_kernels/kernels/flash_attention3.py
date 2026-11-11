from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ..kernel_utils.kernel_primitive import (
    KernelSpec,
    Operand,
    Output,
    _validate_shardings,
    ct,
    derive_bwd_spec,
    grad,
    make_kernel_primitive,
    out_res,
    res,
    shardable_kernel,
)

try:
    from jax._src.lib import cuda_versions
except Exception:  # pragma: no cover - depends on local jax build
    cuda_versions = None

if TYPE_CHECKING:
    from probjax.nn.pallas_kernels.kernels import attention_mgpu as _attention_mgpu
    from probjax.nn.pallas_kernels.kernels.attention_mgpu import (
        TuningConfig,
    )
    from probjax.nn.pallas_kernels.kernels.attention_mgpu import (
        attention as _attention_impl,
    )
    from probjax.nn.pallas_kernels.kernels.attention_mgpu import (
        attention_with_pipeline_emitter as _attention_with_pipeline_emitter_impl,
    )

    _FLASH3_IMPORT_ERROR: Exception | None = None
else:
    try:
        from probjax.nn.pallas_kernels.kernels import attention_mgpu as _attention_mgpu
        from probjax.nn.pallas_kernels.kernels.attention_mgpu import (
            TuningConfig,
        )
        from probjax.nn.pallas_kernels.kernels.attention_mgpu import (
            attention as _attention_impl,
        )
        from probjax.nn.pallas_kernels.kernels.attention_mgpu import (
            attention_with_pipeline_emitter as _attention_with_pipeline_emitter_impl,
        )

        _FLASH3_IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - depends on local jax build
        _attention_mgpu = None
        TuningConfig = Any
        _attention_impl = None
        _attention_with_pipeline_emitter_impl = None
        _FLASH3_IMPORT_ERROR = exc


_attention_forward_impl = (
    getattr(_attention_mgpu, "_attention_forward", None)
    if _attention_mgpu is not None
    else None
)
_attention_backward_impl = (
    getattr(_attention_mgpu, "_attention_bwd", None)
    if _attention_mgpu is not None
    else None
)


def _gpu_supports_mosaic() -> bool:
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        return False
    for device in gpu_devices:
        capability = getattr(device, "compute_capability", None)
        if capability is None:
            continue
        major = (
            capability[0] if isinstance(capability, tuple) else int(float(capability))
        )
        if major >= 9:
            return True
    return False


def _cuda_runtime_version() -> int | None:
    if cuda_versions is None:
        return None
    try:
        return cuda_versions.cuda_runtime_get_version()
    except Exception:
        return None


def _ensure_flash3_supported(*, causal: bool, use_pipeline_emitter: bool) -> None:
    if _FLASH3_IMPORT_ERROR is not None:
        raise RuntimeError(
            "flash_attention3 is not available in this JAX build. "
            f"Detected jax={jax.__version__}.",
        ) from _FLASH3_IMPORT_ERROR
    if jax.default_backend() != "gpu":
        raise RuntimeError("flash_attention3 requires a GPU backend.")
    if not _gpu_supports_mosaic():
        raise RuntimeError(
            "flash_attention3 requires a Mosaic-compatible GPU "
            "(NVIDIA Hopper, compute capability >= 9.0).",
        )
    if use_pipeline_emitter and causal:
        raise NotImplementedError(
            "Causal attention is not supported with the pipeline emitter.",
        )
    cuda_runtime_version = _cuda_runtime_version()
    if (
        causal
        and cuda_runtime_version is not None
        and 12080 <= cuda_runtime_version < 12091
    ):
        raise RuntimeError(
            "Causal flash_attention3 is unsupported for CUDA runtime versions "
            "12.8.0 <= CUDA < 12.9.1 due to a ptxas issue.",
        )


# ---------------------------------------------------------------------------
# flash-attention-3 primitives
# ---------------------------------------------------------------------------
# Two forward primitives (with/without the lse residual — outputs are static
# per primitive) plus the backward; sharding rules, validation, batching and
# abstract eval all derive from the specs.

# Queries and keys have their own sequence and head factors: the Mosaic kernel
# takes Lq != Lkv (cross-attention) and fewer K/V heads (GQA/MQA).
_FLASH_FWD_SPEC = KernelSpec(
    name="flash3_fwd",
    operands=(
        Operand("q", ("batch", "q_seq", "heads", "head_dim")),
        Operand("k", ("batch", "kv_seq", "kv_heads", "head_dim")),
        Operand("v", ("batch", "kv_seq", "kv_heads", "head_dim")),
        # Per-batch number of valid keys (KVLenMask / padded keys).
        Operand("kv_lengths", ("batch",), optional=True),
    ),
    outputs=(Output(("batch", "q_seq", "heads", "head_dim"), dtype_like="q"),),
    shardable=frozenset({"batch", "heads", "kv_heads"}),
)

_FLASH_FWD_RES_SPEC = KernelSpec(
    name="flash3_fwd_res",
    operands=_FLASH_FWD_SPEC.operands,
    outputs=(
        Output(("batch", "q_seq", "heads", "head_dim"), dtype_like="q"),
        Output(("batch", "heads", "q_seq"), dtype_like=jnp.float32),
    ),
    shardable=_FLASH_FWD_SPEC.shardable,
)

_FLASH_BWD_SPEC = derive_bwd_spec(
    _FLASH_FWD_RES_SPEC,
    name="flash3_bwd",
    operands=(
        ct(0, name="do"),
        res("q"),
        res("k"),
        res("v"),
        out_res(0, "out"),
        out_res(1, "lse"),
        res("kv_lengths"),
    ),
    outputs=(grad("q"), grad("k"), grad("v")),
)


def _run_flash_forward_raw(
    q, k, v, kv_lengths=None, *, config, save_residuals, use_pipeline_emitter
):
    if use_pipeline_emitter:
        if kv_lengths is not None:
            raise NotImplementedError("KV lengths are not supported with the pipeline emitter.")
        implementation = _attention_with_pipeline_emitter_impl
    else:
        implementation = _attention_forward_impl or _attention_impl
    if implementation is None:
        raise RuntimeError("This JAX build does not provide the Flash3 forward kernel.")
    extra = {} if kv_lengths is None else {"kv_lengths": kv_lengths}
    result: Any = implementation(q, k, v, config=config, save_residuals=save_residuals, **extra)
    if save_residuals:
        out, (lse,) = result
        return out, lse
    return result


def _run_flash_backward_raw(do, q, k, v, out, lse, kv_lengths=None, *, config):
    if kv_lengths is not None:
        return _attention_backward_impl(
            config, False, (q, k, v, out, lse), do, kv_lengths=kv_lengths
        )
    if _attention_backward_impl is None:
        attention_impl = _attention_impl
        if attention_impl is None:
            raise RuntimeError(
                "This JAX build does not provide the Flash3 backward kernel."
            )
        _, pullback = jax.vjp(
            lambda q, k, v: attention_impl(
                q, k, v, config=config, save_residuals=False
            ),
            q,
            k,
            v,
        )
        return pullback(do)
    return _attention_backward_impl(config, False, (q, k, v, out, lse), do)


def _flash_fwd_impl(q, k, v, kv_lengths, *, config, use_pipeline_emitter):
    return _run_flash_forward_raw(
        q,
        k,
        v,
        kv_lengths,
        config=config,
        save_residuals=False,
        use_pipeline_emitter=use_pipeline_emitter,
    )


def _flash_fwd_res_impl(q, k, v, kv_lengths, *, config, use_pipeline_emitter):
    return _run_flash_forward_raw(
        q,
        k,
        v,
        kv_lengths,
        config=config,
        save_residuals=True,
        use_pipeline_emitter=use_pipeline_emitter,
    )


def _flash_bwd_impl(do, q, k, v, out, lse, kv_lengths, *, config):
    return _run_flash_backward_raw(do, q, k, v, out, lse, kv_lengths, config=config)


flash3_fwd_p = make_kernel_primitive(_FLASH_FWD_SPEC, impl=_flash_fwd_impl)
flash3_fwd_res_p = make_kernel_primitive(_FLASH_FWD_RES_SPEC, impl=_flash_fwd_res_impl)
flash3_bwd_p = make_kernel_primitive(_FLASH_BWD_SPEC, impl=_flash_bwd_impl)


def _flash_forward(q, k, v, config, save_residuals, use_pipeline_emitter, kv_lengths=None):
    prim = flash3_fwd_res_p if save_residuals else flash3_fwd_p
    return shardable_kernel(
        prim,
        {"q": q, "k": k, "v": v, "kv_lengths": kv_lengths},
        config=config,
        use_pipeline_emitter=use_pipeline_emitter,
    )


@functools.lru_cache(maxsize=None)
def _make_flash_attention(config: TuningConfig):
    @jax.custom_vjp
    def _attn(q, k, v, kv_lengths):
        return _flash_forward(q, k, v, config, False, False, kv_lengths)

    def _attn_fwd(q, k, v, kv_lengths):
        out, lse = _flash_forward(q, k, v, config, True, False, kv_lengths)
        return out, (q, k, v, out, lse, kv_lengths)

    def _attn_bwd(res, do):
        do_q, do_k, do_v, out, lse, kv_lengths = res
        dq, dk, dv = shardable_kernel(
            flash3_bwd_p,
            {"do": do, "q": do_q, "k": do_k, "v": do_v, "out": out, "lse": lse,
             "kv_lengths": kv_lengths},
            config=config,
        )
        return dq, dk, dv, None  # integer lengths: no cotangent

    _attn.defvjp(_attn_fwd, _attn_bwd)
    return _attn


def attention(q, k, v, config: TuningConfig, save_residuals: bool = False, kv_lengths=None):
    _ensure_flash3_supported(causal=config.causal, use_pipeline_emitter=False)
    if save_residuals:
        out, lse = _flash_forward(q, k, v, config, True, False, kv_lengths)
        return out, (lse,)
    return _make_flash_attention(config)(q, k, v, kv_lengths)


def attention_with_pipeline_emitter(
    q,
    k,
    v,
    config: TuningConfig,
    save_residuals: bool = False,
):
    _ensure_flash3_supported(causal=config.causal, use_pipeline_emitter=True)
    result = _flash_forward(q, k, v, config, save_residuals, True)
    if save_residuals:
        out, lse = result
        return out, (lse,)
    return result


def _validate_no_unsupported_sharding(arr: jax.Array, name: str) -> None:
    """Raise ``ValueError`` if *arr* is sharded on sequence (dim 1) or head_dim (dim 3).

    flash_attention3 delegates to JAX's built-in Mosaic GPU attention which does
    not accept user-controlled shard_map wrapping.  We therefore reject any
    NamedSharding that would cause XLA to insert all-gathers on unsupported axes.
    Batch (dim 0) and heads (dim 2) sharding are allowed (JAX handles them).
    """

    class _Shape:
        def __init__(self, sharding):
            self.sharding = sharding

    try:
        sharding = getattr(arr, "sharding", None)
        if sharding is None:
            import jax._src.core as core

            sharding = getattr(core.get_aval(arr), "sharding", None)
        spec_op = _FLASH_FWD_SPEC.operand("q")
        probe = KernelSpec(
            name=f"flash3:{name}",
            operands=(Operand(name, spec_op.dims),),
            outputs=(Output(spec_op.dims, dtype_like=name),),
            shardable=_FLASH_FWD_SPEC.shardable,
        )
        _validate_shardings(probe, (), [_Shape(sharding)])
    except ValueError:
        raise  # Re-raise our own validation errors.
    except Exception:
        pass  # Unable to inspect sharding; let JAX handle it.


def mha_flash(
    query,
    key,
    value,
    mask=None,
    bias=None,
    dropout_rng=None,
    dropout_rate: float = 0.0,
    broadcast_dropout: bool = True,
    deterministic: bool = True,
    dtype=None,
    precision=None,
    module=None,
    sm_scale: float | None = None,
    enable_gqa: bool = False,
    block_q: int | None = None,
    block_k: int | None = None,
    block_kv: int | None = None,
    max_concurrent_steps: int = 2,
    use_schedule_barrier: bool = True,
    causal: bool = False,
    compute_wgs_bwd: int | None = None,
    block_q_dkv: int | None = None,
    block_kv_dkv: int | None = None,
    block_q_dq: int | None = None,
    block_kv_dq: int | None = None,
    save_residuals: bool = False,
    use_pipeline_emitter: bool = False,
):
    # Kept for API compatibility with Flax attention_fn call sites.
    del module, precision, dropout_rng

    _ensure_flash3_supported(
        causal=causal,
        use_pipeline_emitter=use_pipeline_emitter,
    )

    from probjax.nn.pallas_kernels.attention_mask_bias import KVLenMask, NoMask

    kv_lengths = None
    if mask is not None and type(mask) is not NoMask:
        if type(mask) is not KVLenMask:
            raise NotImplementedError(
                "mha_flash supports mask=None, NoMask or KVLenMask (use causal=True for "
                f"causal attention), got {type(mask).__name__}.",
            )
        kv_lengths = jnp.asarray(mask.kv_lengths, jnp.int32)
    if bias is not None:
        raise NotImplementedError("mha_flash does not support additive attention bias.")

    if deterministic:
        dropout_rate = 0.0
    if dropout_rate != 0.0:
        raise NotImplementedError("mha_flash does not support dropout.")

    # Validate sharding on the original inputs BEFORE any transformations
    # (astype, unsqueeze) that could lose NamedSharding annotations.
    _validate_no_unsupported_sharding(query, "query")
    _validate_no_unsupported_sharding(key, "key")
    _validate_no_unsupported_sharding(value, "value")

    if dtype is not None:
        query = query.astype(dtype)
        key = key.astype(dtype)
        value = value.astype(dtype)

    added_batch_dim = query.ndim == 3
    if query.ndim == 3:
        query = query[None]
    if key.ndim == 3:
        key = key[None]
    if value.ndim == 3:
        value = value[None]

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"Expected 4D query/key/value, got {query.ndim=}, {key.ndim=}, {value.ndim=}.",
        )
    if key.shape != value.shape:
        raise ValueError(
            f"Expected key and value shapes to match, got {key.shape=} and {value.shape=}."
        )
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "Batch size and head dimension must match between query and key/value.",
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Expected matching dtypes, got {query.dtype=}, {key.dtype=}, {value.dtype=}.",
        )

    if not enable_gqa and query.shape[-2] != key.shape[-2]:
        raise ValueError(
            f"Expected same head count unless enable_gqa=True, got Hq={query.shape[-2]}, Hkv={key.shape[-2]}.",
        )
    if enable_gqa and (query.shape[-2] % key.shape[-2] != 0):
        raise ValueError(
            f"Expected Hq to be divisible by Hkv for GQA, got Hq={query.shape[-2]}, Hkv={key.shape[-2]}.",
        )

    # Tiles: when the caller sets none, use the configuration tuned on H100
    # (bf16, head_dim <= 128; docs/benchmarks/gpu_attention_h100_device.json): 64 x 128
    # forward tiles, backward dK/dV 128 x 64 and dQ 64 x 128 with two compute
    # warpgroups -- 2.4-4.1x faster fwd+bwd than 128 x 128 / one warpgroup, and
    # within 8-17% of cuDNN.  Deeper pipelines or 128 x 128 tiles with two
    # warpgroups exceed the 227 KB of shared memory at head_dim 128.
    untuned = (block_q, block_k, block_kv, compute_wgs_bwd, block_q_dkv, block_kv_dkv,
               block_q_dq, block_kv_dq)
    if all(x is None for x in untuned) and query.shape[-1] <= 128:
        block_q, block_kv, compute_wgs_bwd = 64, 128, 2
        block_q_dkv, block_kv_dkv, block_q_dq, block_kv_dq = 128, 64, 64, 128
    block_q = 128 if block_q is None else block_q
    block_k = 128 if block_k is None else block_k
    compute_wgs_bwd = 1 if compute_wgs_bwd is None else compute_wgs_bwd
    block_kv = block_k if block_kv is None else block_kv
    q_len = query.shape[1]
    kv_len = key.shape[1]
    head_dim = query.shape[-1]
    if (
        block_q_dkv is None
        and block_kv_dkv is None
        and block_q_dq is None
        and block_kv_dq is None
    ):
        block_q_dkv = block_q
        block_kv_dkv = block_kv
        block_q_dq = block_q
        block_kv_dq = block_kv
    # Sequence multiples the forward and both backward kernels need.
    if use_pipeline_emitter:
        q_multiple, kv_multiple = block_q, block_kv
    else:
        q_multiple = math.lcm(2 * block_q, block_q_dq * compute_wgs_bwd, block_q_dkv)
        kv_multiple = math.lcm(block_kv, block_kv_dkv * compute_wgs_bwd, block_kv_dq)
    q_pad = -q_len % q_multiple
    kv_pad = -kv_len % kv_multiple
    if causal and (q_pad or kv_pad or kv_lengths is not None):
        raise NotImplementedError(
            f"causal mha_flash needs {q_len=} / {kv_len=} to be multiples of "
            f"{q_multiple} / {kv_multiple} and no KV lengths."
        )
    if kv_pad:
        # Padded keys are masked in the kernel by the KV length.
        full = jnp.full((query.shape[0],), kv_len, jnp.int32)
        kv_lengths = full if kv_lengths is None else jnp.minimum(kv_lengths, full)
    if kv_lengths is not None and kv_lengths.shape != (query.shape[0],):
        raise ValueError(f"KVLenMask lengths must have shape (batch,), got {kv_lengths.shape}.")
    d_pad = -head_dim % 64

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    query = query * jnp.asarray(sm_scale, dtype=query.dtype)
    # Exact padding: extra query rows are sliced off; zero head-dim columns add
    # nothing to q.k and their (zero) value columns are sliced off.
    if q_pad or d_pad:
        query = jnp.pad(query, ((0, 0), (0, q_pad), (0, 0), (0, d_pad)))
    if kv_pad or d_pad:
        key = jnp.pad(key, ((0, 0), (0, kv_pad), (0, 0), (0, d_pad)))
        value = jnp.pad(value, ((0, 0), (0, kv_pad), (0, 0), (0, d_pad)))

    config = TuningConfig(
        block_q=block_q,
        block_kv=block_kv,
        max_concurrent_steps=max_concurrent_steps,
        use_schedule_barrier=use_schedule_barrier,
        causal=causal,
        compute_wgs_bwd=compute_wgs_bwd,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
    )

    out = (
        attention_with_pipeline_emitter(
            query,
            key,
            value,
            config=config,
            save_residuals=save_residuals,
        )
        if use_pipeline_emitter
        else attention(
            query,
            key,
            value,
            config=config,
            save_residuals=save_residuals,
            kv_lengths=kv_lengths,
        )
    )

    if save_residuals:
        out_value, residuals = out
        out_value = out_value[:, :q_len, :, :head_dim]
        residuals = tuple(r[..., :q_len] for r in residuals)  # lse: (B, H, Lq)
    else:
        out_value, residuals = out[:, :q_len, :, :head_dim], None

    if added_batch_dim:
        out_value = out_value[0]
        residuals = None if residuals is None else tuple(r[0] for r in residuals)
    return (out_value, residuals) if save_residuals else out_value
