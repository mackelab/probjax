from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp

try:
    from jax._src.lib import cuda_versions
except Exception:  # pragma: no cover - depends on local jax build
    cuda_versions = None

try:
    from jax.experimental.pallas.ops.gpu import attention_mgpu as _attention_mgpu
    from jax.experimental.pallas.ops.gpu.attention_mgpu import (
        TuningConfig,
    )
    from jax.experimental.pallas.ops.gpu.attention_mgpu import (
        attention as _attention_impl,
    )
    from jax.experimental.pallas.ops.gpu.attention_mgpu import (
        attention_with_pipeline_emitter as _attention_with_pipeline_emitter_impl,
    )

    _FLASH3_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local jax build
    _attention_mgpu = None
    TuningConfig = Any  # type: ignore[assignment]
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


def _use_custom_partitioning() -> bool:
    from jax._src import mesh as mesh_lib

    mesh = mesh_lib.get_concrete_mesh()
    return not mesh.empty and mesh.size > 1


def _validate_flash_sharding(sharding, name: str) -> None:
    spec = getattr(sharding, "spec", None)
    if spec is None or len(spec) == 0:
        return

    for dim_idx, axis in enumerate(spec):
        if axis is None:
            continue
        if dim_idx == 1:
            raise ValueError(
                f"flash_attention3 does not support sharding on the sequence "
                f"dimension (dim 1) of `{name}`. Got PartitionSpec{tuple(spec)} "
                f"which shards dim 1 over mesh axis '{axis}'."
            )
        if dim_idx == 3:
            raise ValueError(
                f"flash_attention3 does not support sharding on the head_dim "
                f"dimension (dim 3) of `{name}`. Got PartitionSpec{tuple(spec)} "
                f"which shards dim 3 over mesh axis '{axis}'."
            )


def _result_sharding_rule(*, include_lse: bool) -> tuple[str, tuple[str, ...]]:
    rule = (
        "batch seq heads head_dim, "
        "batch seq heads head_dim, "
        "batch seq heads head_dim -> "
        "batch seq heads head_dim"
    )
    if include_lse:
        rule += ", batch heads seq"
    return rule, ("seq", "head_dim")


def _run_flash_forward_raw(
    q,
    k,
    v,
    *,
    config: TuningConfig,
    save_residuals: bool,
    use_pipeline_emitter: bool,
):
    if use_pipeline_emitter:
        result = _attention_with_pipeline_emitter_impl(
            q,
            k,
            v,
            config=config,
            save_residuals=save_residuals,
        )
    elif _attention_forward_impl is not None:
        result = _attention_forward_impl(q, k, v, config, save_residuals)
    else:
        result = _attention_impl(q, k, v, config=config, save_residuals=save_residuals)

    if save_residuals:
        out, (lse,) = result
        return out, lse
    return result


def _run_flash_backward_raw(
    do,
    q,
    k,
    v,
    out,
    lse,
    *,
    config: TuningConfig,
):
    if _attention_backward_impl is None:
        _, vjp_fun = jax.vjp(
            lambda q, k, v: _attention_impl(
                q,
                k,
                v,
                config=config,
                save_residuals=False,
            ),
            q,
            k,
            v,
        )
        return vjp_fun(do)
    return _attention_backward_impl(config, False, (q, k, v, out, lse), do)


@functools.lru_cache(maxsize=None)
def _make_cp_flash_forward(
    config: TuningConfig,
    save_residuals: bool,
    use_pipeline_emitter: bool,
):
    from jax.experimental.custom_partitioning import custom_partitioning

    rule, repl = _result_sharding_rule(include_lse=save_residuals)

    def _call_raw(q, k, v):
        return _run_flash_forward_raw(
            q,
            k,
            v,
            config=config,
            save_residuals=save_residuals,
            use_pipeline_emitter=use_pipeline_emitter,
        )

    @custom_partitioning
    def _fwd(q, k, v):
        return _call_raw(q, k, v)

    def _partition(mesh, arg_shapes, result_shape):
        for shape, name in zip(jax.tree.leaves(arg_shapes), ("q", "k", "v")):
            _validate_flash_sharding(shape.sharding, name)
        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(q, k, v):
            return _call_raw(q, k, v)

        return mesh, lower_fn, result_shardings, arg_shardings

    _partition._cp_raw_fn = _call_raw  # type: ignore[attr-defined]

    _fwd.def_partition(
        partition=_partition,
        sharding_rule=rule,
        need_replication_factors=repl,
    )
    return _fwd


@functools.lru_cache(maxsize=None)
def _make_cp_flash_backward(config: TuningConfig):
    from jax.experimental.custom_partitioning import custom_partitioning

    def _call_bwd_raw(do, q, k, v, out, lse):
        return _run_flash_backward_raw(do, q, k, v, out, lse, config=config)

    @custom_partitioning
    def _bwd(do, q, k, v, out, lse):
        return _call_bwd_raw(do, q, k, v, out, lse)

    def _partition(mesh, arg_shapes, result_shape):
        for shape, name in zip(jax.tree.leaves(arg_shapes)[1:4], ("q", "k", "v")):
            _validate_flash_sharding(shape.sharding, name)
        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(do, q, k, v, out, lse):
            return _call_bwd_raw(do, q, k, v, out, lse)

        return mesh, lower_fn, result_shardings, arg_shardings

    _partition._cp_raw_fn = _call_bwd_raw  # type: ignore[attr-defined]

    _bwd.def_partition(
        partition=_partition,
        sharding_rule=(
            "batch seq heads head_dim, "
            "batch seq heads head_dim, "
            "batch seq heads head_dim, "
            "batch seq heads head_dim, "
            "batch seq heads head_dim, "
            "batch heads seq -> "
            "batch seq heads head_dim, "
            "batch seq heads head_dim, "
            "batch seq heads head_dim"
        ),
        need_replication_factors=("seq", "head_dim"),
    )
    return _bwd


@functools.lru_cache(maxsize=None)
def _make_flash_attention(config: TuningConfig):
    @jax.custom_vjp
    def _attn(q, k, v):
        if _use_custom_partitioning():
            return _make_cp_flash_forward(
                config,
                False,
                False,
            )(q, k, v)
        return _run_flash_forward_raw(
            q,
            k,
            v,
            config=config,
            save_residuals=False,
            use_pipeline_emitter=False,
        )

    def _attn_fwd(q, k, v):
        if _use_custom_partitioning():
            out, lse = _make_cp_flash_forward(
                config,
                True,
                False,
            )(q, k, v)
        else:
            out, lse = _run_flash_forward_raw(
                q,
                k,
                v,
                config=config,
                save_residuals=True,
                use_pipeline_emitter=False,
            )
        return out, (q, k, v, out, lse)

    def _attn_bwd(res, do):
        q, k, v, out, lse = res
        if _use_custom_partitioning():
            return _make_cp_flash_backward(config)(do, q, k, v, out, lse)
        return _run_flash_backward_raw(do, q, k, v, out, lse, config=config)

    _attn.defvjp(_attn_fwd, _attn_bwd)
    return _attn


def attention(q, k, v, config: TuningConfig, save_residuals: bool = False):
    _ensure_flash3_supported(causal=config.causal, use_pipeline_emitter=False)
    if save_residuals:
        if _use_custom_partitioning():
            out, lse = _make_cp_flash_forward(config, True, False)(q, k, v)
        else:
            out, lse = _run_flash_forward_raw(
                q,
                k,
                v,
                config=config,
                save_residuals=True,
                use_pipeline_emitter=False,
            )
        return out, (lse,)
    return _make_flash_attention(config)(q, k, v)


def attention_with_pipeline_emitter(
    q,
    k,
    v,
    config: TuningConfig,
    save_residuals: bool = False,
):
    _ensure_flash3_supported(causal=config.causal, use_pipeline_emitter=True)
    if _use_custom_partitioning():
        result = _make_cp_flash_forward(config, save_residuals, True)(q, k, v)
    else:
        result = _run_flash_forward_raw(
            q,
            k,
            v,
            config=config,
            save_residuals=save_residuals,
            use_pipeline_emitter=True,
        )
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
    try:
        sharding = getattr(arr, "sharding", None)
        if sharding is not None:
            _validate_flash_sharding(sharding, name)
            return
        import jax._src.core as core

        aval = core.get_aval(arr)
        _validate_flash_sharding(getattr(aval, "sharding", None), name)
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
    block_q: int = 128,
    block_k: int = 128,
    block_kv: int | None = None,
    max_concurrent_steps: int = 2,
    use_schedule_barrier: bool = True,
    causal: bool = False,
    compute_wgs_bwd: int = 1,
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

    if mask is not None:
        raise NotImplementedError(
            "mha_flash currently supports mask=None only; use causal=True for causal attention.",
        )
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

    block_kv = block_k if block_kv is None else block_kv
    q_len = query.shape[1]
    kv_len = key.shape[1]
    q_multiple = block_q if use_pipeline_emitter else (2 * block_q)
    if q_len % q_multiple:
        raise ValueError(
            f"{q_len=} must be a multiple of {q_multiple=} for this kernel."
        )
    if kv_len % block_kv:
        raise ValueError(f"{kv_len=} must be a multiple of {block_kv=}.")

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

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(query.shape[-1])
    query = query * jnp.asarray(sm_scale, dtype=query.dtype)

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
        )
    )

    if added_batch_dim:
        if save_residuals:
            out_value, residuals = out
            return out_value[0], tuple(r[0] for r in residuals)
        return out[0]
    return out
