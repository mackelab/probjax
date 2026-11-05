"""Device-specific attention defaults.

``flex_attention`` and ``dot_product_attention`` consult this table when the
caller left the kernel choice at its defaults.  Entries are keyed by the CUDA
compute-capability major version; devices without an entry keep the generic
rules (shared-memory based tiles in ``pallas_kernels.kernels.attention``).

sm_90 (H100) -- measured on an H100 NVL (94 GB), bf16, JAX 0.11.2, cuDNN 9.27
(docs/benchmarks/gpu_attention_h100_device.json, gpu_attention_h100.json):

* cuDNN's fused attention (``jax.nn.dot_product_attention(implementation=
  "cudnn")``) is 1.2-2.1x faster than the best Triton backward once the work
  B * H * Lq * Lkv reaches ~2**26 for head_dim >= 64 and ~2**30 for head_dim
  32, except for query-heavy attention over very short KV (<= 256 keys,
  Lq >= 16 * Lkv) and for <= 64 queries over >= 16x as many keys (split-KV
  forward + single-pass backward, 6-12% faster), where Triton wins.  It is exact for no mask, a per-batch KV
  length mask and causal attention with equal lengths -- the only cases routed.
* For the Triton backward, ``"auto"`` resolves to the split (dQ / dK-dV)
  kernels: the fused kernel was up to 1.9x slower on asymmetric lengths and at
  most 4% faster elsewhere; the single-pass kernel wins only for short queries
  over long KV.

Set ``PROBJAX_ATTENTION_DEVICE_DEFAULTS=0`` to disable the table (generic rules
everywhere) or ``PROBJAX_ATTENTION_CUDNN=0`` to keep Triton but use the tuned
backward choice.
"""

from __future__ import annotations

import dataclasses
import functools
import os

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class DeviceAttentionDefaults:
    # Minimum B * H * Lq * Lkv for the cuDNN route, keyed by padded head_dim.
    cudnn_min_work: dict[int, int]
    # Query-heavy attention over short KV stays on Triton:
    # Lkv <= cudnn_short_kv and Lq >= cudnn_short_kv_ratio * Lkv.
    cudnn_short_kv: int
    cudnn_short_kv_ratio: int
    # Very few queries over long KV stay on Triton (split-KV forward + single-pass
    # backward): Lq <= cudnn_min_q and Lkv >= onepass_kv_ratio * Lq.
    cudnn_min_q: int
    # "auto" backward: split, except single-pass when Lq <= onepass_max_q and
    # Lkv >= onepass_kv_ratio * Lq.
    onepass_max_q: int
    onepass_kv_ratio: int


DEVICE_ATTENTION_DEFAULTS: dict[int, DeviceAttentionDefaults] = {
    9: DeviceAttentionDefaults(
        cudnn_min_work={32: 2**30, 64: 2**26, 128: 2**26},
        cudnn_short_kv=256,
        cudnn_short_kv_ratio=16,
        cudnn_min_q=64,
        onepass_max_q=256,
        onepass_kv_ratio=16,
    ),
}


def _env_off(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() in ("0", "false", "no", "off")


@functools.lru_cache(maxsize=None)
def _compute_capability_major() -> int | None:
    try:
        if jax.default_backend() != "gpu":
            return None
        cc = getattr(jax.devices()[0], "compute_capability", None)
    except RuntimeError:
        return None
    if cc is None:
        return None
    return int(cc[0]) if isinstance(cc, tuple) else int(float(cc))


def device_attention_defaults() -> DeviceAttentionDefaults | None:
    """The tuned defaults for the default device, or None (generic rules)."""
    if _env_off("PROBJAX_ATTENTION_DEVICE_DEFAULTS"):
        return None
    major = _compute_capability_major()
    return None if major is None else DEVICE_ATTENTION_DEFAULTS.get(major)


def resolve_auto_backward(q_len: int, kv_len: int, dtype_bytes: int = 2) -> str | None:
    """Device choice for ``backward_pass_impl="auto"``; None keeps the generic rule.

    The single-pass backward is only chosen for 16-bit inputs (the measured
    case); fp32 keeps the deterministic split kernels.
    """
    d = device_attention_defaults()
    if d is None:
        return None
    if (
        dtype_bytes < 4
        and q_len <= d.onepass_max_q
        and kv_len >= d.onepass_kv_ratio * q_len
    ):
        return "triton_onepass"
    return "triton_split"


def _cudnn_lengths(mask, q_len: int, kv_len: int):
    """(kv_lengths, causal) when ``mask`` is exactly expressible in cuDNN, else None."""
    from probjax.nn.pallas_kernels import CausalMask, KVLenMask, NoMask

    if mask is None or type(mask) is NoMask:
        return None, False
    if type(mask) is KVLenMask:
        return jnp.asarray(mask.kv_lengths, jnp.int32), False
    if type(mask) is CausalMask and q_len == kv_len:
        return None, True
    return "unsupported", False


def _fused_eligible(query, key, value, *, mask, bias, dropout_rate: float):
    """Shape/dtype/mask checks shared by the cuDNN and Mosaic routes.

    Returns ``(B, l_q, h_q, dim, l_kv, kv_lengths, causal)`` or None.
    """
    if bias is not None or dropout_rate > 0.0:
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return None
    if not (query.dtype == key.dtype == value.dtype) or query.dtype not in (
        jnp.bfloat16,
        jnp.float16,
    ):
        return None
    B, l_q, h_q, dim = query.shape
    _, l_kv, h_kv, dim_k = key.shape
    if dim != dim_k or value.shape[-1] != dim or value.shape[:-1] != key.shape[:-1]:
        return None
    if h_q % h_kv:
        return None
    lengths = _cudnn_lengths(mask, l_q, l_kv)
    if isinstance(lengths[0], str):
        return None
    kv_lengths, causal = lengths
    if kv_lengths is not None and kv_lengths.shape != (B,):
        return None
    return B, l_q, h_q, dim, l_kv, kv_lengths, causal


def _worth_fusing(d: DeviceAttentionDefaults, B, l_q, h_q, dim, l_kv) -> bool:
    """Work threshold and the Triton-favoured regions (measured on H100)."""
    bd = max(32, 1 << (dim - 1).bit_length())
    min_work = d.cudnn_min_work.get(bd)
    if min_work is None or B * h_q * l_q * l_kv < min_work:
        return False
    if l_kv <= d.cudnn_short_kv and l_q >= d.cudnn_short_kv_ratio * l_kv:
        return False
    if l_q <= d.cudnn_min_q and l_kv >= d.onepass_kv_ratio * l_q:
        return False
    return True


def cudnn_attention_or_none(
    query, key, value, *, mask, bias, dropout_rate: float, sm_scale: float,
    force: bool = False,
):
    """cuDNN fused attention when the device table routes this call there.

    Inputs are ``[B, T, H, D]`` (K/V may have fewer heads, GQA).  Returns the
    output, or None to fall through (``force=True``: skip the device table and
    the thresholds, only check that cuDNN computes the call exactly).
    """
    d = device_attention_defaults()
    if not force and (d is None or _env_off("PROBJAX_ATTENTION_CUDNN")):
        return None
    e = _fused_eligible(query, key, value, mask=mask, bias=bias, dropout_rate=dropout_rate)
    if e is None:
        return None
    B, l_q, h_q, dim, l_kv, kv_lengths, causal = e
    if dim % 8 or dim > 128:
        return None
    if not force and not _worth_fusing(d, B, l_q, h_q, dim, l_kv):
        return None
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        scale=sm_scale,
        is_causal=causal,
        key_value_seq_lengths=kv_lengths,
        implementation="cudnn",
    )


def mosaic_attention_or_none(
    query, key, value, *, mask, bias, dropout_rate: float, sm_scale: float,
    force: bool = False,
):
    """Mosaic GPU (Hopper FlashAttention-3, ``mha_flash``) when routed there.

    On sm_90 default calls try cuDNN first; this covers what cuDNN does not
    take (head_dim not a multiple of 8, ``PROBJAX_ATTENTION_CUDNN=0``) with the
    same masks: none, ``KVLenMask`` and equal-length causal (unpadded).  Keys
    and queries of any length are padded exactly.  ``force=True`` skips the
    device table and thresholds.
    """
    d = device_attention_defaults()
    if not force and (d is None or _env_off("PROBJAX_ATTENTION_MOSAIC")):
        return None
    if _compute_capability_major() != 9:
        return None
    e = _fused_eligible(query, key, value, mask=mask, bias=bias, dropout_rate=dropout_rate)
    if e is None:
        return None
    B, l_q, h_q, dim, l_kv, kv_lengths, causal = e
    # head_dim <= 32 is padded to 64 (twice the work): Triton is faster there
    # (ISAB 256 x 32768, d32, KV lengths: Mosaic 16.4 ms, Triton 13.6 ms).
    if dim > 128 or (dim <= 32 and not force):
        return None
    if causal and (l_q % 128 or l_kv % 128):
        return None
    if not force and not _worth_fusing(d, B, l_q, h_q, dim, l_kv):
        return None
    from probjax.nn.pallas_kernels import KVLenMask
    from probjax.nn.pallas_kernels.kernels.flash_attention3 import mha_flash

    return mha_flash(
        query,
        key,
        value,
        mask=None if kv_lengths is None else KVLenMask(kv_lengths),
        sm_scale=sm_scale,
        enable_gqa=query.shape[-2] != key.shape[-2],
        causal=causal,
    )
