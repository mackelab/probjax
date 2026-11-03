"""GPU attention benchmark: probjax Pallas kernels vs JAX and PyTorch attention.

Run from the repository root (requires a CUDA GPU; torch is optional)::

    python -m docs.scripts.benchmark_gpu_attention \
        --output docs/benchmarks/gpu_attention.json

Implementations (all receive ``[batch, seq, heads, head_dim]`` inputs, torch
gets the transposed ``[batch, heads, seq, head_dim]`` layout it expects):

* ``probjax_mha``     -- raw Pallas/Triton kernel ``probjax.nn.pallas_kernels.mha``
* ``probjax_flex``    -- layer-level ``probjax.nn.layers.attention.flex_attention``
                         (adds pow-2 padding + length masks for ragged shapes)
* ``probjax_default`` -- ``probjax.nn.layers.attention.dot_product_attention``,
                         the default ``attention_fn`` of probjax transformers
* ``jax_xla`` / ``jax_cudnn`` -- ``jax.nn.dot_product_attention(implementation=...)``
* ``torch_flash`` / ``torch_efficient`` / ``torch_math`` -- ``F.scaled_dot_product_attention``
  pinned to one backend with ``torch.nn.attention.sdpa_kernel``.

Every case checks the output against an fp32 XLA reference.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from docs.scripts._gpu_bench import (
    max_abs_err,
    run_safely,
    time_jax,
    time_torch,
    write_results,
)
from probjax.nn.layers.attention import dot_product_attention, flex_attention
from probjax.nn.pallas_kernels import CausalMask, mha

JAX_IMPLS = ("probjax_mha", "probjax_flex", "probjax_default", "jax_xla", "jax_cudnn")
TORCH_IMPLS = ("torch_flash", "torch_efficient", "torch_math")


def _jax_attention(impl: str, causal: bool, scale: float):
    if impl == "probjax_mha":
        mask = CausalMask() if causal else None
        return lambda q, k, v: mha(q, k, v, mask=mask, sm_scale=scale)
    if impl == "probjax_flex":
        mask = CausalMask() if causal else None
        return lambda q, k, v: flex_attention(q, k, v, mask=mask, sm_scale=scale)
    if impl == "probjax_default":
        mask = CausalMask() if causal else None
        # flax applies 1/sqrt(d) itself.
        return lambda q, k, v: dot_product_attention(q, k, v, mask=mask)
    implementation = impl.removeprefix("jax_")
    return lambda q, k, v: jax.nn.dot_product_attention(
        q, k, v, scale=scale, is_causal=causal, implementation=implementation
    )


def _loss_and_grad(attn):
    def loss(q, k, v, g):
        return jnp.sum(attn(q, k, v).astype(jnp.float32) * g)

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))


def _reference(q, k, v, g, causal, scale):
    # Batch elements are independent, so element 0 is an exact reference slice
    # and keeps the fp32 score matrix small at long sequence lengths.
    f32 = [x[:1].astype(jnp.float32) for x in (q, k, v)]
    g = g[:1]
    with jax.default_matmul_precision("highest"):
        attn = _jax_attention("jax_xla", causal, scale)
        out = jax.jit(attn)(*f32)
        grads = _loss_and_grad(attn)(*f32, g)
    return np.asarray(out), [np.asarray(x) for x in grads]


def _torch_case(impl, q, k, v, g, causal, scale, mode, ref_out, ref_grads):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = {
        "torch_flash": SDPBackend.FLASH_ATTENTION,
        "torch_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "torch_math": SDPBackend.MATH,
    }[impl]
    tdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        str(q.dtype)
    ]

    def to_torch(x):
        # [B, T, H, D] -> [B, H, T, D]
        return torch.from_numpy(np.asarray(x, np.float32)).to("cuda", tdtype).transpose(1, 2).contiguous()

    tq, tk, tv = (to_torch(x).requires_grad_(mode == "fwd_bwd") for x in (q, k, v))
    tg = to_torch(g).to(torch.float32)

    def fwd():
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(tq, tk, tv, is_causal=causal, scale=scale)

    def fwd_bwd():
        with sdpa_kernel(backend):
            out = F.scaled_dot_product_attention(tq, tk, tv, is_causal=causal, scale=scale)
        return torch.autograd.grad((out.float() * tg).sum(), (tq, tk, tv))

    fn = fwd if mode == "fwd" else fwd_bwd
    with torch.no_grad() if mode == "fwd" else torch.enable_grad():
        res = fn()
        if mode == "fwd":
            err = max_abs_err(res[:1].float().transpose(1, 2).cpu().numpy(), ref_out)
        else:
            err = max(
                max_abs_err(r[:1].float().transpose(1, 2).cpu().numpy(), ref)
                for r, ref in zip(res, ref_grads, strict=True)
            )
        torch.cuda.reset_peak_memory_stats()
        timing = time_torch(fn)
        peak = torch.cuda.max_memory_allocated() / 2**20
    del tq, tk, tv, tg
    torch.cuda.empty_cache()
    return dict(max_abs_err=err, peak_mb=peak, **timing)


def _jax_case(impl, q, k, v, g, causal, scale, mode, ref_out, ref_grads):
    attn = _jax_attention(impl, causal, scale)
    if mode == "fwd":
        fn = jax.jit(attn)
        args = (q, k, v)
    else:
        fn = _loss_and_grad(attn)
        args = (q, k, v, g)
    timing = time_jax(fn, *args)  # first: so compile_s includes compilation
    if mode == "fwd":
        err = max_abs_err(fn(*args)[:1], ref_out)
    else:
        res = fn(*args)
        err = max(max_abs_err(r[:1], ref) for r, ref in zip(res, ref_grads, strict=True))
    stats = {}
    try:
        mem = fn.lower(*args).compile().memory_analysis()
        stats["xla_temp_mb"] = mem.temp_size_in_bytes / 2**20
    except Exception:
        pass
    return dict(max_abs_err=err, **stats, **timing)


def attention_flops(batch, heads, q_len, kv_len, dim, causal, mode):
    flops = 4 * batch * heads * q_len * kv_len * dim
    if causal:
        flops /= 2
    if mode == "fwd_bwd":
        flops *= 3.5  # fwd + bwd (2.5x fwd, incl. recompute) -- FlashAttention convention
    return flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument("--ragged-seq-lens", type=int, nargs="+", default=[600, 1500, 3000])
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--modes", nargs="+", default=["fwd", "fwd_bwd"])
    parser.add_argument("--causal", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--impls", nargs="+", default=list(JAX_IMPLS + TORCH_IMPLS))
    args = parser.parse_args()

    try:
        import torch  # noqa: F401

        have_torch = torch.cuda.is_available()
    except Exception:
        have_torch = False

    rows = []
    seq_lens = sorted(set(args.seq_lens) | set(args.ragged_seq_lens))
    for dtype_name in args.dtypes:
        dtype = jnp.dtype(dtype_name)
        for dim in args.head_dims:
            for seq in seq_lens:
                for causal in map(bool, args.causal):
                    keys = jax.random.split(jax.random.key(seq * 7 + dim), 4)
                    shape = (args.batch, seq, args.heads, dim)
                    q, k, v = (jax.random.normal(kk, shape, dtype) for kk in keys[:3])
                    g = jax.random.normal(keys[3], shape, jnp.float32)
                    scale = 1.0 / math.sqrt(dim)
                    ref_out, ref_grads = _reference(q, k, v, g, causal, scale)
                    for mode in args.modes:
                        for impl in args.impls:
                            if impl.startswith("torch") and not have_torch:
                                continue
                            if impl == "jax_cudnn" and dtype == jnp.float32:
                                continue  # cuDNN flash path needs fp16/bf16
                            if impl == "torch_flash" and dtype == jnp.float32:
                                continue  # FlashAttention-2 needs fp16/bf16
                            if impl == "torch_math" and seq > 4096:
                                continue  # materializes the score matrix
                            case = _torch_case if impl.startswith("torch") else _jax_case
                            res = run_safely(
                                lambda case=case, impl=impl, mode=mode: case(
                                    impl, q, k, v, g, causal, scale, mode,
                                    ref_out, ref_grads,
                                )
                            )
                            row = dict(
                                impl=impl, dtype=dtype_name, head_dim=dim, seq=seq,
                                causal=causal, mode=mode, batch=args.batch,
                                heads=args.heads, **res,
                            )
                            if "median_ms" in res:
                                row["tflops"] = (
                                    attention_flops(args.batch, args.heads, seq, seq, dim, causal, mode)
                                    / (res["median_ms"] * 1e-3) / 1e12
                                )
                            rows.append(row)
                            print(
                                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                                flush=True,
                            )
                    write_results(args.output, vars(args) | {"output": str(args.output)}, rows)
    write_results(args.output, vars(args) | {"output": str(args.output)}, rows)


if __name__ == "__main__":
    main()
