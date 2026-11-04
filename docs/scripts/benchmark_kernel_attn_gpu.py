"""Lane ``attn-gpu``: GPU time *and* memory of softmax attention.

Run from the repository root (CUDA GPU required; torch optional)::

    python -m docs.scripts.benchmark_kernel_attn_gpu --device gpu --suite dispatch \
        --output docs/benchmarks/kernel_attn_gpu_dispatch.json
    python -m docs.scripts.benchmark_kernel_attn_gpu --device gpu --suite peak \
        --output docs/benchmarks/kernel_attn_gpu_memory.json

Suites (the full kernel-vs-baseline grid lives in ``benchmark_gpu_attention.py``
and the tile sweep in ``benchmark_gpu_attention_tiles.py``):

* ``dispatch`` -- small/medium sequence lengths typical of probjax models
  (large batch, small head dim): ``probjax_default`` (``dot_product_attention``,
  i.e. the default ``attention_fn``), ``probjax_flax`` (the Flax einsum path it
  used before, kept as the exact fallback), ``probjax_flex``, ``jax_cudnn``.
  Used to choose the GPU dispatch threshold of ``dot_product_attention``.
* ``peak`` -- true device peak memory (``peak_bytes_in_use`` read in a fresh
  subprocess per case, ``torch.cuda.max_memory_allocated`` for torch) for
  forward and forward+backward, plus the largest sequence length that still
  runs per implementation (doubling search, batch x heads fixed).
* ``residuals`` -- bytes saved for the backward pass (``jax.vjp`` residuals)
  per implementation.

Every timed row carries ``xla_temp_mb`` / ``xla_arg_mb`` / ``xla_out_mb`` from
``compiled.memory_analysis()``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from docs.scripts._gpu_bench import run_safely, time_jax, write_results

JAX_IMPLS = (
    "probjax_default",
    "probjax_flax",
    "probjax_flex",
    "probjax_mha",
    "jax_xla",
    "jax_cudnn",
)
TORCH_IMPLS = ("torch_flash", "torch_efficient", "torch_math")


def _attention(impl: str, causal: bool):
    from flax.nnx import dot_product_attention as flax_dpa

    from probjax.nn.layers.attention import dot_product_attention, flex_attention
    from probjax.nn.pallas_kernels import CausalMask, mha

    mask = CausalMask() if causal else None
    if impl == "probjax_default":
        return lambda q, k, v: dot_product_attention(q, k, v, mask=mask)
    if impl == "probjax_flax":

        def flax_fn(q, k, v):
            dense = None
            if causal:
                t = q.shape[-3]
                dense = jnp.tril(jnp.ones((t, t), bool))
            return flax_dpa(q, k, v, mask=dense)

        return flax_fn
    if impl == "probjax_flex":
        return lambda q, k, v: flex_attention(q, k, v, mask=mask)
    if impl == "probjax_mha":
        return lambda q, k, v: mha(
            q, k, v, mask=mask, sm_scale=1 / math.sqrt(q.shape[-1])
        )
    implementation = impl.removeprefix("jax_")
    return lambda q, k, v: jax.nn.dot_product_attention(
        q, k, v, is_causal=causal, implementation=implementation
    )


def _grad_fn(attn):
    def loss(q, k, v, g):
        return jnp.sum(attn(q, k, v).astype(jnp.float32) * g)

    return jax.grad(loss, argnums=(0, 1, 2))


def _inputs(batch, seq, heads, dim, dtype, seed=0):
    keys = jax.random.split(jax.random.key(seed), 4)
    shape = (batch, seq, heads, dim)
    q, k, v = (jax.random.normal(kk, shape, dtype) for kk in keys[:3])
    g = jax.random.normal(keys[3], shape, jnp.float32)
    return q, k, v, g


def rel_err(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def _mem_stats(fn, *args) -> dict:
    try:
        mem = fn.lower(*args).compile().memory_analysis()
        return dict(
            xla_temp_mb=mem.temp_size_in_bytes / 2**20,
            xla_arg_mb=mem.argument_size_in_bytes / 2**20,
            xla_out_mb=mem.output_size_in_bytes / 2**20,
        )
    except Exception:  # pragma: no cover - backend dependent
        return {}


# ---------------------------------------------------------------------------
# dispatch suite
# ---------------------------------------------------------------------------


def suite_dispatch(args, rows, record):
    for dtype_name, dim, seq, causal in itertools.product(
        args.dtypes, args.head_dims, args.seq_lens, map(bool, args.causal)
    ):
        dtype = jnp.dtype(dtype_name)
        q, k, v, g = _inputs(args.batch, seq, args.heads, dim, dtype, seed=seq + dim)
        f32 = [x.astype(jnp.float32) for x in (q, k, v)]
        with jax.default_matmul_precision("highest"):
            ref_attn = _attention("jax_xla", causal)
            ref_out = np.asarray(jax.jit(ref_attn)(*f32))
            ref_grads = [np.asarray(x) for x in jax.jit(_grad_fn(ref_attn))(*f32, g)]
        for mode, impl in itertools.product(args.modes, args.impls):
            if impl == "jax_cudnn" and dtype == jnp.float32:
                continue
            attn = _attention(impl, causal)

            def case(
                attn=attn,
                mode=mode,
                q=q,
                k=k,
                v=v,
                g=g,
                ref_out=ref_out,
                ref_grads=ref_grads,
            ):
                if mode == "fwd":
                    fn, fargs, refs = jax.jit(attn), (q, k, v), [ref_out]
                else:
                    fn, fargs, refs = jax.jit(_grad_fn(attn)), (q, k, v, g), ref_grads
                timing = time_jax(fn, *fargs)
                outs = fn(*fargs)
                outs = [outs] if mode == "fwd" else list(outs)
                err = max(rel_err(o, r) for o, r in zip(outs, refs, strict=True))
                return dict(rel_err=err, **_mem_stats(fn, *fargs), **timing)

            record(
                dict(
                    suite="dispatch",
                    impl=impl,
                    dtype=dtype_name,
                    head_dim=dim,
                    seq=seq,
                    causal=causal,
                    mode=mode,
                    batch=args.batch,
                    heads=args.heads,
                    **run_safely(case),
                )
            )


# ---------------------------------------------------------------------------
# residuals suite
# ---------------------------------------------------------------------------


def suite_residuals(args, rows, record):
    for dtype_name, dim, seq in itertools.product(
        args.dtypes, args.head_dims, args.seq_lens
    ):
        dtype = jnp.dtype(dtype_name)
        q, k, v, _ = _inputs(args.batch, seq, args.heads, dim, dtype)
        for impl in args.impls:
            if impl.startswith("torch") or (
                impl == "jax_cudnn" and dtype == jnp.float32
            ):
                continue
            attn = _attention(impl, False)

            def case(attn=attn, q=q, k=k, v=v):
                _, vjp_fn = jax.vjp(attn, q, k, v)
                leaves = jax.tree_util.tree_leaves(vjp_fn)
                # Inputs are residuals of every impl; count them separately.
                inputs = {id(x) for x in (q, k, v)}
                saved = [x for x in leaves if hasattr(x, "nbytes")]
                extra = [x for x in saved if id(x) not in inputs]
                return dict(
                    residual_mb=sum(x.nbytes for x in saved) / 2**20,
                    residual_non_input_mb=sum(x.nbytes for x in extra) / 2**20,
                    residual_shapes=[f"{x.dtype}{list(x.shape)}" for x in extra][:12],
                )

            record(
                dict(
                    suite="residuals",
                    impl=impl,
                    dtype=dtype_name,
                    head_dim=dim,
                    seq=seq,
                    batch=args.batch,
                    heads=args.heads,
                    **run_safely(case),
                )
            )


# ---------------------------------------------------------------------------
# peak suite (one fresh subprocess per case)
# ---------------------------------------------------------------------------


def _child(spec: dict) -> dict:
    impl, mode, causal = spec["impl"], spec["mode"], spec["causal"]
    dtype = spec["dtype"]
    shape = (spec["batch"], spec["seq"], spec["heads"], spec["head_dim"])
    if impl.startswith("torch"):
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend = {
            "torch_flash": SDPBackend.FLASH_ATTENTION,
            "torch_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "torch_math": SDPBackend.MATH,
        }[impl]
        tdtype = getattr(torch, dtype)
        b, t, h, d = shape
        q, k, v = (
            torch.randn(
                b, h, t, d, device="cuda", dtype=tdtype, requires_grad=mode == "fwd_bwd"
            )
            for _ in range(3)
        )
        g = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel(backend):
            if mode == "fwd":
                with torch.no_grad():
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                torch.autograd.grad((out.float() * g).sum(), (q, k, v))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        free, _total = torch.cuda.mem_get_info()
        return dict(
            peak_mb=peak / 2**20, inputs_mb=base / 2**20, device_free_mb=free / 2**20
        )

    q, k, v, g = _inputs(*shape, jnp.dtype(dtype))
    jax.block_until_ready((q, k, v, g))
    dev = jax.devices()[0]
    base = dev.memory_stats()["bytes_in_use"]
    attn = _attention(impl, causal)
    if mode == "fwd":
        fn, fargs = jax.jit(attn), (q, k, v)
    else:
        fn, fargs = jax.jit(_grad_fn(attn)), (q, k, v, g)
    stats = _mem_stats(fn, *fargs)
    jax.block_until_ready(fn(*fargs))
    mstats = dev.memory_stats()
    return dict(
        peak_mb=mstats["peak_bytes_in_use"] / 2**20,
        inputs_mb=base / 2**20,
        device_limit_mb=mstats.get("bytes_limit", 0) / 2**20,
        **stats,
    )


def _run_child(spec: dict, timeout: float = 600) -> dict:
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    cmd = [
        sys.executable,
        "-m",
        "docs.scripts.benchmark_kernel_attn_gpu",
        "--child",
        json.dumps(spec),
        "--output",
        os.devnull,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return dict(error="timeout")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("CHILD_RESULT "):
            return json.loads(line.removeprefix("CHILD_RESULT "))
    err = [ln for ln in proc.stderr.splitlines() if "Error" in ln or "error" in ln]
    msg = err[-1] if err else proc.stderr.strip().splitlines()[-1:] or ["?"]
    return dict(error=str(msg)[:300])


def suite_peak(args, rows, record):
    for dtype_name, dim, causal, mode, impl in itertools.product(
        args.dtypes, args.head_dims, map(bool, args.causal), args.modes, args.impls
    ):
        if impl in ("jax_cudnn", "torch_flash") and dtype_name == "float32":
            continue
        base = dict(
            suite="peak",
            impl=impl,
            dtype=dtype_name,
            head_dim=dim,
            causal=causal,
            mode=mode,
            batch=args.batch,
            heads=args.heads,
        )
        max_ok = None
        for seq in args.seq_lens:
            res = _run_child(base | dict(seq=seq))
            record(base | dict(seq=seq, **res))
            if "error" in res:
                break
            max_ok = seq
        if args.find_max and max_ok is not None and max_ok == args.seq_lens[-1]:
            seq = max_ok
            while seq < args.max_seq:
                seq *= 2
                res = _run_child(base | dict(seq=seq))
                record(base | dict(seq=seq, search=True, **res))
                if "error" in res:
                    break
                max_ok = seq
        record(base | dict(suite="peak_max_seq", max_seq_ok=max_ok))


# ---------------------------------------------------------------------------
# transformer suite (default attention_fn end to end)
# ---------------------------------------------------------------------------


def suite_transformer(args, rows, record):
    """``Transformer`` with its default ``attention_fn`` (forward + AdamW step).

    Run once from a checkout of the old code and once from the new one (same
    runner) to get before/after numbers for the default path.
    """
    from flax import nnx

    from docs.scripts.benchmark_gpu_nn import _bench_module
    from probjax.nn import Transformer

    for dtype_name, tokens in itertools.product(args.dtypes, args.seq_lens):
        dtype = jnp.dtype(dtype_name)
        meta = dict(
            suite="transformer",
            attention="probjax_default",
            dtype=dtype_name,
            tokens=tokens,
            batch=args.batch,
            model_dim=args.model_dim,
        )

        def build_and_run(meta=meta, tokens=tokens, dtype=dtype):
            model = Transformer(
                model_dim=args.model_dim,
                num_heads=8,
                num_layers=4,
                attn_size=args.model_dim // 8,
                dtype=dtype,
                rngs=nnx.Rngs(0),
            )
            x = jax.random.normal(
                jax.random.key(1), (args.batch, tokens, args.model_dim), dtype
            )
            sub = []
            _bench_module(sub, meta, model, (x,), lambda m, x: m(x))
            for row in sub:
                record(row)
            return {}

        err = run_safely(build_and_run)
        if err:
            record(dict(meta, **err))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--suite", choices=["dispatch", "peak", "residuals", "transformer"]
    )
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512]
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--modes", nargs="+", default=["fwd", "fwd_bwd"])
    parser.add_argument("--causal", nargs="+", type=int, default=[0])
    parser.add_argument("--impls", nargs="+", default=None)
    parser.add_argument("--find-max", action="store_true")
    parser.add_argument("--max-seq", type=int, default=1 << 18)
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child is not None:
        print("CHILD_RESULT " + json.dumps(_child(json.loads(args.child))), flush=True)
        return
    if args.device == "cpu":
        raise SystemExit("CPU attention is benchmarked by the attn-cpu lane.")
    if args.impls is None:
        args.impls = list(JAX_IMPLS + (TORCH_IMPLS if args.suite == "peak" else ()))

    rows = []
    if args.append and args.output.exists():
        rows = json.loads(args.output.read_text())["results"]

    def record(row):
        row = dict(row, tag=args.tag)
        rows.append(row)
        print(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
            flush=True,
        )

    suites = {
        "dispatch": suite_dispatch,
        "peak": suite_peak,
        "residuals": suite_residuals,
        "transformer": suite_transformer,
    }
    suites[args.suite](args, rows, record)
    config = vars(args) | {"output": str(args.output)}
    write_results(args.output, config, rows)


if __name__ == "__main__":
    main()
