"""Lane ``attn-ext``: native GQA/MQA and split-KV decoding for softmax attention.

Run from the repository root, holding the shared bench lock::

    python -m docs.scripts.benchmark_kernel_attn_ext --device gpu --suite gqa \
        --output docs/benchmarks/kernel_attn_ext_gpu.json
    python -m docs.scripts.benchmark_kernel_attn_ext --device gpu --suite decode \
        --output docs/benchmarks/kernel_attn_ext_gpu.json --append
    JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_attn_ext \
        --device cpu --suite gqa --output docs/benchmarks/kernel_attn_ext_cpu.json

Suites (inputs ``q [B, Lq, Hq, D]``, ``k/v [B, Lkv, Hkv, D]``):

* ``gqa`` -- training-shaped GQA/MQA (``Lq == Lkv``), forward and
  forward+backward. ``flex_native`` is ``flex_attention(enable_gqa=True)``
  (kernels index KV head ``h // G``, no copy); ``flex_repeat`` is the previous
  behaviour (``jnp.repeat`` K/V to ``Hq`` heads, then the kernel). Baselines:
  ``jax_xla`` / ``jax_cudnn`` (``jax.nn.dot_product_attention``, native GQA) and
  ``torch_sdpa`` (``enable_gqa=True``). Rows carry ``rel_err`` vs an fp32
  reference (out and dq/dk/dv), ``rel_vs_repeat`` (native vs repeat path) and
  ``residual_mb`` (bytes saved for the backward, inputs excluded).
* ``decode`` -- few queries, long KV: ``Lq in {1, 4, 16}``,
  ``Lkv in {4k, 16k, 64k}``, ``B in {1, 8}``. ``probjax_splitkv`` is the default
  ``flex_attention`` (auto split-KV), ``probjax_nosplit`` the regular forward
  (``split_kv=False``), plus ``jax_xla``, ``jax_cudnn`` and ``torch_sdpa``.
* ``splitsweep`` -- the split-KV kernel with a fixed number of splits (and the
  regular forward as ``num_splits=0``), used to set the dispatch heuristic.
* ``peak`` -- device peak memory in a fresh subprocess per case
  (``peak_bytes_in_use``; ``torch.cuda.max_memory_allocated`` for torch).

Timing: every case runs ``--repeats`` interleaved rounds (all implementations
of a shape, then again), each round a ``time_jax``/``time_torch`` median with
IQR. Every JAX row also records ``xla_{temp,arg,out}_mb`` from
``compiled.memory_analysis()``.
"""

from __future__ import annotations

import argparse
import functools
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

from docs.scripts._gpu_bench import run_safely, time_jax, time_torch, write_results

MB = 2**20


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _jax_attention(impl: str):
    from probjax.nn.layers.attention import flex_attention

    if impl == "flex_native":
        return lambda q, k, v: flex_attention(q, k, v, enable_gqa=True)
    if impl == "flex_repeat":

        def repeat(q, k, v):
            g = q.shape[2] // k.shape[2]
            k, v = (jnp.repeat(x, g, axis=2) for x in (k, v))
            return flex_attention(q, k, v, split_kv=False)

        return repeat
    if impl == "probjax_splitkv":
        return lambda q, k, v: flex_attention(q, k, v, enable_gqa=True)
    if impl == "probjax_nosplit":
        return lambda q, k, v: flex_attention(q, k, v, enable_gqa=True, split_kv=False)
    if impl.startswith("jax_"):
        implementation = impl.removeprefix("jax_")
        return lambda q, k, v: jax.nn.dot_product_attention(
            q, k, v, implementation=implementation
        )
    raise ValueError(impl)


def _grad_fn(attn):
    def loss(q, k, v, g):
        return jnp.sum(attn(q, k, v).astype(jnp.float32) * g)

    return jax.grad(loss, argnums=(0, 1, 2))


def _torch_fn(mode: str):
    import torch
    import torch.nn.functional as F

    def fwd(q, k, v, g):
        with torch.no_grad():
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    def fwd_bwd(q, k, v, g):
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return torch.autograd.grad((out.float() * g).sum(), (q, k, v))

    return fwd if mode == "fwd" else fwd_bwd


def _torch_inputs(q, k, v, g, mode, device):
    import torch

    def conv(x, grad):
        t = torch.from_numpy(np.asarray(x.astype(jnp.float32))).to(device)
        t = t.to(getattr(torch, str(x.dtype))).transpose(1, 2).contiguous()
        return t.requires_grad_(grad)

    grad = mode == "fwd_bwd"
    return (conv(q, grad), conv(k, grad), conv(v, grad), conv(g, False))


def _inputs(b, lq, lkv, hq, hkv, d, dtype, seed=0):
    keys = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(keys[0], (b, lq, hq, d), jnp.float32).astype(dtype)
    k = jax.random.normal(keys[1], (b, lkv, hkv, d), jnp.float32).astype(dtype)
    v = jax.random.normal(keys[2], (b, lkv, hkv, d), jnp.float32).astype(dtype)
    g = jax.random.normal(keys[3], (b, lq, hq, d), jnp.float32)
    return q, k, v, g


def _reference(q, k, v, g, mode):
    """fp32 dense reference (repeat convention), out or (dq, dk, dv)."""

    def attn(q, k, v):
        rep = q.shape[2] // k.shape[2]
        k, v = (jnp.repeat(x, rep, axis=2) for x in (k, v))
        s = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
        return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(s, axis=-1), v)

    f32 = [x.astype(jnp.float32) for x in (q, k, v)]
    with jax.default_matmul_precision("highest"):
        if mode == "fwd":
            return [np.asarray(jax.jit(attn)(*f32))]
        return [np.asarray(x) for x in jax.jit(_grad_fn(attn))(*f32, g)]


def rel_err(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _mem_stats(fn, *args) -> dict:
    try:
        mem = fn.lower(*args).compile().memory_analysis()
        return dict(
            xla_temp_mb=mem.temp_size_in_bytes / MB,
            xla_arg_mb=mem.argument_size_in_bytes / MB,
            xla_out_mb=mem.output_size_in_bytes / MB,
        )
    except Exception:  # pragma: no cover - backend dependent
        return {}


def _residual_mb(attn, q, k, v) -> dict:
    _, vjp_fn = jax.vjp(attn, q, k, v)
    inputs = {id(x) for x in (q, k, v)}
    saved = [x for x in jax.tree_util.tree_leaves(vjp_fn) if hasattr(x, "nbytes")]
    extra = [x for x in saved if id(x) not in inputs]
    return dict(
        residual_mb=sum(x.nbytes for x in extra) / MB,
        residual_shapes=[f"{x.dtype}{list(x.shape)}" for x in extra][:8],
    )


def _skip(impl, dtype_name, device, hq, hkv) -> bool:
    return impl == "jax_cudnn" and (device == "cpu" or dtype_name == "float32")


# ---------------------------------------------------------------------------
# Generic interleaved timing over one shape
# ---------------------------------------------------------------------------


def _bench_shape(args, record, meta, q, k, v, g, impls, refs_for, extra=None):
    """Time ``impls`` for both modes on one shape, ``args.repeats`` rounds."""
    for mode in args.modes:
        refs = refs_for(mode)
        cases = {}
        outs_by_impl = {}
        for impl in impls:
            if _skip(impl, meta["dtype"], args.device, meta["hq"], meta["hkv"]):
                continue
            if impl == "torch_sdpa":

                def setup(mode=mode):
                    import torch

                    dev = "cuda" if args.device == "gpu" else "cpu"
                    targs = _torch_inputs(q, k, v, g, mode, dev)
                    fn = _torch_fn(mode)
                    out = fn(*targs)
                    out = [out] if mode == "fwd" else list(out)
                    outs = [
                        x.detach().float().transpose(1, 2).cpu().numpy() for x in out
                    ]
                    info = {}
                    if args.device == "gpu":
                        torch.cuda.synchronize()
                    return fn, targs, outs, info

            else:
                attn = _jax_attention(impl)

                def setup(attn=attn, mode=mode, impl=impl):
                    if mode == "fwd":
                        fn, fargs = jax.jit(attn), (q, k, v)
                    else:
                        fn, fargs = jax.jit(_grad_fn(attn)), (q, k, v, g)
                    out = fn(*fargs)
                    out = [out] if mode == "fwd" else list(out)
                    info = _mem_stats(fn, *fargs)
                    if mode == "fwd_bwd" and impl.startswith("flex"):
                        info.update(_residual_mb(attn, q, k, v))
                    return fn, fargs, [np.asarray(x) for x in out], info

            res = run_safely(lambda setup=setup: dict(state=setup()))
            if "error" in res:
                record(dict(meta, impl=impl, mode=mode, **res))
                continue
            cases[impl] = res["state"]
            outs_by_impl[impl] = res["state"][2]
        info_by_impl = {}
        for impl, (_fn, _fargs, outs, info) in cases.items():
            errs = {}
            if refs is not None:
                errs["rel_err"] = max(
                    rel_err(o, r) for o, r in zip(outs, refs, strict=True)
                )
                errs["rel_err_parts"] = [
                    rel_err(o, r) for o, r in zip(outs, refs, strict=True)
                ]
            base = outs_by_impl.get(args.baseline_impl)
            if base is not None and impl != args.baseline_impl:
                errs[f"rel_vs_{args.baseline_impl}"] = max(
                    rel_err(o, r) for o, r in zip(outs, base, strict=True)
                )
            info_by_impl[impl] = dict(info, **errs)
        for rnd in range(args.repeats):
            for impl, (fn, fargs, _outs, _info) in cases.items():
                timer = time_torch if impl == "torch_sdpa" else time_jax
                timing = run_safely(lambda fn=fn, fargs=fargs, t=timer: t(fn, *fargs))
                record(
                    dict(
                        meta,
                        impl=impl,
                        mode=mode,
                        round=rnd,
                        **(extra or {}),
                        **info_by_impl[impl],
                        **timing,
                    )
                )
        cases.clear()
        jax.clear_caches()


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def suite_gqa(args, rows, record):
    for dtype_name, d, t, hkv in itertools.product(
        args.dtypes, args.head_dims, args.seq_lens, args.kv_heads
    ):
        dtype = jnp.dtype(dtype_name)
        b, hq = args.batch, args.heads
        q, k, v, g = _inputs(b, t, t, hq, hkv, d, dtype, seed=t + d + hkv)
        meta = dict(
            suite="gqa",
            dtype=dtype_name,
            batch=b,
            lq=t,
            lkv=t,
            hq=hq,
            hkv=hkv,
            head_dim=d,
        )

        def refs_for(mode, q=q, k=k, v=v, g=g, t=t):
            if t > args.ref_max_seq:
                return None
            return _reference(q, k, v, g, mode)

        _bench_shape(args, record, meta, q, k, v, g, args.impls, refs_for)


def suite_decode(args, rows, record):
    for dtype_name, d, hkv, lkv, b, lq in itertools.product(
        args.dtypes,
        args.head_dims,
        args.kv_heads,
        args.kv_lens,
        args.batches,
        args.q_lens,
    ):
        itemsize = jnp.dtype(dtype_name).itemsize
        kv_bytes = 2 * b * lkv * hkv * d * itemsize
        meta = dict(
            suite="decode",
            dtype=dtype_name,
            batch=b,
            lq=lq,
            lkv=lkv,
            hq=args.heads,
            hkv=hkv,
            head_dim=d,
            kv_mb=kv_bytes / MB,
        )
        if kv_bytes > args.max_kv_gb * 2**30:
            record(dict(meta, error=f"skipped: K+V > {args.max_kv_gb} GB"))
            continue
        q, k, v, g = _inputs(b, lq, lkv, args.heads, hkv, d, jnp.dtype(dtype_name))

        def refs_for(mode, q=q, k=k, v=v, g=g, lkv=lkv, b=b):
            if b * lkv > args.ref_max_seq * 64:
                return None
            return _reference(q, k, v, g, mode)

        _bench_shape(args, record, meta, q, k, v, g, args.impls, refs_for)


def suite_splitsweep(args, rows, record):
    from probjax.nn.layers.attention import flex_attention
    from probjax.nn.pallas_kernels.kernels.attention_splitkv import splitkv_forward

    for dtype_name, d, hkv, lkv, b, lq in itertools.product(
        args.dtypes,
        args.head_dims,
        args.kv_heads,
        args.kv_lens,
        args.batches,
        args.q_lens,
    ):
        q, k, v, _ = _inputs(b, lq, lkv, args.heads, hkv, d, jnp.dtype(dtype_name))
        meta = dict(
            suite="splitsweep",
            dtype=dtype_name,
            batch=b,
            lq=lq,
            lkv=lkv,
            hq=args.heads,
            hkv=hkv,
            head_dim=d,
        )
        fns = {
            0: jax.jit(
                lambda q, k, v: flex_attention(q, k, v, enable_gqa=True, split_kv=False)
            )
        }
        for s in args.splits:
            if s <= lkv // 64:
                fns[s] = jax.jit(
                    functools.partial(
                        splitkv_forward, sm_scale=1 / math.sqrt(d), num_splits=s
                    )
                )
        for rnd in range(args.repeats):
            for s, fn in fns.items():
                record(
                    dict(
                        meta,
                        num_splits=s,
                        round=rnd,
                        **run_safely(lambda fn=fn, a=(q, k, v): time_jax(fn, *a)),
                    )
                )
        jax.clear_caches()


def suite_blockk(args, rows, record):
    """Split-KV key-tile size (auto splits), interleaved rounds."""
    from probjax.nn.pallas_kernels.kernels.attention_splitkv import splitkv_forward

    for dtype_name, d, hkv, lkv, b, lq in itertools.product(
        args.dtypes,
        args.head_dims,
        args.kv_heads,
        args.kv_lens,
        args.batches,
        args.q_lens,
    ):
        q, k, v, _ = _inputs(b, lq, lkv, args.heads, hkv, d, jnp.dtype(dtype_name))
        meta = dict(
            suite="blockk",
            dtype=dtype_name,
            batch=b,
            lq=lq,
            lkv=lkv,
            hq=args.heads,
            hkv=hkv,
            head_dim=d,
        )
        fns = {
            bk: jax.jit(
                functools.partial(
                    splitkv_forward, sm_scale=1 / math.sqrt(d), block_k=bk
                )
            )
            for bk in args.block_ks
        }
        for rnd in range(args.repeats):
            for bk, fn in fns.items():
                record(
                    dict(
                        meta,
                        block_k=bk,
                        round=rnd,
                        **run_safely(lambda fn=fn, a=(q, k, v): time_jax(fn, *a)),
                    )
                )
        jax.clear_caches()


# -- peak (fresh subprocess per case) ---------------------------------------


def _child(spec: dict) -> dict:
    impl, mode = spec["impl"], spec["mode"]
    dtype = jnp.dtype(spec["dtype"])
    # Draw directly in the target dtype: an fp32 draw + cast would set the
    # (non-resettable) JAX peak counter before the measured call runs.
    b, lq, lkv, hq, hkv, d = (
        spec[n] for n in ("batch", "lq", "lkv", "hq", "hkv", "head_dim")
    )
    keys = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(keys[0], (b, lq, hq, d), dtype)
    k = jax.random.normal(keys[1], (b, lkv, hkv, d), dtype)
    v = jax.random.normal(keys[2], (b, lkv, hkv, d), dtype)
    g = jax.random.normal(keys[3], (b, lq, hq, d), jnp.float32)
    if impl == "torch_sdpa":
        import torch

        targs = _torch_inputs(q, k, v, g, mode, "cuda")
        del q, k, v, g
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        _torch_fn(mode)(*targs)
        torch.cuda.synchronize()
        return dict(peak_mb=torch.cuda.max_memory_allocated() / MB, inputs_mb=base / MB)
    jax.block_until_ready((q, k, v, g))
    dev = jax.devices()[0]
    base = dev.memory_stats()["bytes_in_use"]
    attn = _jax_attention(impl)
    if mode == "fwd":
        fn, fargs = jax.jit(attn), (q, k, v)
    else:
        fn, fargs = jax.jit(_grad_fn(attn)), (q, k, v, g)
    stats = _mem_stats(fn, *fargs)
    jax.block_until_ready(fn(*fargs))
    return dict(
        peak_mb=dev.memory_stats()["peak_bytes_in_use"] / MB,
        inputs_mb=base / MB,
        **stats,
    )


def _run_child(spec: dict, timeout: float = 600) -> dict:
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    cmd = [
        sys.executable,
        "-m",
        "docs.scripts.benchmark_kernel_attn_ext",
        "--child",
        json.dumps(spec),
        "--output",
        os.devnull,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return dict(error="timeout")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("CHILD_RESULT "):
            return json.loads(line.removeprefix("CHILD_RESULT "))
    err = [ln for ln in proc.stderr.splitlines() if "rror" in ln]
    return dict(error=(err[-1] if err else proc.stderr[-300:])[:300])


# Headline cases: training GQA (T=4096) and long-context decode.
PEAK_CASES = [
    dict(lq=4096, lkv=4096, batch=1, hq=32, hkv=8, head_dim=128),
    dict(lq=4096, lkv=4096, batch=1, hq=32, hkv=1, head_dim=128),
    dict(lq=1, lkv=65536, batch=8, hq=32, hkv=8, head_dim=128),
    dict(lq=16, lkv=65536, batch=8, hq=32, hkv=8, head_dim=128),
]


def suite_peak(args, rows, record):
    for case, dtype_name in itertools.product(PEAK_CASES, args.dtypes):
        decode = case["lq"] < 64
        modes = ["fwd"] if decode else ["fwd", "fwd_bwd"]
        impls = (
            ["probjax_splitkv", "probjax_nosplit", "flex_repeat", "torch_sdpa"]
            if decode
            else ["flex_native", "flex_repeat", "torch_sdpa"]
        )
        for mode, impl in itertools.product(modes, impls):
            spec = dict(case, dtype=dtype_name, mode=mode, impl=impl)
            record(dict(spec, suite="peak", **_run_child(spec)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--suite",
        choices=["gqa", "decode", "splitsweep", "blockk", "peak"],
        default="gqa",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, nargs="+", default=None)
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--q-lens", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--splits", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--block-ks", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--modes", nargs="+", default=["fwd", "fwd_bwd"])
    parser.add_argument("--impls", nargs="+", default=None)
    parser.add_argument("--baseline-impl", default=None)
    parser.add_argument("--ref-max-seq", type=int, default=1024)
    parser.add_argument("--max-kv-gb", type=float, default=4.5)
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child is not None:
        print("CHILD_RESULT " + json.dumps(_child(json.loads(args.child))), flush=True)
        return
    if args.kv_heads is None:
        args.kv_heads = [8, 4, 1] if args.suite == "gqa" else [32, 8]
    if args.impls is None:
        if args.suite == "gqa":
            args.impls = ["flex_native", "flex_repeat", "jax_xla", "jax_cudnn"]
        else:
            args.impls = ["probjax_splitkv", "probjax_nosplit", "jax_xla", "jax_cudnn"]
        args.impls.append("torch_sdpa")
    if args.baseline_impl is None:
        args.baseline_impl = "flex_repeat" if args.suite == "gqa" else "probjax_nosplit"

    rows = []
    if args.append and args.output.exists():
        rows = json.loads(args.output.read_text())["results"]

    def record(row):
        row = dict(row, tag=args.tag, device_arg=args.device)
        rows.append(row)
        print(
            {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in row.items()
                if k not in ("residual_shapes", "rel_err_parts")
            },
            flush=True,
        )

    suites = {
        "gqa": suite_gqa,
        "decode": suite_decode,
        "splitsweep": suite_splitsweep,
        "blockk": suite_blockk,
        "peak": suite_peak,
    }
    suites[args.suite](args, rows, record)
    config = vars(args) | {"output": str(args.output)}
    write_results(args.output, config, rows)


if __name__ == "__main__":
    main()
