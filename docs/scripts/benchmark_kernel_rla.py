"""Residual linear attention (RLA) benchmark: time and memory, CPU and GPU.

Run from the repository root (hold the shared bench lock for timing runs)::

    python -m docs.scripts.benchmark_kernel_rla --device gpu \
        --suites kernel layer --output docs/benchmarks/kernel_rla_gpu.json

Suites:

* ``kernel`` -- ``residual_linear_attention`` implementations at
  ``[B, H, L, D]``: ``scan`` (per-token ``lax.scan`` reference), ``chunked``
  (chunk-parallel XLA form, the CPU default and GPU fallback), ``pallas``
  (Pallas forward, chunked-XLA backward -- the GPU default) and
  ``pallas_scanbwd`` (Pallas forward with the previous step-scan VJP
  backward, i.e. the GPU path before this change). Forward and fwd+bwd
  (grads w.r.t. q, k, v, h0).
* ``chunks`` -- chunk-size sweep of the chunked form.
* ``layer`` -- the ``RAttention`` layer end to end (forward and the grad of a
  scalar loss w.r.t. inputs and parameters) for each linear-attention
  implementation.
* ``peak`` -- true device peak memory (``peak_bytes_in_use``) of headline
  fwd+bwd cases, each in a fresh subprocess; GPU only.

Every timed row also reports XLA's compiled memory analysis (``xla_temp_mb``,
``xla_arg_mb``, ``xla_out_mb``) and relative errors vs the fp32 scan reference.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from docs.scripts._gpu_bench import run_safely, time_jax, write_results

MB = 2**20
WINDOW = 64


def _rla():
    from probjax.nn.pallas_kernels.kernels import linear_attention as la

    return la


def _pallas_scanbwd(q, k, v, h0, *, window_size, feature_map, chunk_size):
    """Pallas forward + step-scan VJP backward (the pre-change GPU default)."""
    la = _rla()
    fm = la.FeatureMap(feature_map)

    @functools.partial(jax.custom_vjp)
    def core(q, k, v, h0):
        return la._linear_attention_pallas_impl(q, k, v, h0, fm, chunk_size)

    def fwd(q, k, v, h0):
        return core(q, k, v, h0), (q, k, v, h0)

    def bwd(res, ct):
        _, pullback = jax.vjp(lambda *a: la._linear_attention_scan_core(*a, fm), *res)
        return pullback(ct)

    core.defvjp(fwd, bwd)
    shift = window_size + 1
    k = la.right_shift_and_zero_pad(k, shift)
    v = la.right_shift_and_zero_pad(v, shift)
    return core(q, k, v, h0)


def _impl(name, chunk_size):
    la = _rla()
    if name == "scan":
        return lambda q, k, v, h0: la.residual_linear_attention_scan(
            q, k, v, h0, window_size=WINDOW
        )
    if name == "chunked":
        return lambda q, k, v, h0: la.residual_linear_attention_chunked(
            q, k, v, h0, window_size=WINDOW, chunk_size=chunk_size
        )
    if name == "pallas":
        return lambda q, k, v, h0: la.residual_linear_attention_pallas(
            q, k, v, h0, window_size=WINDOW, chunk_size=chunk_size
        )
    if name == "pallas_scanbwd":
        return lambda q, k, v, h0: _pallas_scanbwd(
            q,
            k,
            v,
            h0,
            window_size=WINDOW,
            feature_map="softmax",
            chunk_size=la._gpu_chunk_size(q, v, chunk_size),
        )
    raise ValueError(name)


def _loss(fn):
    def loss(q, k, v, h0):
        out, state = fn(q, k, v, h0)
        return jnp.sum(out.astype(jnp.float32) ** 2) + 1e-3 * jnp.sum(state**2)

    return loss


def _jitted(fn, mode):
    if mode == "fwd":
        return jax.jit(fn)
    return jax.jit(jax.grad(_loss(fn), argnums=(0, 1, 2, 3)))


def _memory(jfn, *args) -> dict:
    try:
        ma = jfn.lower(*args).compile().memory_analysis()
        return dict(
            xla_temp_mb=ma.temp_size_in_bytes / MB,
            xla_arg_mb=ma.argument_size_in_bytes / MB,
            xla_out_mb=ma.output_size_in_bytes / MB,
        )
    except Exception as exc:  # noqa: BLE001
        return dict(xla_memory_error=str(exc)[:200])


def _rel(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _inputs(batch, heads, kv_heads, seq, dim, dtype, seed=0):
    ks = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(ks[0], (batch, heads, seq, dim)) / np.sqrt(dim)
    k = jax.random.normal(ks[1], (batch, kv_heads, seq, dim)) / np.sqrt(dim)
    v = jax.random.normal(ks[2], (batch, kv_heads, seq, dim))
    h0 = 0.1 * jax.random.normal(ks[3], (batch, heads, 2 * dim, dim))
    return tuple(x.astype(dtype) for x in (q, k, v)) + (h0,)


def _available(name, device, args, chunk_size):
    if name.startswith("pallas"):
        if device != "gpu":
            return False  # interpret mode on CPU is not a contender
        return _rla()._gpu_chunk_size(args[0], args[2], chunk_size) is not None
    return True


def bench_kernel(rows, device, seqs, dims, dtypes, impls, batch, heads, chunk):
    for dtype_name in dtypes:
        dtype = jnp.dtype(dtype_name)
        for dim in dims:
            for seq in seqs:
                args = _inputs(batch, heads, heads, seq, dim, dtype)
                args32 = tuple(x.astype(jnp.float32) for x in args)
                ref = {}
                for mode in ("fwd", "fwd_bwd"):
                    for name in impls:
                        if not _available(name, device, args, chunk):
                            continue
                        shape = dict(
                            batch=batch, heads=heads, seq=seq, dim=dim, dtype=dtype_name
                        )

                        def case(
                            name=name, mode=mode, args=args, args32=args32, ref=ref
                        ):
                            fn = _impl(name, chunk)
                            jfn = _jitted(fn, mode)
                            row = dict(_memory(jfn, *args))
                            row.update(time_jax(jfn, *args, repeats=5))
                            # Errors vs the fp32 scan reference (computed once).
                            if mode not in ref:
                                ref[mode] = jax.device_get(
                                    _jitted(_impl("scan", chunk), mode)(*args32)
                                )
                            out = jax.device_get(jfn(*args))
                            leaves = jax.tree.leaves(out)
                            ref_leaves = jax.tree.leaves(ref[mode])
                            row["rel_err"] = max(
                                _rel(a, b)
                                for a, b in zip(leaves, ref_leaves, strict=True)
                            )
                            return row

                        if name == "scan" and mode == "fwd_bwd" and seq > 8192:
                            continue
                        result = run_safely(case)
                        row = dict(
                            suite="kernel", impl=name, mode=mode, chunk=chunk, **shape
                        )
                        row.update(result)
                        rows.append(row)
                        print(json.dumps(row), flush=True)


def bench_chunks(rows, device, seqs, dims, chunks, batch, heads):
    for dim in dims:
        for seq in seqs:
            args = _inputs(batch, heads, heads, seq, dim, jnp.float32)
            for chunk in chunks:
                for mode in ("fwd", "fwd_bwd"):

                    def case(chunk=chunk, mode=mode, args=args):
                        jfn = _jitted(_impl("chunked", chunk), mode)
                        row = dict(_memory(jfn, *args))
                        row.update(time_jax(jfn, *args, repeats=5))
                        return row

                    row = dict(
                        suite="chunks",
                        impl="chunked",
                        mode=mode,
                        chunk=chunk,
                        batch=batch,
                        heads=heads,
                        seq=seq,
                        dim=dim,
                        dtype="float32",
                    )
                    row.update(run_safely(case))
                    rows.append(row)
                    print(json.dumps(row), flush=True)


def bench_layer(rows, device, seqs, impls, batch, heads, dim, window, chunk):
    from flax import nnx

    from probjax.nn.layers.attention import dot_product_attention
    from probjax.nn.layers.rattention import RAttention

    features = heads * dim
    for seq in seqs:
        x = jax.random.normal(jax.random.key(1), (batch, seq, features))
        for name in impls:
            if name == "pallas_scanbwd":
                continue
            if name == "pallas" and device != "gpu":
                continue
            layer = RAttention(
                num_heads=heads,
                in_features=features,
                qkv_features=features,
                out_features=features,
                sliding_window_size=window,
                chunk_size=chunk,
                linear_attention_implementation=name,
                attention_fn=dot_product_attention,
                rngs=nnx.Rngs(0),
            )
            graphdef, params = nnx.split(layer)

            def fwd(params, x, graphdef=graphdef):
                return nnx.merge(graphdef, params)(x)

            def grad(params, x, graphdef=graphdef):
                def loss(params, x):
                    return jnp.mean(nnx.merge(graphdef, params)(x) ** 2)

                return jax.grad(loss, argnums=(0, 1))(params, x)

            for mode, fn in (("fwd", fwd), ("fwd_bwd", grad)):

                def case(fn=fn, params=params, x=x):
                    jfn = jax.jit(fn)
                    row = dict(_memory(jfn, params, x))
                    row.update(time_jax(jfn, params, x, repeats=5))
                    return row

                row = dict(
                    suite="layer",
                    impl=name,
                    mode=mode,
                    chunk=chunk,
                    batch=batch,
                    heads=heads,
                    seq=seq,
                    dim=dim,
                    window=window,
                    dtype="float32",
                )
                row.update(run_safely(case))
                rows.append(row)
                print(json.dumps(row), flush=True)


_PEAK_CHILD = """
import json, sys, jax
from docs.scripts.benchmark_kernel_rla import _impl, _inputs, _jitted
name, mode, b, h, l, d, chunk = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:8])
args = _inputs(b, h, h, l, d, jax.numpy.float32)
jax.block_until_ready(args)
base = jax.devices()[0].memory_stats()["peak_bytes_in_use"]
out = {}
try:
    jax.block_until_ready(_jitted(_impl(name, chunk), mode)(*args))
    out["peak_mb"] = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**20
    out["inputs_mb"] = base / 2**20
except Exception as exc:
    out["error"] = (type(exc).__name__ + ": " + str(exc).splitlines()[0])[:300]
print("RESULT " + json.dumps(out))
"""


def bench_peak(rows, cases, chunk):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    for name, mode, b, h, seq, dim in cases:
        proc = subprocess.run(
            [sys.executable, "-c", _PEAK_CHILD, name, mode]
            + [str(x) for x in (b, h, seq, dim, chunk)],
            capture_output=True,
            text=True,
            env=env,
            cwd=Path(__file__).resolve().parents[2],
        )
        tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        result = dict(error=tail[:300])
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                result = json.loads(line[7:])
        row = dict(
            suite="peak",
            impl=name,
            mode=mode,
            chunk=chunk,
            batch=b,
            heads=h,
            seq=seq,
            dim=dim,
            dtype="float32",
        )
        row.update(result)
        rows.append(row)
        print(json.dumps(row), flush=True)


def _peak_cases(device):
    cases = []
    # Headline: B=4, H=8, D=64 fwd+bwd, growing L until OOM.
    for seq in (2048, 8192, 32768, 65536, 131072):
        for name in ("scan", "pallas_scanbwd", "pallas", "chunked"):
            cases.append((name, "fwd_bwd", 4, 8, seq, 64))
    for seq in (8192, 65536):
        for name in ("scan", "pallas", "chunked"):
            cases.append((name, "fwd", 4, 8, seq, 64))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["kernel", "layer"],
        help="kernel chunks layer peak",
    )
    parser.add_argument("--seqs", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--dims", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--dtypes", nargs="+", default=["float32"])
    parser.add_argument(
        "--impls",
        nargs="+",
        default=["scan", "chunked", "pallas", "pallas_scanbwd"],
    )
    parser.add_argument("--chunks", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layer-seqs", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if jax.default_backend() != args.device:
        raise SystemExit(f"backend is {jax.default_backend()}, wanted {args.device}")
    chunk = args.chunk or 128
    rows: list[dict] = []
    if args.append and args.output.exists():
        rows = json.loads(args.output.read_text())["results"]
    if "kernel" in args.suites:
        bench_kernel(
            rows,
            args.device,
            args.seqs,
            args.dims,
            args.dtypes,
            args.impls,
            args.batch,
            args.heads,
            chunk,
        )
    if "chunks" in args.suites:
        bench_chunks(
            rows, args.device, args.seqs, args.dims, args.chunks, args.batch, args.heads
        )
    if "layer" in args.suites:
        bench_layer(
            rows,
            args.device,
            args.layer_seqs,
            args.impls,
            args.batch,
            8,
            64,
            128,
            chunk,
        )
    if "peak" in args.suites:
        bench_peak(rows, _peak_cases(args.device), chunk)
    config = dict(
        suites=args.suites,
        window=WINDOW,
        chunk=chunk,
        batch=args.batch,
        heads=args.heads,
        layer=dict(batch=args.batch, heads=8, head_dim=64, window=128),
    )
    write_results(args.output, config, rows)


if __name__ == "__main__":
    main()
