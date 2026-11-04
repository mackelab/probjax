"""Benchmark of ``kernel_mv`` / ``rbf_kde_density`` on CPU or GPU (time and memory).

Run from the repository root::

    python -m docs.scripts.benchmark_kernel_kmv --device gpu \
        --output docs/benchmarks/kernel_kmv_gpu.json
    JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_kmv --device cpu \
        --output docs/benchmarks/kernel_kmv_cpu.json

Implementations (RBF kernel, ``q [B, N, D]``, ``k [B, M, D]``, ``v [B, M, O]``,
``N == M``):

* ``dense``            -- ``rbf_kernel_mv_naive``: materializes the [N, M] matrix
* ``xla``              -- ``implementation="xla"``: row-blocked XLA (CPU default)
* ``pallas``           -- ``implementation="pallas"``: tiled Pallas kernel (GPU default)
* ``pallas_interpret`` -- Pallas in interpret mode (the old CPU default)
* ``torch_dense``      -- torch ``exp(-cdist**2 / 2l**2) @ v`` (external baseline)
* ``default``          -- ``rbf_kernel_mv`` with default arguments (used to time a
  baseline checkout of probjax for before/after comparisons)

Each row records the median/IQR time, XLA's compiled memory analysis
(``xla_temp_mb`` / ``xla_arg_mb`` / ``xla_out_mb``) and the relative error
(``||x - x_ref|| / ||x_ref||``) of the output (fwd) or of each gradient
(fwd_bwd, loss ``0.5 * sum(out**2)``) vs the XLA path at HIGHEST matmul precision.

Sections (``--sections``): ``grid`` (N x D x O sweep), ``kde`` (``rbf_kde_density``
end to end), ``tiles`` (Pallas block-size sweep, GPU), ``peak`` (true device peak
memory per case, each in a fresh subprocess), ``max_n`` (largest N per
implementation that runs fwd+bwd, fresh subprocess per attempt).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from docs.scripts._gpu_bench import run_safely, time_jax, write_results
from probjax.nn.pallas_kernels import (
    rbf_kde_density,
    rbf_kernel_mv,
    rbf_kernel_mv_naive,
)

ARGNUMS = (0, 1, 2, 3)
GRAD_NAMES = ("dq", "dk", "dv", "dlengthscale")


def make_impl(name, block_q=64, block_k=64, block_o=64):
    if name == "dense":
        return lambda q, k, v, ls: rbf_kernel_mv_naive(q, k, v, ls)
    if name == "default":  # public defaults; also runs against older probjax
        return lambda q, k, v, ls: rbf_kernel_mv(q, k, v, ls)
    if name == "xla":
        return lambda q, k, v, ls: rbf_kernel_mv(q, k, v, ls, implementation="xla")
    if name == "pallas":
        return lambda q, k, v, ls: rbf_kernel_mv(
            q, k, v, ls, block_q, block_k, block_o, implementation="pallas"
        )
    if name == "pallas_interpret":
        return lambda q, k, v, ls: rbf_kernel_mv(
            q, k, v, ls, block_q, block_k, block_o, interpret=True
        )
    raise ValueError(name)


def make_inputs(b, n, d, o, m=None):
    m = n if m is None else m
    ks = jax.random.split(jax.random.key(n * 7 + d * 3 + o), 3)
    q = jax.random.normal(ks[0], (b, n, d))
    k = jax.random.normal(ks[1], (b, m, d))
    v = jax.random.normal(ks[2], (b, m, o))
    # Lengthscale ~ sqrt(D) keeps the kernel matrix away from 0/1 for every D.
    ls = jnp.asarray(float(np.sqrt(d)), jnp.float32)
    return q, k, v, ls


def build(fn, mode):
    if mode == "fwd":
        return jax.jit(fn)

    def loss(*args):
        return 0.5 * jnp.sum(fn(*args).astype(jnp.float32) ** 2)

    return jax.jit(jax.grad(loss, argnums=ARGNUMS))


def memory_mb(compiled):
    try:
        ma = compiled.memory_analysis()
        return dict(
            xla_temp_mb=ma.temp_size_in_bytes / 2**20,
            xla_arg_mb=ma.argument_size_in_bytes / 2**20,
            xla_out_mb=ma.output_size_in_bytes / 2**20,
        )
    except Exception as exc:  # noqa: BLE001
        return dict(memory_error=str(exc)[:200])


def rel_err(x, ref):
    x = np.asarray(x, np.float64)
    ref = np.asarray(ref, np.float64)
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(x - ref) / denom) if denom > 0 else 0.0


def reference(args, mode):
    fn = make_impl("xla")
    with jax.default_matmul_precision("highest"):
        out = build(fn, mode)(*args)
    return jax.tree.map(np.asarray, out)


def time_case(fn, args, mode, ref, repeats):
    jitted = build(fn, mode)
    t0 = time.perf_counter()
    compiled = jitted.lower(*args).compile()
    lower_s = time.perf_counter() - t0
    row = memory_mb(compiled)
    row.update(time_jax(compiled, *args, repeats=repeats))
    row["compile_s"] = lower_s
    out = compiled(*args)
    if ref is not None:
        if mode == "fwd":
            row["rel_err"] = rel_err(out, ref)
        else:
            row["rel_err"] = {
                n: rel_err(g, r) for n, g, r in zip(GRAD_NAMES, out, ref, strict=True)
            }
    return row


def time_torch_case(args, mode, ref, repeats):
    """Dense torch baseline: exp(-cdist^2 / 2l^2) @ v (grads for q, k, v only)."""
    import torch

    cuda = jax.default_backend() == "gpu"
    dev = "cuda" if cuda else "cpu"
    q, k, v = (torch.from_numpy(np.array(x)).to(dev) for x in args[:3])
    ls = float(args[3])

    def f(q, k, v):
        r2 = torch.cdist(q, k) ** 2
        return torch.exp(-0.5 * r2 / ls**2) @ v

    if mode == "fwd":

        def fn():
            return f(q, k, v)

    else:
        leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]

        def fn():
            out = f(*leaves)
            return torch.autograd.grad(0.5 * (out**2).sum(), leaves)

    def sync():
        if cuda:
            torch.cuda.synchronize()

    try:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = fn()
        sync()
        compile_s = time.perf_counter() - start
        start = time.perf_counter()
        fn()
        sync()
        inner = int(min(max(0.05 / max(time.perf_counter() - start, 1e-6), 1), 200))
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(inner):
                fn()
            sync()
            samples.append((time.perf_counter() - start) / inner * 1e3)
        q1, med, q3 = np.percentile(samples, [25, 50, 75])
        row = dict(compile_s=compile_s, inner=inner, median_ms=med, q1_ms=q1, q3_ms=q3)
        if cuda:
            row["peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        if ref is not None:
            if mode == "fwd":
                row["rel_err"] = rel_err(out.detach().cpu().numpy(), ref)
            else:
                row["rel_err"] = {
                    n: rel_err(g.cpu().numpy(), r)
                    for n, g, r in zip(GRAD_NAMES[:3], out, ref[:3], strict=True)
                }
        return row
    finally:
        if cuda:
            torch.cuda.empty_cache()


def sweep_grid(rows, impls, sizes, dims, outs, modes, repeats, max_n):
    for n in sizes:
        for d in dims:
            for o in outs:
                args = make_inputs(1, n, d, o)
                for mode in modes:
                    ref = run_safely(lambda a=args, m=mode: dict(ref=reference(a, m)))
                    ref = ref.get("ref")
                    for name in impls:
                        if n > max_n.get(name, 1 << 62):
                            continue
                        shape = dict(batch=1, n=n, m=n, dim=d, out=o)
                        if name == "torch_dense":
                            res = run_safely(
                                lambda a=args, m=mode, r=ref: time_torch_case(
                                    a, m, r, repeats
                                )
                            )
                        else:
                            res = run_safely(
                                lambda nm=name, a=args, m=mode, r=ref: time_case(
                                    make_impl(nm), a, m, r, repeats
                                )
                            )
                        row = dict(
                            family="kernel_mv", impl=name, mode=mode, **shape, **res
                        )
                        rows.append(row)
                        print(_fmt(row), flush=True)


def sweep_kde(rows, impls, sizes, dims, modes, repeats, max_n):
    def kde(name):
        if name == "dense":

            def fn(query, data, ls):
                d = query.shape[-1]
                w = jnp.full((data.shape[0], 1), 1.0 / data.shape[0], query.dtype)
                dens = rbf_kernel_mv_naive(query[None], data[None], w[None], ls)[
                    0, :, 0
                ]
                dens = dens / (jnp.sqrt(2.0 * jnp.pi) * ls) ** d
                return jnp.log(jnp.maximum(dens, jnp.finfo(dens.dtype).tiny))

            return fn
        if name == "pallas_interpret":
            kwargs = dict(interpret=True)
        elif name == "default":
            kwargs = {}
        else:
            kwargs = dict(implementation=name)
        return lambda query, data, ls: rbf_kde_density(
            query, data, ls, log_density=True, **kwargs
        )

    for n in sizes:
        for d in dims:
            ks = jax.random.split(jax.random.key(n + d), 2)
            query = jax.random.normal(ks[0], (n, d))
            data = jax.random.normal(ks[1], (n, d))
            ls = jnp.asarray(0.5, jnp.float32)
            args = (query, data, ls)
            for mode in modes:
                for name in impls:
                    if n > max_n.get(name, 1 << 62) or name == "torch_dense":
                        continue

                    def case(name=name, mode=mode, args=args):
                        fn = kde(name)
                        if mode == "fwd":
                            jitted = jax.jit(fn)
                        else:
                            jitted = jax.jit(
                                jax.grad(lambda *a: jnp.sum(fn(*a)), argnums=(0, 2))
                            )
                        compiled = jitted.lower(*args).compile()
                        row = memory_mb(compiled)
                        row.update(time_jax(compiled, *args, repeats=repeats))
                        return row

                    res = run_safely(case)
                    row = dict(
                        family="kde_density",
                        impl=name,
                        mode=mode,
                        n=n,
                        m=n,
                        dim=d,
                        **res,
                    )
                    rows.append(row)
                    print(_fmt(row), flush=True)


def sweep_tiles(rows, sizes, dims, outs, tiles, modes, repeats):
    for n in sizes:
        for d in dims:
            for o in outs:
                args = make_inputs(1, n, d, o)
                for mode in modes:
                    for bq, bk, bo in tiles:
                        fn = make_impl("pallas", bq, bk, bo)
                        res = run_safely(
                            lambda f=fn, a=args, m=mode: time_case(
                                f, a, m, None, repeats
                            )
                        )
                        row = dict(
                            family="kernel_mv_tiles",
                            impl="pallas",
                            mode=mode,
                            n=n,
                            m=n,
                            dim=d,
                            out=o,
                            block_q=bq,
                            block_k=bk,
                            block_o=bo,
                            **res,
                        )
                        rows.append(row)
                        print(_fmt(row), flush=True)


def _child(spec):
    """Run one case in this (fresh) process and print a JSON line."""
    name, mode, n, d, o = (
        spec["impl"],
        spec["mode"],
        spec["n"],
        spec["dim"],
        spec["out"],
    )
    args = make_inputs(1, n, d, o)
    t0 = time.perf_counter()
    out = build(make_impl(name), mode)(*args)
    jax.block_until_ready(out)
    res = dict(run_s=time.perf_counter() - t0)
    stats = jax.devices()[0].memory_stats() or {}
    if "peak_bytes_in_use" in stats:
        res["peak_mb"] = stats["peak_bytes_in_use"] / 2**20
    print("RESULT " + json.dumps(res), flush=True)


def run_child(spec, timeout):
    cmd = [
        sys.executable,
        "-m",
        "docs.scripts.benchmark_kernel_kmv",
        "--child",
        json.dumps(spec),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=os.environ.copy()
        )
    except subprocess.TimeoutExpired:
        return dict(error=f"timeout>{timeout}s")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    tail = [ln for ln in proc.stderr.splitlines() if ln.strip()][-1:] or ["?"]
    return dict(error=tail[0][:300])


def sweep_peak(rows, impls, sizes, d, o, modes, timeout):
    for n in sizes:
        for mode in modes:
            for name in impls:
                spec = dict(impl=name, mode=mode, n=n, dim=d, out=o)
                row = dict(family="kernel_mv_peak", **spec, **run_child(spec, timeout))
                rows.append(row)
                print(_fmt(row), flush=True)


def sweep_max_n(rows, impls, ladder, d, o, timeout):
    for name in impls:
        for n in ladder:
            spec = dict(impl=name, mode="fwd_bwd", n=n, dim=d, out=o)
            res = run_child(spec, timeout)
            row = dict(family="kernel_mv_max_n", **spec, **res)
            rows.append(row)
            print(_fmt(row), flush=True)
            if "error" in res:
                break


def _fmt(row):
    keys = (
        "family",
        "impl",
        "mode",
        "n",
        "dim",
        "out",
        "block_q",
        "block_k",
        "block_o",
        "median_ms",
        "xla_temp_mb",
        "peak_mb",
        "run_s",
        "rel_err",
        "error",
    )
    parts = []
    for key in keys:
        if key in row:
            val = row[key]
            if isinstance(val, float):
                val = f"{val:.4g}"
            elif isinstance(val, dict):
                val = {k_: f"{v_:.1e}" for k_, v_ in val.items()}
            parts.append(f"{key}={val}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sections", nargs="+", default=["grid", "kde"])
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument("--outs", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--modes", nargs="+", default=["fwd", "fwd_bwd"])
    parser.add_argument("--impls", nargs="+")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--tiles",
        nargs="+",
        default=["64,64,64", "64,64,16", "128,64,16", "64,128,16", "128,128,16"],
    )
    parser.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=[16384, 32768, 49152, 65536, 98304, 131072, 262144],
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.child:
        _child(json.loads(args.child))
        return

    backend = jax.default_backend()
    if args.device == "cpu" and backend != "cpu":
        raise SystemExit("--device cpu: run with JAX_PLATFORMS=cpu")
    if args.device == "gpu" and backend != "gpu":
        raise SystemExit("--device gpu needs a GPU backend")
    gpu = args.device == "gpu"
    impls = args.impls or (
        ["dense", "xla", "pallas"] if gpu else ["dense", "xla", "pallas_interpret"]
    )
    sizes = args.sizes or ([1024, 4096, 16384, 65536] if gpu else [1024, 4096, 16384])
    # Interpret mode is ~1000x slower; dense on CPU beyond 16k is minutes.
    max_n = {"pallas_interpret": 1024}
    rows: list[dict] = []
    config = {
        k_: (str(v_) if isinstance(v_, Path) else v_) for k_, v_ in vars(args).items()
    }
    config.update(impls=impls, sizes=sizes)
    for section in args.sections:
        if section == "grid":
            sweep_grid(
                rows,
                impls,
                sizes,
                args.dims,
                args.outs,
                args.modes,
                args.repeats,
                max_n,
            )
        elif section == "kde":
            sweep_kde(rows, impls, sizes, [2, 8], args.modes, args.repeats, max_n)
        elif section == "tiles":
            tiles = [tuple(int(x) for x in t.split(",")) for t in args.tiles]
            sweep_tiles(
                rows, sizes, args.dims, args.outs, tiles, args.modes, args.repeats
            )
        elif section == "peak":
            sweep_peak(rows, impls, sizes, 8, 16, args.modes, args.timeout)
        elif section == "max_n":
            sweep_max_n(rows, impls, args.ladder, 8, 16, args.timeout)
        if args.output:
            write_results(args.output, config, rows)


if __name__ == "__main__":
    main()
