"""Benchmark of the gated delta rule (Gated DeltaNet) implementations.

Run from the repository root (hold the shared bench lock for timings)::

    python -m docs.scripts.benchmark_kernel_delta --device gpu \\
        --output docs/benchmarks/kernel_delta_gpu.json
    JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_delta --device cpu \\
        --output docs/benchmarks/kernel_delta_cpu.json

Implementations (``--impls``):

* ``recurrent``   -- token-by-token ``lax.scan`` (``gated_delta_rule_recurrent``).
* ``chunked``     -- chunkwise WY/UT form in pure JAX with the memory-lean
  end-to-end VJP (residuals: inputs + chunk-boundary states).
* ``chunked_fat`` -- same forward, but the chunk-local WY quantities are
  stored for backward instead of recomputed (memory baseline).
* ``pallas``      -- chunked form with Pallas-Triton inter-chunk kernels (GPU).
* ``ssd``         -- Mamba-2 SSD (``probjax.nn.pallas_kernels.ssd``) at equal
  shapes, as a cost reference (no delta-rule correction term).

Every row records time (median/IQR), XLA ``memory_analysis`` (temp/arg/out MB),
for ``fwd_bwd`` the residual bytes held by ``jax.vjp``, and relative errors
(``max|a-b| / max|b|``) vs an fp32 ``HIGHEST``-precision chunked reference.
``--peak`` re-runs headline cases in fresh subprocesses to record the true
device peak (``peak_bytes_in_use``; GPU only).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from docs.scripts._gpu_bench import run_safely, time_jax, write_results
from probjax.nn.pallas_kernels.kernels import delta_rule as dr

MB = 2**20
GRAD_NAMES = ("q", "k", "v", "log_alpha", "beta")


def make_inputs(b, h, seq, dk, dv, dtype, seed=0):
    ks = jax.random.split(jax.random.key(seed), 6)
    q = dr.l2_normalize(jax.random.normal(ks[0], (b, h, seq, dk))).astype(dtype)
    k = dr.l2_normalize(jax.random.normal(ks[1], (b, h, seq, dk))).astype(dtype)
    v = jax.random.normal(ks[2], (b, h, seq, dv)).astype(dtype)
    # Gates alpha in ~(0.9, 1): typical Gated DeltaNet decays.
    log_alpha = jax.nn.log_sigmoid(jax.random.normal(ks[3], (b, h, seq)) + 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(ks[4], (b, h, seq)))
    ct = jax.random.normal(ks[5], (b, h, seq, dv), jnp.float32)
    return (q, k, v, log_alpha, beta), ct


def impl_fn(name, chunk_size=64):
    if name == "recurrent":
        return lambda *a: dr.gated_delta_rule_recurrent(*a)[0]
    if name == "chunked":
        return lambda *a: dr.gated_delta_rule_chunked(*a, chunk_size=chunk_size)[0]
    if name == "chunked_fat":
        return lambda *a: dr._chunked_impl(
            *a,
            None,
            chunk_size=chunk_size,
            qk_l2norm=False,
            precision=None,
            use_pallas=False,
            store_prep=True,
        )[0]
    if name == "pallas":
        return lambda *a: dr.gated_delta_rule_pallas(*a, chunk_size=chunk_size)[0]
    if name == "reference":
        return lambda *a: dr.gated_delta_rule_chunked(
            *(x.astype(jnp.float32) for x in a),
            chunk_size=chunk_size,
            precision=jax.lax.Precision.HIGHEST,
        )[0]
    if name == "ssd":
        from probjax.nn.pallas_kernels import ssd

        def f(q, k, v, log_alpha, beta):
            del beta
            return ssd(
                q.astype(jnp.float32) if jax.default_backend() == "cpu" else q,
                k.astype(jnp.float32) if jax.default_backend() == "cpu" else k,
                v.astype(jnp.float32),
                log_alpha,
            )

        return f
    raise ValueError(name)


def supported(name, dk, dv, chunk_size, device):
    if name == "pallas":
        from probjax.nn.pallas_kernels.kernels import delta_rule_pallas as dp

        return device == "gpu" and dp.supports_shape(chunk_size, dk, dv)
    return True


def build(name, mode, ct, chunk_size):
    fn = impl_fn(name, chunk_size)
    if mode == "fwd":
        return jax.jit(fn)

    def loss(*args):
        return jnp.vdot(fn(*args).astype(jnp.float32), ct)

    argnums = (0, 1, 2, 3) if name == "ssd" else (0, 1, 2, 3, 4)
    return jax.jit(jax.grad(loss, argnums=argnums))


def mem_analysis(jfn, args):
    ma = jfn.lower(*args).compile().memory_analysis()
    if ma is None:
        return {}
    return dict(
        xla_temp_mb=ma.temp_size_in_bytes / MB,
        xla_arg_mb=ma.argument_size_in_bytes / MB,
        xla_out_mb=ma.output_size_in_bytes / MB,
    )


def residual_mb(name, args, chunk_size):
    """Bytes held by the vjp closure that are not the primal inputs."""
    _, vjp = jax.vjp(impl_fn(name, chunk_size), *args)
    ids = {id(a) for a in args}
    leaves = [x for x in jax.tree.leaves(vjp) if hasattr(x, "nbytes")]
    out = sum(x.nbytes for x in leaves if id(x) not in ids) / MB
    del vjp
    return out


def rel_err(out, ref):
    errs = []
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(ref), strict=True):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        errs.append(float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)))
    return max(errs), errs


def bench_case(shape, mode, names, device, repeats, rows, chunk_size, with_err=True):
    b, h, seq, dk, dv, dtype = shape
    args, ct = make_inputs(b, h, seq, dk, dv, jnp.dtype(dtype))
    ref = None
    if with_err:
        ref = run_safely(
            lambda: dict(out=build("reference", mode, ct, chunk_size)(*args))
        ).get("out")
    for name in names:
        base = dict(
            impl=name,
            mode=mode,
            batch=b,
            heads=h,
            seq=seq,
            dk=dk,
            dv=dv,
            dtype=dtype,
            chunk_size=chunk_size,
        )
        if not supported(name, dk, dv, chunk_size, device):
            rows.append(dict(base, skipped="unsupported shape"))
            continue

        def case(name=name):
            jfn = build(name, mode, ct, chunk_size)
            res = time_jax(jfn, *args, repeats=repeats)
            res.update(mem_analysis(jfn, args))
            if mode == "fwd_bwd":
                res["residual_mb"] = residual_mb(name, args, chunk_size)
            if ref is not None and name != "ssd":
                res["rel_err"], per = rel_err(jfn(*args), ref)
                if mode == "fwd_bwd":
                    res["rel_err_grads"] = dict(zip(GRAD_NAMES, per, strict=False))
            return res

        row = dict(base, **run_safely(case))
        rows.append(row)
        print(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
            flush=True,
        )


# ----------------------------------------------------------------- subprocess


def _peak_child(spec):
    """Run one case (compile + 2 calls) and print the device peak bytes."""
    b, h, seq, dk, dv, dtype = spec["shape"]
    args, ct = make_inputs(b, h, seq, dk, dv, jnp.dtype(dtype))
    jfn = build(spec["impl"], spec["mode"], ct, spec["chunk_size"])
    base = (jax.devices()[0].memory_stats() or {}).get("bytes_in_use", 0)
    jax.block_until_ready(jfn(*args))
    jax.block_until_ready(jfn(*args))
    stats = jax.devices()[0].memory_stats() or {}
    print(
        "PEAK_JSON "
        + json.dumps(
            dict(
                peak_mb=stats.get("peak_bytes_in_use", 0) / MB,
                inputs_mb=base / MB,
            )
        )
    )


def run_peak(spec, timeout=600):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "docs.scripts.benchmark_kernel_delta",
            "--peak-child",
            json.dumps(spec),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PEAK_JSON "):
            return json.loads(line[len("PEAK_JSON ") :])
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    msg = next(
        (ln for ln in reversed(tail) if "Error" in ln or "error" in ln),
        tail[-1] if tail else "",
    )
    return dict(error=msg[:300])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--output", type=Path)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq-lens", nargs="+", type=int, default=None)
    p.add_argument("--dims", nargs="+", type=int, default=[64, 128])
    p.add_argument("--dtypes", nargs="+", default=None)
    p.add_argument("--impls", nargs="+", default=None)
    p.add_argument("--modes", nargs="+", default=["fwd", "fwd_bwd"])
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[64])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--max-recurrent-seq",
        type=int,
        default=None,
        help="skip the token scan above this length (it is O(L) sequential)",
    )
    p.add_argument("--peak", action="store_true", help="subprocess device peaks")
    p.add_argument("--no-err", action="store_true")
    p.add_argument("--append", action="store_true", help="merge into --output")
    p.add_argument("--peak-child", type=str, default=None, help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.peak_child is not None:
        _peak_child(json.loads(a.peak_child))
        return

    gpu = a.device == "gpu"
    if gpu and jax.default_backend() != "gpu":
        raise SystemExit("--device gpu but JAX backend is " + jax.default_backend())
    batch = a.batch or (4 if gpu else 2)
    seq_lens = a.seq_lens or ([512, 2048, 8192] if gpu else [512, 2048])
    dtypes = a.dtypes or (["bfloat16", "float32"] if gpu else ["float32"])
    impls = a.impls or (
        ["recurrent", "chunked", "chunked_fat", "pallas", "ssd"]
        if gpu
        else ["recurrent", "chunked", "chunked_fat", "ssd"]
    )
    max_rec = a.max_recurrent_seq or (8192 if gpu else 2048)

    rows = []
    if a.append and a.output and a.output.exists():
        rows = json.loads(a.output.read_text())["results"]
    for chunk in a.chunk_sizes:
        for dtype in dtypes:
            for d in a.dims:
                for seq in seq_lens:
                    names = [n for n in impls if n != "recurrent" or seq <= max_rec]
                    for mode in a.modes:
                        bench_case(
                            (batch, a.heads, seq, d, d, dtype),
                            mode,
                            names,
                            a.device,
                            a.repeats,
                            rows,
                            chunk,
                            with_err=not a.no_err,
                        )
    if a.peak:
        for row in list(rows):
            if "error" in row or "skipped" in row or "peak_mb" in row:
                continue
            spec = dict(
                impl=row["impl"],
                mode=row["mode"],
                chunk_size=row["chunk_size"],
                shape=[
                    row["batch"],
                    row["heads"],
                    row["seq"],
                    row["dk"],
                    row["dv"],
                    row["dtype"],
                ],
            )
            row.update(run_peak(spec))
            print("peak", spec, row.get("peak_mb"), flush=True)

    config = dict(
        device=a.device,
        batch=batch,
        heads=a.heads,
        seq_lens=seq_lens,
        dims=a.dims,
        dtypes=dtypes,
        impls=impls,
        chunk_sizes=a.chunk_sizes,
        repeats=a.repeats,
    )
    if a.output:
        write_results(a.output, config, rows)


if __name__ == "__main__":
    main()
