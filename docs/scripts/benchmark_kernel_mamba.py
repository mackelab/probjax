"""Benchmark the Mamba (S6) selective scan on CPU or GPU: time and memory.

Implementations (all fp32, same contract ``[B, L, D]`` / ``[S, D]``):

* ``scan``     -- ``_mamba_scan_reference``: sequential ``lax.scan``, autodiff
                  (saves every hidden state ``[L, B, S, D]`` for the VJP).
* ``assoc``    -- ``_mamba_scan_associative``: whole-sequence associative scan,
                  autodiff. This was ``compute_mamba_scan`` on CPU and on GPU
                  for every multi-tile shape before this lane ("before").
* ``chunked``  -- ``mamba_scan_chunked``: chunked XLA scan with a custom VJP
                  that recomputes states per chunk (the CPU path now; on GPU
                  measured with ``inner="assoc", chunk_size=64``).
* ``pallas``   -- ``compute_mamba_scan`` on GPU: Triton sequence-loop kernels
                  (the GPU path now).

Each row reports median/IQR time, XLA memory analysis (temp/argument/output
MB of the compiled executable), and relative error vs ``scan`` (max abs error
divided by max abs of the reference, max over y or over all six gradients).
``--peak`` additionally measures the true device peak of headline cases in
fresh subprocesses, and ``--max-len`` finds the largest L (fwd+bwd) that fits
on the device per implementation.

Run (hold the bench lock)::

    flock $LOCK python -m docs.scripts.benchmark_kernel_mamba --device gpu \
        --output docs/benchmarks/kernel_mamba_gpu.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

DEVICE_FLAG = "--device"
if DEVICE_FLAG in sys.argv and sys.argv[sys.argv.index(DEVICE_FLAG) + 1] == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from docs.scripts._gpu_bench import run_safely, time_jax, write_results  # noqa: E402
from probjax.nn.pallas_kernels import compute_mamba_scan  # noqa: E402
from probjax.nn.pallas_kernels.kernels import mamba as mamba_mod  # noqa: E402
from probjax.nn.pallas_kernels.kernels.mamba_chunked import (  # noqa: E402
    mamba_scan_chunked,
)

BATCH, STATE = 4, 16


def impls(device: str) -> dict:
    out = {
        "scan": mamba_mod._mamba_scan_reference,
        "assoc": mamba_mod._mamba_scan_associative,
        "chunked": mamba_scan_chunked,
    }
    if device == "gpu":
        # On GPU the fused per-step loop pays a kernel launch per step; the
        # in-chunk associative scan is the better pure-XLA variant there.
        out["chunked"] = functools.partial(
            mamba_scan_chunked, chunk_size=64, inner="assoc"
        )
        out["pallas"] = functools.partial(
            compute_mamba_scan, seq_tile_size=64, dim_tile_size=128
        )
    return out


def make_inputs(seq: int, dim: int, batch: int = BATCH, state: int = STATE):
    ks = jax.random.split(jax.random.key(seq + dim), 6)
    return (
        jax.random.normal(ks[0], (batch, seq, dim)),
        -jnp.abs(jax.random.normal(ks[1], (state, dim))),
        jax.random.normal(ks[2], (batch, seq, state)),
        jax.random.normal(ks[3], (batch, seq, state)),
        jax.nn.softplus(jax.random.normal(ks[4], (batch, seq, dim))),
        jax.random.normal(ks[5], (1, dim)),
    )


def jitted(fn, mode: str):
    if mode == "fwd":
        return jax.jit(fn)

    def loss(*xs):
        return 0.5 * jnp.sum(fn(*xs) ** 2)

    return jax.jit(jax.grad(loss, argnums=tuple(range(6))))


def memory_mb(jfn, args) -> dict:
    mem = jfn.lower(*args).compile().memory_analysis()
    mb = 2.0**20
    return dict(
        xla_temp_mb=round(mem.temp_size_in_bytes / mb, 2),
        xla_arg_mb=round(mem.argument_size_in_bytes / mb, 2),
        xla_out_mb=round(mem.output_size_in_bytes / mb, 2),
    )


def rel_err(out, ref) -> float:
    errs = []
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(ref), strict=True):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        errs.append(float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)))
    return max(errs)


def bench(device: str, seq_lens, dims, names, repeats: int) -> list[dict]:
    rows = []
    fns = impls(device)
    for dim in dims:
        for seq in seq_lens:
            args = make_inputs(seq, dim)
            for mode in ("fwd", "fwd_bwd"):
                ref = None
                for name in names:
                    if name not in fns:
                        continue

                    def case(name=name, mode=mode, args=args):
                        nonlocal ref
                        jfn = jitted(fns[name], mode)
                        row = memory_mb(jfn, args)
                        row.update(time_jax(jfn, *args, repeats=repeats))
                        out = jfn(*args)
                        if name == "scan":
                            ref = out
                        elif ref is not None:
                            row["rel_err_vs_scan"] = rel_err(out, ref)
                        return row

                    shape = dict(batch=BATCH, seq=seq, dim=dim, state=STATE)
                    row = dict(impl=name, mode=mode, shape=shape, **run_safely(case))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    return rows


def _child(spec: str) -> None:
    """Run one case in this (fresh) process and print the device peak."""
    name, mode, seq, dim = spec.split(":")
    device = "gpu" if jax.default_backend() == "gpu" else "cpu"
    args = make_inputs(int(seq), int(dim))
    jfn = jitted(impls(device)[name], mode)
    jax.block_until_ready(jfn(*args))
    stats = jax.devices()[0].memory_stats() or {}
    print("PEAK", json.dumps(dict(peak_mb=stats.get("peak_bytes_in_use", 0) / 2**20)))


def run_child(spec: str, device: str, timeout: int = 600) -> dict:
    env = dict(os.environ)
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run(
        [sys.executable, "-m", "docs.scripts.benchmark_kernel_mamba", "--child", spec],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PEAK "):
            return json.loads(line[5:])
    tail = (proc.stderr or "").strip().splitlines()[-1:] or ["failed"]
    return dict(error=tail[0][:300])


def peak_rows(device: str, cases) -> list[dict]:
    rows = []
    for name, mode, seq, dim in cases:
        res = run_child(f"{name}:{mode}:{seq}:{dim}", device)
        row = dict(
            impl=name,
            mode=mode,
            shape=dict(batch=BATCH, seq=seq, dim=dim, state=STATE),
            kind="device_peak",
            **res,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def max_len_rows(device: str, names, dim: int, start: int, limit: int) -> list[dict]:
    """Largest power-of-two L whose fwd+bwd runs (fresh process per try)."""
    rows = []
    for name in names:
        best, seq, err = None, start, None
        while seq <= limit:
            res = run_child(f"{name}:fwd_bwd:{seq}:{dim}", device)
            if "error" in res:
                err = res["error"]
                break
            best = dict(seq=seq, peak_mb=res["peak_mb"])
            seq *= 2
        row = dict(
            impl=name,
            mode="fwd_bwd",
            kind="max_len",
            shape=dict(batch=BATCH, dim=dim, state=STATE),
            max_seq=best and best["seq"],
            peak_mb_at_max=best and best["peak_mb"],
            limit=limit,
            first_failure=err,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seq-lens", default="512,2048,8192")
    parser.add_argument("--dims", default="256,512,1024")
    parser.add_argument("--impls", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--peak", action="store_true")
    parser.add_argument("--max-len", action="store_true")
    parser.add_argument("--max-len-limit", type=int, default=1 << 18)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        _child(args.child)
        return

    names = (args.impls or ",".join(impls(args.device))).split(",")
    rows = []
    if args.output and args.append and args.output.exists():
        rows = json.loads(args.output.read_text())["results"]
    seq_lens = [int(v) for v in args.seq_lens.split(",") if v]
    dims = [int(v) for v in args.dims.split(",") if v]
    if seq_lens and dims:
        rows += bench(args.device, seq_lens, dims, names, args.repeats)
    if args.peak:
        cases = [
            (n, m, s, 512)
            for n in names
            for m in ("fwd", "fwd_bwd")
            for s in (2048, 8192)
        ]
        rows += peak_rows(args.device, cases)
    if args.max_len:
        rows += max_len_rows(args.device, names, 1024, 1024, args.max_len_limit)
    if args.output:
        config = dict(
            batch=BATCH,
            state=STATE,
            dtype="float32",
            gpu_scan_config={k: dict(v) for k, v in mamba_mod._GPU_SCAN_CONFIG.items()},
            note="rel_err = max|out-scan| / max|scan| (max over y or all grads)",
        )
        write_results(args.output, config, rows)


if __name__ == "__main__":
    main()
