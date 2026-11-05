"""Benchmark of the SSD (Mamba-2 state-space duality) implementations.

Run from the repository root (hold the shared bench lock for timings)::

    python -m docs.scripts.benchmark_kernel_ssd --device gpu \\
        --output docs/benchmarks/kernel_ssd_gpu.json
    JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_ssd --device cpu \\
        --output docs/benchmarks/kernel_ssd_cpu.json

Implementations:

* ``chunked`` -- pure-JAX chunked SSD (``ssd_chunked``; default for CPU and the
  GPU fallback). Variants (``--variants``, e.g. ``chunked_c128_none_matmul``)
  sweep chunk size / remat / inter-chunk propagation / matmul precision.
* ``scan``    -- ``ssd_linear_scan`` (per-step ``lax.scan``; needs dk % 128 == 0).
* ``pallas``  -- the Triton Pallas kernel (GPU; only shapes it supports).

Every row records time (median/IQR), XLA ``memory_analysis`` (temp/arg/out MB),
for ``fwd_bwd`` the residuals saved by ``jax.vjp`` (MB), and relative errors vs
an fp32 ``HIGHEST``-precision chunked reference (forward output and each
gradient; ``max|a-b| / max|b|``). ``--peak`` runs headline cases in fresh
subprocesses to record the true device peak (GPU), and ``--max-len`` searches
the largest power-of-two sequence length whose fwd+bwd fits on the device.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from docs.scripts._gpu_bench import run_safely, time_jax, write_results
from probjax.nn.pallas_kernels.kernels.ssd_xla import ssd_chunked

ssd_mod = importlib.import_module("probjax.nn.pallas_kernels.kernels.ssd")

MB = 2**20

# name: (batch, groups, heads, dk, dv)
CONFIGS = {
    # Pallas-supported reference config (also the old gpu_kernels config).
    "g2h8_d128": (4, 2, 8, 128, 128),
    # Mamba-2-like: one q/k group, many value heads, headdim 64.
    "g1h24_dk128_dv64": (2, 1, 24, 128, 64),
    # SSDCell(num_heads=1): rejected by the Triton kernel (num_heads < 2).
    "g1h1_d128": (4, 1, 1, 128, 128),
    # Small dims, not multiples of 64/128: neither Pallas nor scan support it.
    "g2h4_dk48_dv40": (4, 2, 4, 48, 40),
}

def variant_kwargs(name: str) -> dict:
    """``chunked[_c<C>][_states|_none|_full][_matmul][_tf32]`` -> ssd_chunked kwargs."""
    kw: dict = {}
    for tok in name.split("_")[1:]:
        if tok.startswith("c") and tok[1:].isdigit():
            kw["chunk_size"] = int(tok[1:])
        elif tok in ("states", "none", "full"):
            kw["remat"] = tok
        elif tok == "matmul":
            kw["inter_chunk"] = "matmul"
        elif tok == "tf32":
            kw["precision"] = "tensorfloat32"
        else:
            raise ValueError(f"unknown chunked variant token {tok!r} in {name!r}")
    return kw


def make_inputs(cfg, seq, seed=0):
    b, g, h, dk, dv = CONFIGS[cfg]
    ks = jax.random.split(jax.random.key(seed), 6)
    q = jax.random.normal(ks[0], (b, g, seq, dk)) / np.sqrt(dk)
    k = jax.random.normal(ks[1], (b, g, seq, dk))
    v = jax.random.normal(ks[2], (b, h, seq, dv))
    # Decays in (e^-1, 1): typical Mamba-2 range (softplus(dt) * -exp(A)).
    log_alpha = -jax.nn.sigmoid(jax.random.normal(ks[3], (b, h, seq)))
    h0 = jax.random.normal(ks[4], (b, h, dk, dv)) * 0.1
    ct = jax.random.normal(ks[5], (b, h, seq, dv))
    return (q, k, v, log_alpha, h0), ct


def impl_fn(name):
    if name == "scan":
        return lambda q, k, v, a, h0: ssd_mod.ssd_linear_scan(q, k, v, a, h0)[0]
    if name == "pallas":
        return lambda q, k, v, a, h0: ssd_mod.ssd(
            q, k, v, a, h0, implementation="pallas"
        )
    if name == "reference":
        return functools.partial(ssd_chunked, remat="none")
    return functools.partial(ssd_chunked, **variant_kwargs(name))


def supported(name, cfg, seq, device):
    b, g, h, dk, dv = CONFIGS[cfg]
    if name == "scan":
        return dk % 128 == 0
    if name == "pallas":
        return device == "gpu" and ssd_mod._gpu_supports_ssd_pallas_for_shape(
            seq_len=seq,
            num_groups=g,
            num_heads=h,
            dk=dk,
            dv=dv,
            pallas_backend=ssd_mod._pallas_backend(),
        )
    return True


def build(name, mode, ct):
    fn = impl_fn(name)
    if mode == "fwd":
        return jax.jit(fn)

    def loss(*args):
        return jnp.vdot(fn(*args), ct)

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))


def mem_analysis(jfn, args):
    ma = jfn.lower(*args).compile().memory_analysis()
    if ma is None:
        return {}
    return dict(
        xla_temp_mb=ma.temp_size_in_bytes / MB,
        xla_arg_mb=ma.argument_size_in_bytes / MB,
        xla_out_mb=ma.output_size_in_bytes / MB,
    )


def residual_mb(name, args):
    """Bytes held by the vjp closure that are not the primal inputs."""
    _, vjp = jax.vjp(impl_fn(name), *args)
    ids = {id(a) for a in args}
    leaves = [x for x in jax.tree.leaves(vjp) if hasattr(x, "nbytes")]
    return sum(x.nbytes for x in leaves if id(x) not in ids) / MB


def rel_err(out, ref):
    errs = []
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(ref), strict=True):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        errs.append(float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)))
    return max(errs), errs


def bench_case(cfg, seq, mode, names, device, repeats, rows, skip_err=False):
    args, ct = make_inputs(cfg, seq)
    b, g, h, dk, dv = CONFIGS[cfg]
    ref = None
    if not skip_err:
        ref = run_safely(lambda: dict(out=build("reference", mode, ct)(*args)))
        ref = ref.get("out")
    for name in names:
        base = dict(
            impl=name,
            mode=mode,
            config=cfg,
            batch=b,
            groups=g,
            heads=h,
            seq=seq,
            dk=dk,
            dv=dv,
        )
        if not supported(name, cfg, seq, device):
            rows.append(dict(base, skipped="unsupported shape"))
            continue

        def case(name=name):
            jfn = build(name, mode, ct)
            res = time_jax(jfn, *args, repeats=repeats)
            res.update(mem_analysis(jfn, args))
            if mode == "fwd_bwd":
                res["residual_mb"] = residual_mb(name, args)
            if ref is not None:
                res["rel_err"], per = rel_err(jfn(*args), ref)
                if mode == "fwd_bwd":
                    res["rel_err_grads"] = dict(
                        zip(("q", "k", "v", "log_alpha", "h0"), per, strict=True)
                    )
            return res

        row = dict(base, **run_safely(case))
        rows.append(row)
        print(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
            flush=True,
        )


# ----------------------------------------------------------------- subprocess


def _peak_child(spec):
    """Run one case (compile + 2 calls) and print device peak bytes."""
    args, ct = make_inputs(spec["config"], spec["seq"])
    jfn = build(spec["impl"], spec["mode"], ct)
    jax.block_until_ready(jfn(*args))
    jax.block_until_ready(jfn(*args))
    stats = jax.devices()[0].memory_stats() or {}
    print(
        "PEAK_JSON " + json.dumps(dict(peak_mb=stats.get("peak_bytes_in_use", 0) / MB))
    )


def run_peak(spec, timeout=600):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "docs.scripts.benchmark_kernel_ssd",
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
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--seq-lens", nargs="+", type=int, default=None)
    p.add_argument("--impls", nargs="+", default=None)
    p.add_argument("--variants", nargs="*", default=[], help="extra chunked variants")
    p.add_argument("--modes", nargs="+", default=["fwd", "fwd_bwd"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--max-scan-seq",
        type=int,
        default=None,
        help="skip scan above this length (CPU: slow)",
    )
    p.add_argument("--peak", action="store_true", help="device peak via subprocesses")
    p.add_argument("--max-len", action="store_true", help="largest L for fwd+bwd")
    p.add_argument("--peak-child", type=str, default=None, help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.peak_child:
        _peak_child(json.loads(a.peak_child))
        return
    if jax.default_backend() != a.device:
        raise SystemExit(
            f"--device {a.device} but JAX backend is {jax.default_backend()}"
        )

    seq_lens = a.seq_lens or ([512, 2048, 8192] if a.device == "gpu" else [512, 2048])
    impls = a.impls or (
        ["chunked", "scan", "pallas"] if a.device == "gpu" else ["chunked", "scan"]
    )
    impls = impls + list(a.variants)
    rows: list[dict] = []

    if not (a.peak or a.max_len):
        for cfg in a.configs:
            for seq in seq_lens:
                for mode in a.modes:
                    names = [
                        n
                        for n in impls
                        if not (n == "scan" and a.max_scan_seq and seq > a.max_scan_seq)
                    ]
                    bench_case(cfg, seq, mode, names, a.device, a.repeats, rows)

    if a.peak:
        for cfg in a.configs:
            for seq in seq_lens:
                for mode in a.modes:
                    for name in impls:
                        if not supported(name, cfg, seq, a.device):
                            continue
                        spec = dict(impl=name, mode=mode, config=cfg, seq=seq)
                        row = dict(kind="peak", **spec, **run_peak(spec))
                        rows.append(row)
                        print(row, flush=True)

    if a.max_len:
        for cfg in a.configs:
            for name in impls:
                best, seq, err = None, 1024, None
                while seq <= 2**20:
                    if not supported(name, cfg, seq, a.device):
                        break
                    spec = dict(impl=name, mode="fwd_bwd", config=cfg, seq=seq)
                    res = run_peak(spec)
                    if "error" in res:
                        err = res["error"]
                        break
                    best = dict(seq=seq, peak_mb=res["peak_mb"])
                    seq *= 2
                row = dict(
                    kind="max_len",
                    impl=name,
                    mode="fwd_bwd",
                    config=cfg,
                    largest_ok=best,
                    first_failure_error=err,
                )
                rows.append(row)
                print(row, flush=True)

    if a.output:
        write_results(
            a.output,
            dict(
                device=a.device,
                configs={c: CONFIGS[c] for c in a.configs},
                seq_lens=seq_lens,
                impls=impls,
                repeats=a.repeats,
            ),
            rows,
        )


if __name__ == "__main__":
    main()
