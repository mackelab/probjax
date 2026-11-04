"""Attention benchmark for the XLA (CPU) path of ``mha`` / ``mla`` (lane attn-cpu).

Run from the repository root, holding the shared bench lock::

    JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_attn-cpu \
        --device cpu --output docs/benchmarks/kernel_attn-cpu_cpu.json
    python -m docs.scripts.benchmark_kernel_attn-cpu \
        --device gpu --output docs/benchmarks/kernel_attn-cpu_gpu.json

Implementations (inputs are ``[batch, seq, heads, head_dim]``):

* ``probjax_mha``   -- ``probjax.nn.pallas_kernels.mha`` as dispatched on the
                       device (CPU: the XLA path, auto dense/tiled; GPU: Triton)
* ``xla_dense``     -- the XLA path forced to the dense strategy
* ``xla_tiled``     -- the XLA path forced to the tiled (flash-style) strategy
* ``mha_interpret`` -- ``mha(..., interpret=True)``, the previous CPU path
                       (Pallas interpret mode; small shapes only, it is slow)
* ``jax_xla``       -- ``jax.nn.dot_product_attention(implementation="xla")``
* ``torch_sdpa``    -- ``torch.nn.functional.scaled_dot_product_attention``
                       (CPU tensors on ``--device cpu``)

Variants: ``plain`` (no mask), ``causal`` and ``masked_bias`` (per-batch KV-length
mask composed with a causal mask + a learnable T5 relative-position bias whose table
receives gradients; torch gets the equivalent dense additive mask and no bias
gradient).  Modes ``fwd`` and ``fwdbwd`` (gradient of ``sum(out * ct)``).

Every row records the median/IQR time, ``xla_{temp,arg,out}_mb`` from the
compiled executable's ``memory_analysis()`` and the relative error against an
fp32 ``jax_xla`` reference (gradients included).  ``--peaks`` adds the peak
resident memory of headline cases, each run in a fresh subprocess, and
``--max-fit`` a compile-only memory sweep to long sequences.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from docs.scripts._gpu_bench import run_safely, time_jax, write_results
from probjax.nn.pallas_kernels import (
    CausalMask,
    KVLenMask,
    T5RelativePositionBias,
    mha,
)

try:
    from probjax.nn.pallas_kernels.kernels.attention_xla import xla_attention
except ImportError:  # pre-XLA-path checkouts (GPU before/after comparisons)
    xla_attention = None

MB = 2**20
NUM_BUCKETS = 32


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


def make_inputs(batch, seq, heads, dim, dtype, variant):
    keys = jax.random.split(jax.random.key(0), 5)
    q, k, v, ct = (
        jax.random.normal(kk, (batch, seq, heads, dim), jnp.float32).astype(dtype)
        for kk in keys[:4]
    )
    table = None
    lengths = None
    if variant == "masked_bias":
        table = 0.5 * jax.random.normal(keys[4], (heads, NUM_BUCKETS), jnp.float32)
        lengths = jnp.full((batch,), max(1, (3 * seq) // 4), jnp.int32)
    return dict(q=q, k=k, v=v, ct=ct, table=table, lengths=lengths)


def _mask_bias(variant, table, lengths):
    if variant == "plain":
        return None, None
    if variant == "causal":
        return CausalMask(), None
    mask = KVLenMask(lengths) & CausalMask()
    bias = T5RelativePositionBias(table, num_buckets=NUM_BUCKETS)
    return mask, bias


def _dense_mask_bias(variant, table, lengths, batch, seq, heads):
    """Dense ``[B,1,T,T]`` bool mask and ``[1,H,T,T]`` bias for the baselines."""
    _, bias = _mask_bias(variant, table, lengths)
    dmask = dbias = None
    if variant == "masked_bias":
        idx = jnp.arange(seq)
        causal = idx[None, :] <= idx[:, None]
        keep = idx[None, None, :] < lengths[:, None, None]
        dmask = (causal[None] & keep)[:, None]  # [B, 1, T, T]
        dbias = bias.dense(seq, seq, batch_size=1, num_heads=heads)
    return dmask, dbias


def jax_fn(impl, variant, scale, batch, seq, heads):
    """Return f(q, k, v, table, lengths) -> out for a JAX implementation."""

    def f(q, k, v, table, lengths):
        if impl == "jax_xla":
            dmask, dbias = _dense_mask_bias(variant, table, lengths, batch, seq, heads)
            return jax.nn.dot_product_attention(
                q,
                k,
                v,
                bias=None if dbias is None else dbias.astype(q.dtype),
                mask=dmask,
                scale=scale,
                is_causal=variant == "causal",
                implementation="xla",
            )
        mask, bias = _mask_bias(variant, table, lengths)
        if impl == "probjax_mha":
            return mha(q, k, v, mask=mask, bias=bias, sm_scale=scale)
        if impl == "mha_interpret":
            return mha(q, k, v, mask=mask, bias=bias, sm_scale=scale, interpret=True)
        strategy = {"xla_dense": "dense", "xla_tiled": "tiled"}[impl]
        return xla_attention(
            q, k, v, mask=mask, bias=bias, sm_scale=scale, strategy=strategy
        )

    return f


def build(impl, mode, variant, inputs, scale):
    q = inputs["q"]
    batch, seq, heads, _ = q.shape
    f = jax_fn(impl, variant, scale, batch, seq, heads)
    table, lengths = inputs["table"], inputs["lengths"]
    if mode == "fwd":
        fn = jax.jit(lambda q, k, v, table: f(q, k, v, table, lengths))
    else:
        argnums = (0, 1, 2, 3) if table is not None else (0, 1, 2)

        def loss(q, k, v, table):
            out = f(q, k, v, table, lengths)
            return jnp.sum(out.astype(jnp.float32) * inputs["ct"].astype(jnp.float32))

        fn = jax.jit(jax.grad(loss, argnums=argnums))
    return fn, (inputs["q"], inputs["k"], inputs["v"], table)


def rel_err(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    num = sum(
        float(jnp.sum((x.astype(jnp.float32) - y) ** 2))
        for x, y in zip(la, lb, strict=True)
    )
    den = sum(float(jnp.sum(y.astype(jnp.float32) ** 2)) for y in lb)
    return math.sqrt(num / max(den, 1e-30))


def memory_mb(compiled):
    try:
        ma = compiled.memory_analysis()
        return dict(
            xla_temp_mb=ma.temp_size_in_bytes / MB,
            xla_arg_mb=ma.argument_size_in_bytes / MB,
            xla_out_mb=ma.output_size_in_bytes / MB,
        )
    except Exception as exc:  # pragma: no cover - backend dependent
        return dict(memory_error=str(exc)[:200])


# ---------------------------------------------------------------------------
# torch baseline
# ---------------------------------------------------------------------------


def torch_case(mode, variant, inputs, scale, device):
    import torch
    import torch.nn.functional as F

    dev = "cuda" if device == "gpu" else "cpu"
    tdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
        str(inputs["q"].dtype)
    ]

    def tt(x, grad):
        arr = np.asarray(jnp.transpose(x, (0, 2, 1, 3)).astype(jnp.float32))
        return torch.tensor(arr, dtype=tdtype, device=dev, requires_grad=grad)

    grad = mode == "fwdbwd"
    q, k, v = (tt(inputs[n], grad) for n in ("q", "k", "v"))
    ct = tt(inputs["ct"], False)
    attn_mask = None
    if variant == "masked_bias":
        batch, seq, heads, _ = inputs["q"].shape
        dmask, dbias = _dense_mask_bias(
            variant, inputs["table"], inputs["lengths"], batch, seq, heads
        )
        additive = jnp.where(dmask, dbias, -jnp.inf)
        attn_mask = torch.tensor(np.asarray(additive), dtype=tdtype, device=dev)
    causal = variant == "causal"

    def run():
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, scale=scale
        )
        if grad:
            q.grad = k.grad = v.grad = None
            out.backward(ct)
        return out

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    def timed(n):
        sync()
        start = time.perf_counter()
        for _ in range(n):
            run()
        sync()
        return time.perf_counter() - start

    first = timed(1)
    single = max(timed(1), 1e-6)
    inner = int(min(max(0.05 / single, 1), 200))
    samples = np.asarray([timed(inner) for _ in range(5)]) / inner * 1e3
    q1, med, q3 = np.percentile(samples, [25, 50, 75])
    out = run()
    res = [x.grad for x in (q, k, v)] if grad else [out]
    res = [
        jnp.transpose(jnp.asarray(r.float().cpu().numpy()), (0, 2, 1, 3)) for r in res
    ]
    return dict(
        compile_s=first,
        inner=inner,
        median_ms=float(med),
        q1_ms=float(q1),
        q3_ms=float(q3),
    ), res


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def reference(mode, variant, inputs, scale):
    ref_inputs = {
        n: (x.astype(jnp.float32) if x is not None and n in "qkvct" else x)
        for n, x in inputs.items()
    }
    fn, args = build("jax_xla", mode, variant, ref_inputs, scale)
    return fn(*args)


def _ref_row(mode, variant, inputs, scale):
    return dict(r=reference(mode, variant, inputs, scale))


def run_case(impl, mode, variant, inputs, scale, device, ref):
    if impl == "torch_sdpa":
        stats, res = torch_case(mode, variant, inputs, scale, device)
        if ref is not None:
            ref_cmp = ref if mode == "fwd" else list(ref)[:3]
            stats["rel_err"] = rel_err(res if mode != "fwd" else res[0], ref_cmp)
        return stats
    fn, args = build(impl, mode, variant, inputs, scale)
    start = time.perf_counter()
    compiled = fn.lower(*args).compile()
    lower_compile_s = time.perf_counter() - start
    row = dict(lower_compile_s=lower_compile_s, **memory_mb(compiled))
    row.update(time_jax(compiled, *args, repeats=5))
    if ref is not None:
        row["rel_err"] = rel_err(compiled(*args), ref)
    return row


def sweep(args):
    rows = []
    impls = args.impls.split(",")
    for dtype_name in args.dtypes.split(","):
        dtype = jnp.dtype(dtype_name)
        for dim in [int(d) for d in args.dims.split(",")]:
            for seq in [int(s) for s in args.seqs.split(",")]:
                for variant in args.variants.split(","):
                    inputs = make_inputs(
                        args.batch, seq, args.heads, dim, dtype, variant
                    )
                    scale = 1.0 / math.sqrt(dim)
                    for mode in args.modes.split(","):
                        ref = None
                        if not args.no_ref:
                            ref = run_safely(
                                functools.partial(
                                    _ref_row, mode, variant, inputs, scale
                                )
                            ).get("r")
                        for impl in impls:
                            limit = (
                                args.interpret_max_seq
                                if mode == "fwd"
                                else args.interpret_max_seq // 2
                            )
                            if impl == "mha_interpret" and seq > limit:
                                continue
                            if (
                                impl == "xla_dense"
                                and args.batch * args.heads * seq * seq > 2**28
                            ):
                                continue
                            base = dict(
                                impl=impl,
                                mode=mode,
                                variant=variant,
                                dtype=dtype_name,
                                batch=args.batch,
                                seq=seq,
                                heads=args.heads,
                                head_dim=dim,
                            )
                            res = run_safely(
                                functools.partial(
                                    run_case,
                                    impl,
                                    mode,
                                    variant,
                                    inputs,
                                    scale,
                                    args.device,
                                    ref,
                                )
                            )
                            rows.append({**base, **res})
                            print(json.dumps(rows[-1]), flush=True)
    return rows


# ---------------------------------------------------------------------------
# Peak memory (fresh subprocess per case) and compile-only max-fit sweep
# ---------------------------------------------------------------------------


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def peak_case(spec):
    """Run one case once in this (fresh) process; report its memory peak."""
    impl, mode, variant = spec["impl"], spec["mode"], spec["variant"]
    inputs = make_inputs(
        spec["batch"], spec["seq"], spec["heads"], spec["dim"],
        jnp.dtype(spec["dtype"]), variant,
    )  # fmt: skip
    jax.block_until_ready([x for x in inputs.values() if x is not None])
    scale = 1.0 / math.sqrt(spec["dim"])
    out = dict(spec)
    if impl == "torch_sdpa":
        import torch  # noqa: F401 - import before the baseline reading

        base = _rss_mb()
        torch_case(mode, variant, inputs, scale, spec["device"])
    else:
        fn, args = build(impl, mode, variant, inputs, scale)
        compiled = fn.lower(*args).compile()
        out.update(memory_mb(compiled))
        base = _rss_mb()
        jax.block_until_ready(compiled(*args))
    peak = _rss_mb()
    out.update(baseline_rss_mb=base, peak_rss_mb=peak, peak_delta_mb=peak - base)
    if spec["device"] == "gpu" and impl != "torch_sdpa":
        stats = jax.devices()[0].memory_stats() or {}
        out["device_peak_mb"] = stats.get("peak_bytes_in_use", 0) / MB
    if spec["device"] == "gpu" and impl == "torch_sdpa":
        import torch

        out["device_peak_mb"] = torch.cuda.max_memory_allocated() / MB
    return out


def peaks(args):
    rows = []
    cases = []
    for seq in [int(s) for s in args.peak_seqs.split(",")]:
        for impl in args.peak_impls.split(","):
            cases.append(dict(impl=impl, seq=seq))
    for c in cases:
        spec = dict(
            impl=c["impl"],
            mode="fwdbwd",
            variant="plain",
            dtype="float32",
            batch=args.batch,
            seq=c["seq"],
            heads=args.heads,
            dim=64,
            device=args.device,
        )
        cmd = [sys.executable, "-m", "docs.scripts.benchmark_kernel_attn-cpu"]
        cmd += ["--peak-case", json.dumps(spec)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=args.peak_timeout
            )
            lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
            row = json.loads(lines[-1]) if lines else dict(spec)
            if not lines:
                tail = (proc.stderr or "").strip().splitlines()[-1:]
                row["error"] = f"exit {proc.returncode}: {' '.join(tail)[:300]}"
        except subprocess.TimeoutExpired:
            row = dict(spec, error=f"timeout after {args.peak_timeout}s")
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def max_fit(args):
    """Compile-only memory sweep of fwd+bwd to long sequences."""
    try:
        with open("/proc/meminfo") as fh:
            total_mb = int(fh.readline().split()[1]) / 1024.0
    except OSError:  # pragma: no cover
        total_mb = float("nan")
    rows = []
    for impl in args.fit_impls.split(","):
        for seq in [int(s) for s in args.fit_seqs.split(",")]:
            spec = dict(impl=impl, mode="fwdbwd", seq=seq, batch=args.batch)
            spec.update(heads=args.heads, head_dim=64, dtype="float32")

            def one(impl=impl, seq=seq):
                q = jax.ShapeDtypeStruct((args.batch, seq, args.heads, 64), jnp.float32)
                f = jax_fn(impl, "plain", 0.125, args.batch, seq, args.heads)

                def loss(q, k, v):
                    return jnp.sum(f(q, k, v, None, None))

                fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
                mem = memory_mb(fn.lower(q, q, q).compile())
                need = mem["xla_temp_mb"] + mem["xla_arg_mb"] + mem["xla_out_mb"]
                return dict(**mem, total_mb=need, fits_host_ram=need < total_mb)

            rows.append({**spec, **run_safely(one), "host_ram_mb": total_mb})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--output", type=Path)
    p.add_argument("--append", action="store_true", help="merge into --output")
    p.add_argument("--seqs", default="128,256,512,1024,2048,4096")
    p.add_argument("--dims", default="64,128")
    p.add_argument("--dtypes", default="float32")
    p.add_argument("--variants", default="plain,causal,masked_bias")
    p.add_argument("--modes", default="fwd,fwdbwd")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument(
        "--impls",
        default="probjax_mha,xla_dense,xla_tiled,mha_interpret,jax_xla,torch_sdpa",
    )
    p.add_argument("--interpret-max-seq", type=int, default=256)
    p.add_argument("--no-ref", action="store_true")
    p.add_argument("--peaks", action="store_true")
    p.add_argument("--peak-seqs", default="4096,8192,16384")
    p.add_argument("--peak-impls", default="xla_tiled,xla_dense,jax_xla,torch_sdpa")
    p.add_argument("--peak-timeout", type=int, default=300)
    p.add_argument("--max-fit", action="store_true")
    p.add_argument("--fit-seqs", default="4096,8192,16384,32768,65536,131072")
    p.add_argument("--fit-impls", default="xla_tiled,xla_dense,jax_xla")
    p.add_argument("--peak-case", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.peak_case:
        print(json.dumps(peak_case(json.loads(args.peak_case))), flush=True)
        return

    kind = "results"
    if args.peaks:
        rows, kind = peaks(args), "peaks"
    elif args.max_fit:
        rows, kind = max_fit(args), "max_fit"
    else:
        rows = sweep(args)
    if args.output is None:
        return
    config = {k: v for k, v in vars(args).items() if k not in ("output", "peak_case")}
    if args.append and args.output.exists():
        payload = json.loads(args.output.read_text())
        payload.setdefault(kind, []).extend(rows)
        payload.setdefault("configs", []).append(config)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"appended {len(rows)} {kind} rows to {args.output}", flush=True)
    elif kind == "results":
        write_results(args.output, config, rows)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({kind: rows, "config": config}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
