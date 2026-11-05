"""Benchmark of the MoE grouped-matmul kernels and ``MoELayer`` routing paths.

Run from the repository root (hold the shared bench lock for timings)::

    python -m docs.scripts.benchmark_kernel_gmm --device cpu --output docs/benchmarks/kernel_gmm_cpu.json
    python -m docs.scripts.benchmark_kernel_gmm --device gpu --output docs/benchmarks/kernel_gmm_gpu.json

Suites (``--suites``):

* ``layer``   -- ``MoELayer`` forward and train step (``value_and_grad`` of an
  MSE loss + AdamW, as in ``benchmark_gpu_nn``) for every routing / dropless
  implementation / dispatch, with XLA memory analysis (temp / argument /
  output MB) of both compiled functions and the relative error of the output
  and of all gradients against an fp32 ``routing="dense"`` reference.
* ``pieces``  -- train-step cost split: router + top-k, sort + permute,
  layout + scatter, the expert block (grouped matmuls), un-permute + combine.
* ``blockm``  -- ``block_m`` sweep of the Triton grouped GEMM (GPU).
* ``chunk``   -- row-chunk sweep of the CPU ``chunked`` dropless path.
* ``maxtok``  -- largest token count whose train step fits on the device, per
  implementation (token count doubled until it fails; each step runs in a
  fresh subprocess that reports ``peak_bytes_in_use``).
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
import optax
from flax import nnx

from docs.scripts._gpu_bench import run_safely, time_jax, write_results
from probjax.nn.layers import moe as moe_mod
from probjax.nn.layers.moe import MoELayer

MB = 2**20

# (routing, dropless_impl, dropless_dispatch)
GPU_IMPLS = [
    ("dropless", "pallas", "sorted"),
    ("dropless", "pallas", "gather"),
    ("dropless", "pallas", "fused"),
    ("dropless", "pallas", "fused_gsum"),
    ("dropless", "blocked", "sorted"),
    ("dropless", "ragged_dot", "sorted"),
    ("capacity", "auto", "sorted"),
    ("dense", "auto", "sorted"),
]
CPU_IMPLS = [
    ("dropless", "chunked", "sorted"),
    ("dropless", "blocked", "sorted"),
    ("dropless", "ragged_dot", "sorted"),
    ("capacity", "auto", "sorted"),
    ("dense", "auto", "sorted"),
]


def make_layer(cfg, routing, impl, dispatch, dtype=None, seed=0):
    return MoELayer(
        cfg["d_model"],
        cfg["d_hidden"],
        cfg["experts"],
        top_k=cfg["top_k"],
        routing=routing,
        dropless_impl=impl,
        dropless_dispatch=dispatch,
        dtype=dtype,
        rngs=nnx.Rngs(seed),
    )


def make_inputs(cfg, model, dtype):
    """Tokens ``x``; with ``skew > 0`` a shared offset that the router maps
    onto expert 0, so about that fraction of assignments pile onto it."""
    x = jax.random.normal(jax.random.key(1), (cfg["tokens"], cfg["d_model"]))
    skew = cfg.get("skew", 0.0)
    if skew:
        v = jax.random.normal(jax.random.key(2), (cfg["d_model"],))
        v = v / jnp.linalg.norm(v)
        k = model.router.kernel[...]
        k = k - jnp.outer(v, v @ k)  # remove existing v-component
        model.router.kernel[...] = k.at[:, 0].add(v * 8.0)
        hot = jax.random.uniform(jax.random.key(3), (cfg["tokens"],)) < skew
        x = x + jnp.where(hot[:, None], 3.0, 0.0) * v
    return x.astype(dtype)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _mem(jitted, *args):
    try:
        ma = jitted.lower(*args).compile().memory_analysis()
        return dict(
            xla_temp_mb=ma.temp_size_in_bytes / MB,
            xla_arg_mb=ma.argument_size_in_bytes / MB,
            xla_out_mb=ma.output_size_in_bytes / MB,
        )
    except Exception as exc:  # noqa: BLE001
        return dict(mem_error=str(exc)[:200])


def build_fns(model):
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    tx = optax.adamw(1e-4)

    def predict(params, rest, x):
        return nnx.merge(graphdef, params, rest)(x)

    def loss_fn(params, rest, x):
        return jnp.mean(jnp.square(predict(params, rest, x).astype(jnp.float32)))

    forward = jax.jit(predict)

    @jax.jit
    def train_step(params, opt_state, rest, x):
        loss, grads = jax.value_and_grad(loss_fn)(params, rest, x)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 2)))
    return params, rest, tx.init(params), forward, train_step, grad_fn


def _reference(cfg, x):
    """fp32 dense routing with the same parameters (seed 0), HIGHEST precision."""
    with jax.default_matmul_precision("highest"):
        m = make_layer(cfg, "dense", "auto", "sorted")
        xr = make_inputs(cfg, m, jnp.float32)
        params, rest, _, fwd, _, grad_fn = build_fns(m)
        return jax.device_get((fwd(params, rest, xr), grad_fn(params, rest, xr)))


def layer_case(
    cfg, routing, impl, dispatch, dtype_name, check=True, ops=("forward", "train_step")
):
    dtype = jnp.dtype(dtype_name)
    meta = dict(
        suite="layer",
        routing=routing,
        impl=impl,
        dispatch=dispatch,
        dtype=dtype_name,
        **cfg,
    )
    m = make_layer(
        cfg, routing, impl, dispatch, None if dtype == jnp.float32 else dtype
    )
    x = make_inputs(cfg, m, dtype)
    params, rest, opt, fwd, step, grad_fn = build_fns(m)
    counts = jnp.bincount(
        jax.lax.top_k(x.astype(jnp.float32) @ m.router.kernel[...], cfg["top_k"])[
            1
        ].ravel(),
        length=cfg["experts"],
    )
    meta["max_expert_frac"] = float(counts.max() / counts.sum())
    meta["empty_experts"] = int((counts == 0).sum())
    rows = []
    for op, fn, args in (
        ("forward", fwd, (params, rest, x)),
        ("train_step", step, (params, opt, rest, x)),
    ):
        if op not in ops:
            continue
        row = dict(**meta, op=op)
        row.update(
            run_safely(
                lambda fn=fn, args=args: dict(**_mem(fn, *args), **time_jax(fn, *args))
            )
        )
        rows.append(row)
    if check and routing != "capacity":

        def err():
            y_ref, (gp_ref, gx_ref) = _reference(cfg, x)
            y = fwd(params, rest, x)
            gp, gx = grad_fn(params, rest, x)
            gerr = max(
                _rel(a, b)
                for a, b in zip(
                    jax.tree.leaves(gp), jax.tree.leaves(gp_ref), strict=True
                )
            )
            return dict(
                rel_err_out=_rel(y, y_ref),
                rel_err_grad_params=gerr,
                rel_err_grad_x=_rel(gx, gx_ref),
            )

        e = run_safely(err)
        for r in rows:
            r.update(e)
    return rows


def _p(row):
    print(
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
        flush=True,
    )


def suite_layer(rows, args):
    impls = GPU_IMPLS if args.device == "gpu" else CPU_IMPLS
    if args.impls:
        impls = [tuple(s.split(":")) for s in args.impls]
    for dtype in args.dtypes:
        for tokens in args.tokens:
            for experts in args.experts:
                for top_k in args.top_k:
                    for skew in args.skew:
                        cfg = dict(
                            tokens=tokens,
                            d_model=args.d_model,
                            d_hidden=args.d_hidden,
                            experts=experts,
                            top_k=top_k,
                            skew=skew,
                        )
                        for routing, impl, dispatch in impls:
                            if (
                                routing == "dense"
                                and tokens * experts > args.dense_limit
                            ):
                                continue
                            if (
                                args.device == "cpu"
                                and impl == "ragged_dot"
                                and tokens * experts > args.ragged_limit
                            ):
                                continue
                            for r in layer_case(
                                cfg, routing, impl, dispatch, dtype, check=args.check
                            ):
                                rows.append(r)
                                _p(r)
                            jax.clear_caches()


def suite_pieces(rows, args):
    """Time the train-step pieces of the sorted dropless path separately."""
    for dtype_name in args.dtypes:
        for tokens in args.tokens:
            cfg = dict(
                tokens=tokens,
                d_model=args.d_model,
                d_hidden=args.d_hidden,
                experts=args.experts[0],
                top_k=args.top_k[0],
            )
            _pieces(rows, cfg, dtype_name, args.block_m)


def _pieces(rows, cfg, dtype_name, block_m):
    from probjax.nn.layers.grouped_matmul import grouped_swiglu

    dtype = jnp.dtype(dtype_name)
    m = make_layer(
        cfg,
        "dropless",
        "pallas",
        "sorted",
        None if dtype == jnp.float32 else dtype,
    )
    x = make_inputs(cfg, m, dtype)
    wr = m.router.kernel[...]
    wg, wu, wd = (
        w.kernel[...].astype(dtype)
        for w in (m.experts.gate, m.experts.up, m.experts.down)
    )
    k = cfg["top_k"]

    def route(x, wr):
        logits = x.astype(jnp.float32) @ wr
        top, ids = jax.lax.top_k(logits, k)
        return (
            jax.nn.softmax(top, -1),
            ids,
            jnp.bincount(ids.ravel(), length=cfg["experts"]),
        )

    def perm(x, ids):
        order = jnp.argsort(ids.reshape(-1), stable=True)
        inv = jnp.argsort(order)
        xr = jnp.broadcast_to(x[:, None], (x.shape[0], k, x.shape[1])).reshape(
            -1, x.shape[1]
        )
        return moe_mod._permute(xr, order, inv), order, inv

    w, ids, counts = jax.jit(route)(x, wr)
    xs, order, inv = jax.jit(perm)(x, ids)

    def experts(xs, wg, wu, wd):
        return grouped_swiglu(xs, wg, wu, wd, counts, block_m, False)

    def unperm(ys, w):
        y = moe_mod._permute(ys, inv, order).reshape(x.shape[0], k, -1)
        return jnp.sum(y * w[..., None].astype(y.dtype), 1)

    def vg(f, n):
        return jax.jit(
            jax.value_and_grad(
                lambda *a: jnp.sum(f(*a).astype(jnp.float32) ** 2),
                argnums=tuple(range(n)),
            )
        )

    pieces = {
        "route": (
            jax.jit(
                jax.grad(lambda x, wr: jnp.sum(route(x, wr)[0] ** 2), argnums=(0, 1))
            ),
            (x, wr),
        ),
        "sort_permute": (vg(lambda x: perm(x, ids)[0], 1), (x,)),
        "experts_grouped_swiglu": (vg(experts, 4), (xs, wg, wu, wd)),
        "unpermute_combine": (vg(unperm, 2), (xs, w)),
    }
    for name, (fn, a) in pieces.items():
        row = dict(
            suite="pieces",
            piece=name,
            dtype=dtype_name,
            op="fwd+bwd",
            block_m=block_m,
            **cfg,
        )
        row.update(
            run_safely(lambda fn=fn, a=a: dict(**_mem(fn, *a), **time_jax(fn, *a)))
        )
        rows.append(row)
        _p(row)


def suite_blockm(rows, args):
    from probjax.nn.layers import moe

    for dtype in args.dtypes:
        for tokens in args.tokens:
            for bm in args.block_ms:
                cfg = dict(
                    tokens=tokens,
                    d_model=args.d_model,
                    d_hidden=args.d_hidden,
                    experts=args.experts[0],
                    top_k=args.top_k[0],
                )
                orig = moe.ExpertSwiGLU.grouped_forward.__kwdefaults__["block_m"]
                moe.ExpertSwiGLU.grouped_forward.__kwdefaults__["block_m"] = bm
                try:
                    for r in layer_case(
                        cfg, "dropless", "pallas", "sorted", dtype, check=False
                    ):
                        r.update(suite="blockm", block_m=bm)
                        rows.append(r)
                        _p(r)
                finally:
                    moe.ExpertSwiGLU.grouped_forward.__kwdefaults__["block_m"] = orig
                jax.clear_caches()


def suite_chunk(rows, args):
    """Row-chunk sweep of the CPU ``chunked`` path."""
    from probjax.nn.layers import moe

    kw = moe.ExpertSwiGLU.chunked_forward.__kwdefaults__
    for dtype in args.dtypes:
        for tokens in args.tokens:
            for experts in args.experts:
                cfg = dict(
                    tokens=tokens,
                    d_model=args.d_model,
                    d_hidden=args.d_hidden,
                    experts=experts,
                    top_k=args.top_k[0],
                )
                for c in args.chunks:
                    orig, kw["chunk"] = kw["chunk"], c
                    try:
                        for r in layer_case(
                            cfg, "dropless", "chunked", "sorted", dtype, check=False
                        ):
                            r.update(suite="chunk", chunk=c)
                            rows.append(r)
                            _p(r)
                    finally:
                        kw["chunk"] = orig
                    jax.clear_caches()


def _single_peak(spec):
    """Child process: run one train step, print device peak bytes."""
    cfg, routing, impl, dispatch, dtype = spec
    m = make_layer(
        cfg, routing, impl, dispatch, None if dtype == "float32" else jnp.dtype(dtype)
    )
    x = make_inputs(cfg, m, jnp.dtype(dtype))
    params, rest, opt, _, step, _ = build_fns(m)
    jax.block_until_ready(step(params, opt, rest, x))
    stats = jax.devices()[0].memory_stats() or {}
    print("PEAK", json.dumps(dict(peak_mb=stats.get("peak_bytes_in_use", 0) / MB)))


def _peak_subprocess(spec):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "docs.scripts.benchmark_kernel_gmm",
            "--single",
            json.dumps(spec),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    for line in out.stdout.splitlines():
        if line.startswith("PEAK "):
            return json.loads(line[5:])
    tail = (out.stderr.strip().splitlines() or ["?"])[-1]
    return dict(error=tail[:300])


def suite_maxtok(rows, args):
    """Largest token count (powers of two) whose train step runs, per impl."""
    impls = GPU_IMPLS if args.device == "gpu" else CPU_IMPLS
    if args.impls:
        impls = [tuple(s.split(":")) for s in args.impls]
    for dtype in args.dtypes:
        for routing, impl, dispatch in impls:
            best, tokens = None, args.maxtok_start
            while tokens <= args.maxtok_stop:
                cfg = dict(
                    tokens=tokens,
                    d_model=args.d_model,
                    d_hidden=args.d_hidden,
                    experts=args.experts[0],
                    top_k=args.top_k[0],
                    skew=0.0,
                )
                res = _peak_subprocess([cfg, routing, impl, dispatch, dtype])
                row = dict(
                    suite="maxtok",
                    routing=routing,
                    impl=impl,
                    dispatch=dispatch,
                    dtype=dtype,
                    op="train_step",
                    **cfg,
                    **res,
                )
                rows.append(row)
                _p(row)
                if "error" in res:
                    break
                best, tokens = tokens, tokens * 2
            rows.append(
                dict(
                    suite="maxtok_summary",
                    routing=routing,
                    impl=impl,
                    dispatch=dispatch,
                    dtype=dtype,
                    max_tokens=best,
                )
            )
            _p(rows[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--suites", nargs="+", default=["layer"])
    parser.add_argument("--dtypes", nargs="+", default=["float32"])
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096])
    parser.add_argument("--experts", type=int, nargs="+", default=[8])
    parser.add_argument("--top-k", type=int, nargs="+", default=[2])
    parser.add_argument("--skew", type=float, nargs="+", default=[0.0])
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-hidden", type=int, default=1024)
    parser.add_argument("--impls", nargs="+", help="routing:impl:dispatch triples")
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-ms", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--chunks", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument(
        "--dense-limit", type=int, default=2**19, help="max tokens*experts for dense"
    )
    parser.add_argument(
        "--ragged-limit",
        type=int,
        default=2**18,
        help="max tokens*experts for CPU ragged_dot",
    )
    parser.add_argument("--maxtok-start", type=int, default=32768)
    parser.add_argument("--maxtok-stop", type=int, default=2**21)
    parser.add_argument(
        "--slab-elems",
        type=int,
        help="override grouped_matmul.SLAB_ELEMS (0: one slab)",
    )
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--single", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.slab_elems is not None:
        from probjax.nn.layers import grouped_matmul

        grouped_matmul.SLAB_ELEMS = args.slab_elems or None
    if args.single:
        _single_peak(json.loads(args.single))
        return
    if args.device == "cpu" and jax.default_backend() != "cpu":
        raise SystemExit("run with JAX_PLATFORMS=cpu for --device cpu")
    rows: list[dict] = []
    for suite in args.suites:
        globals()[f"suite_{suite}"](rows, args)
    if args.output:
        cfg = {k: v for k, v in vars(args).items() if k not in ("output", "single")}
        write_results(args.output, cfg, rows)


if __name__ == "__main__":
    main()
