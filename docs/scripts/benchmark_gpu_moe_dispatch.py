"""MoE dropless backends on one GPU: probjax MoELayer forward and train step.

    python -m docs.scripts.benchmark_gpu_moe_dispatch --out results.jsonl --tokens 16384 131072 1048576

Decoder-sized MoE: d_model 512, 64 experts, top-8, expert width 128,
shared expert 1024, dropless routing, bf16 activations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.layers.moe import MoELayer

CONFIGS = [
    ("triton", "sorted"), ("triton", "gather"), ("triton", "fused"), ("triton", "fused_gsum"),
    ("mosaic", "sorted"), ("pallas", "sorted"), ("ragged_dot", "sorted"),
]


def _time(fn, args, reps):
    jax.block_until_ready(fn(*args)); jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(fn(*args)); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def build(impl, dispatch, d_model, width, experts, top_k, shared):
    return MoELayer(d_model, width, experts, top_k=top_k, routing="dropless", dropless_impl=impl,
                    dropless_dispatch=dispatch, shared_hidden=shared, dtype=jnp.bfloat16,
                    rngs=nnx.Rngs(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[16384, 131072, 1048576])
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--shared", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    dev = jax.devices()[0]
    meta = dict(label=a.label, gpu=dev.device_kind, jax=jax.__version__,
                d_model=a.d_model, width=a.width, experts=a.experts, top_k=a.top_k, shared=a.shared)
    for n in a.tokens:
        kx, kg = jax.random.split(jax.random.key(n))
        x = jax.random.normal(kx, (1, n, a.d_model), jnp.bfloat16)
        g = jax.random.normal(kg, (1, n, a.d_model), jnp.float32)
        ref = None
        for impl, dispatch in CONFIGS:
            rec = dict(meta, tokens=n, impl=impl, dispatch=dispatch)
            try:
                model = build(impl, dispatch, a.d_model, a.width, a.experts, a.top_k, a.shared)
                gd, params = nnx.split(model, nnx.Param)

                def fwd(params, x, gd=gd):
                    return nnx.merge(gd, params)(x)

                def loss(params, x, g=g, fwd=fwd):
                    y = fwd(params, x)
                    y = y[0] if isinstance(y, tuple) else y
                    return jnp.sum(y.astype(jnp.float32) * g)

                fj = jax.jit(fwd)
                step = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
                rec["fwd_ms"] = round(_time(fj, (params, x), a.reps), 3)
                rec["train_ms"] = round(_time(step, (params, x), a.reps), 3)
                y = fj(params, x)
                y = y[0] if isinstance(y, tuple) else y
                _, (gp, gx) = step(params, x)
                leaves = [l for l in jax.tree.leaves(gp)]
                if ref is None:
                    ref = (y, gx, leaves)
                    rec.update(out_err=0.0, gx_err=0.0, gw_err=0.0)
                else:
                    rel = lambda u, v: float(jnp.max(jnp.abs(u.astype(jnp.float32) - v.astype(jnp.float32)))
                                             / (jnp.max(jnp.abs(v.astype(jnp.float32))) + 1e-6))
                    rec.update(out_err=rel(y, ref[0]), gx_err=rel(gx, ref[1]),
                               gw_err=max(rel(u, v) for u, v in zip(leaves, ref[2], strict=True)))
                rec["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
                traceback.print_exc()
            print(json.dumps(rec), flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
