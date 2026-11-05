"""GPU benchmark of probjax's non-softmax Pallas kernels against their references.

Run from the repository root (requires a CUDA GPU)::

    python -m docs.scripts.benchmark_gpu_kernels \
        --output docs/benchmarks/gpu_kernels.json

Families and the baseline each kernel is compared against:

* ``ssd``        -- Pallas SSD (Mamba-2 state-space duality) vs ``ssd_linear_scan``
* ``mamba``      -- Pallas selective scan vs the ``lax.scan`` reference
* ``rla``        -- residual linear attention, Pallas chunked vs ``lax.scan``
* ``kernel_mv``  -- tiled RBF kernel-matrix-vector product vs dense ``K @ v``
* ``mla``        -- latent attention over a shared cache vs explicitly
                    up-projected K/V fed to cuDNN / XLA attention

Each case records forward and forward+backward timings plus the max abs error
vs the reference implementation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from docs.scripts._gpu_bench import max_abs_err, run_safely, time_jax, write_results
from probjax.nn.pallas_kernels import (
    compute_mamba_scan,
    mla,
    rbf_kernel_mv,
    rbf_kernel_mv_naive,
    residual_linear_attention_pallas,
    residual_linear_attention_scan,
    ssd,
    ssd_linear_scan,
)
from probjax.nn.pallas_kernels.kernels.mamba import _mamba_scan_reference


def _grad_fn(fn, argnums):
    def loss(*args):
        out = fn(*args)
        out = out[0] if isinstance(out, tuple) else out
        return jnp.sum(out.astype(jnp.float32) ** 2) * 0.5

    return jax.jit(jax.grad(loss, argnums=argnums))


def _compare(rows, family, shape, impls, args, argnums, ref_name):
    """Time ``impls`` (name -> fn) on ``args``; errors are vs ``impls[ref_name]``."""
    refs = {}
    for mode in ("fwd", "fwd_bwd"):
        for name, fn in impls.items():

            def case(name=name, fn=fn, mode=mode):
                jfn = jax.jit(fn) if mode == "fwd" else _grad_fn(fn, argnums)
                timing = time_jax(jfn, *args)  # first: so compile_s includes compilation
                out = jfn(*args)
                out = out[0] if (mode == "fwd" and isinstance(out, tuple)) else out
                if name == ref_name:
                    refs[mode] = out
                err = None
                if mode in refs:
                    err = max(
                        max_abs_err(a, b)
                        for a, b in zip(
                            jax.tree.leaves(out), jax.tree.leaves(refs[mode]), strict=True
                        )
                    )
                return dict(max_abs_err=err, **timing)

            row = dict(family=family, impl=name, mode=mode, **shape, **run_safely(case))
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


def bench_ssd(rows, seq_lens):
    for seq in seq_lens:
        b, ng, nh, dk, dv = 4, 2, 8, 128, 128  # ssd_linear_scan needs dk % 128 == 0
        ks = jax.random.split(jax.random.key(seq), 4)
        q = jax.random.normal(ks[0], (b, ng, seq, dk)) / math.sqrt(dk)
        k = jax.random.normal(ks[1], (b, ng, seq, dk)) / math.sqrt(dk)
        v = jax.random.normal(ks[2], (b, nh, seq, dv))
        log_alpha = -jax.nn.softplus(jax.random.normal(ks[3], (b, nh, seq)))
        impls = {
            "scan": lambda q, k, v, a: ssd_linear_scan(q, k, v, a)[0],
            "pallas": lambda q, k, v, a: ssd(q, k, v, a),
        }
        _compare(rows, "ssd", dict(batch=b, seq=seq, heads=nh, dk=dk, dv=dv),
                 impls, (q, k, v, log_alpha), (0, 1, 2, 3), "scan")


def bench_mamba(rows, seq_lens):
    for seq in seq_lens:
        b, dim, state = 4, 512, 16
        ks = jax.random.split(jax.random.key(seq + 1), 6)
        x = jax.random.normal(ks[0], (b, seq, dim))
        a = -jnp.abs(jax.random.normal(ks[1], (state, dim)))
        bb = jax.random.normal(ks[2], (b, seq, state))
        c = jax.random.normal(ks[3], (b, seq, state))
        delta = jax.nn.softplus(jax.random.normal(ks[4], (b, seq, dim)))
        d = jax.random.normal(ks[5], (1, dim))
        impls = {
            "scan": lambda *xs: _mamba_scan_reference(*xs),
            "pallas": lambda *xs: compute_mamba_scan(*xs, seq_tile_size=64, dim_tile_size=128),
        }
        _compare(rows, "mamba", dict(batch=b, seq=seq, dim=dim, state=state),
                 impls, (x, a, bb, c, delta, d), (0, 2, 3, 4), "scan")


def bench_rla(rows, seq_lens):
    for seq in seq_lens:
        b, h, dim = 4, 8, 64
        ks = jax.random.split(jax.random.key(seq + 2), 4)
        q, k, v = (jax.random.normal(kk, (b, h, seq, dim)) / math.sqrt(dim) for kk in ks[:3])
        state_shape = jax.eval_shape(
            lambda q, k, v: residual_linear_attention_scan(
                q, k, v, jnp.zeros((b, h, 2 * dim, dim)), window_size=64
            )[1],
            q, k, v,
        ).shape
        h0 = 0.1 * jax.random.normal(ks[3], state_shape)
        impls = {
            "scan": lambda q, k, v, h0: residual_linear_attention_scan(q, k, v, h0, window_size=64),
            "pallas": lambda q, k, v, h0: residual_linear_attention_pallas(
                q, k, v, h0, window_size=64, chunk_size=128
            ),
        }
        _compare(rows, "rla", dict(batch=b, seq=seq, heads=h, dim=dim),
                 impls, (q, k, v, h0), (0, 1, 2), "scan")


def bench_kernel_mv(rows, sizes):
    for n in sizes:
        b, d, o = 1, 8, 16
        ks = jax.random.split(jax.random.key(n), 3)
        q = jax.random.normal(ks[0], (b, n, d))
        k = jax.random.normal(ks[1], (b, n, d))
        v = jax.random.normal(ks[2], (b, n, o))
        ls = jnp.asarray(1.0)
        impls = {
            "dense": lambda q, k, v: rbf_kernel_mv_naive(q, k, v, ls),
            "pallas": lambda q, k, v: rbf_kernel_mv(q, k, v, ls),
        }
        _compare(rows, "kernel_mv", dict(batch=b, n=n, dim=d, out=o),
                 impls, (q, k, v), (0, 1, 2), "dense")


def bench_mla(rows, seq_lens, dtype):
    for seq in seq_lens:
        b, h, dqk, dv, rank = 4, 8, 64, 64, 128
        ks = jax.random.split(jax.random.key(seq + 3), 4)
        q = jax.random.normal(ks[0], (b, seq, h, dqk), dtype)
        cache = jax.random.normal(ks[1], (b, seq, rank), dtype)
        wk = (jax.random.normal(ks[2], (h, dqk, rank)) / math.sqrt(rank)).astype(dtype)
        wv = (jax.random.normal(ks[3], (h, dv, rank)) / math.sqrt(rank)).astype(dtype)
        scale = 1 / math.sqrt(dqk)

        def expanded(implementation):
            def fn(q, cache, wk, wv):
                kk = jnp.einsum("bsr,hdr->bshd", cache, wk)
                vv = jnp.einsum("bsr,hdr->bshd", cache, wv)
                return jax.nn.dot_product_attention(
                    q, kk, vv, scale=scale, implementation=implementation
                )

            return fn

        impls = {
            "expand_xla": expanded("xla"),
            "pallas_mla": lambda q, cache, wk, wv: mla(q, cache, wk, wv, sm_scale=scale),
        }
        if dtype != jnp.float32:
            impls["expand_cudnn"] = expanded("cudnn")
        _compare(rows, "mla", dict(batch=b, seq=seq, heads=h, rank=rank, dtype=str(jnp.dtype(dtype))),
                 impls, (q, cache, wk, wv), (0, 1, 2, 3), "expand_xla")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+",
                        default=["ssd", "mamba", "rla", "kernel_mv", "mla"])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--kmv-sizes", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    args = parser.parse_args()
    rows: list[dict] = []
    config = vars(args) | {"output": str(args.output)}
    for family in args.families:
        if family == "ssd":
            bench_ssd(rows, args.seq_lens)
        elif family == "mamba":
            bench_mamba(rows, args.seq_lens)
        elif family == "rla":
            bench_rla(rows, args.seq_lens)
        elif family == "kernel_mv":
            bench_kernel_mv(rows, args.kmv_sizes)
        elif family == "mla":
            bench_mla(rows, args.seq_lens, jnp.bfloat16)
        write_results(args.output, config, rows)


if __name__ == "__main__":
    main()
