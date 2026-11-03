"""Tile/pipeline sweep for the Pallas ``mha`` kernel on the current GPU.

Run from the repository root::

    python -m docs.scripts.benchmark_gpu_attention_tiles \
        --output docs/benchmarks/gpu_attention_tiles.json

``auto`` rows use the kernel's own choice (on GPUs with <150 KB shared memory
per block this goes through the adaptive shrink in ``_adjust_block_sizes_for_smem``).
All other rows force the requested ``BlockSizes``/``num_stages``/``num_warps``;
configurations that do not fit in shared memory fail at compile time and are
recorded with an ``error``.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from docs.scripts._gpu_bench import max_abs_err, run_safely, time_jax, write_results
from probjax.nn.pallas_kernels import BlockSizes, mha

# ``kernels.attention`` is shadowed by a re-exported function; import the module.
attn_mod = importlib.import_module("probjax.nn.pallas_kernels.kernels.attention")

_AUTO_LIMIT = attn_mod._query_triton_smem_limit()


def _set_forced(forced: bool):
    attn_mod._SMEM_LIMIT_CACHE = None if forced else _AUTO_LIMIT


def _fns(block_sizes, num_stages, num_warps, scale):
    kw = dict(sm_scale=scale, num_stages=num_stages, num_warps=num_warps)
    if block_sizes is not None:
        kw["block_sizes"] = block_sizes

    def fwd(q, k, v):
        return mha(q, k, v, **kw)

    def loss(q, k, v, g):
        return jnp.sum(fwd(q, k, v).astype(jnp.float32) * g)

    return jax.jit(fwd), jax.jit(jax.grad(loss, argnums=(0, 1, 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    args = parser.parse_args()
    rows = []
    config = vars(args) | {"output": str(args.output), "smem_limit_bytes": _AUTO_LIMIT}

    for dtype_name, dim, seq in itertools.product(args.dtypes, args.head_dims, args.seq_lens):
        dtype = jnp.dtype(dtype_name)
        keys = jax.random.split(jax.random.key(seq + dim), 4)
        shape = (args.batch, seq, args.heads, dim)
        q, k, v = (jax.random.normal(kk, shape, dtype) for kk in keys[:3])
        g = jax.random.normal(keys[3], shape, jnp.float32)
        scale = 1 / math.sqrt(dim)
        base = dict(dtype=dtype_name, head_dim=dim, seq=seq, batch=args.batch, heads=args.heads)

        # Reference output from the kernel's own (auto) configuration.
        _set_forced(False)
        fwd_auto, bwd_auto = _fns(None, 2, None, scale)
        ref_out = fwd_auto(q, k, v)
        ref_grads = bwd_auto(q, k, v, g)

        def record(row):
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

        record(dict(**base, mode="fwd", config="auto", **run_safely(lambda: time_jax(fwd_auto, q, k, v))))
        record(dict(**base, mode="fwd_bwd", config="auto", **run_safely(lambda: time_jax(bwd_auto, q, k, v, g))))

        # Forward sweep.
        _set_forced(True)
        for bq, bk, stages, warps in itertools.product((64, 128), (32, 64, 128), (1, 2, 3), (4, 8)):
            bs = BlockSizes(block_q=bq, block_k=bk)
            fwd, _ = _fns(bs, stages, warps, scale)

            def case(fwd=fwd):
                timing = time_jax(fwd, q, k, v, repeats=5)
                return dict(max_abs_err_vs_auto=max_abs_err(fwd(q, k, v), ref_out), **timing)

            record(dict(**base, mode="fwd", config="forced", block_q=bq, block_k=bk,
                        num_stages=stages, num_warps=warps, **run_safely(case)))

        # Backward sweep (forward tiles fixed at the kernel defaults).
        for bdkv, bdq, stages in itertools.product((32, 64, 128), (32, 64, 128), (1, 2)):
            bs = BlockSizes(block_q=64, block_k=64, block_q_dkv=bdkv, block_kv_dkv=bdkv,
                            block_q_dq=bdq, block_kv_dq=bdq)
            _, bwd = _fns(bs, stages, None, scale)

            def case(bwd=bwd):
                timing = time_jax(bwd, q, k, v, g, repeats=5)
                res = bwd(q, k, v, g)
                err = max(max_abs_err(a, b) for a, b in zip(res, ref_grads, strict=True))
                return dict(max_abs_err_vs_auto=err, **timing)

            record(dict(**base, mode="fwd_bwd", config="forced", block_dkv=bdkv, block_dq=bdq,
                        num_stages=stages, **run_safely(case)))
        _set_forced(False)
        write_results(args.output, config, rows)


if __name__ == "__main__":
    main()
