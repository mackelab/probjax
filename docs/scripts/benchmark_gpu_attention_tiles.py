"""Tile/pipeline sweep for the Pallas ``mha`` kernel on the current GPU.

Run from the repository root (split into chunks so each run stays short)::

    python -m docs.scripts.benchmark_gpu_attention_tiles \
        --output docs/benchmarks/kernel_attn_gpu_tiles.json --append \
        --dtypes bfloat16 --head-dims 64 --seq-lens 2048 --modes fwd

``auto`` rows use the kernel's own choice (on GPUs with <150 KB shared memory
per block this goes through the adaptive shrink in ``_adjust_block_sizes_for_smem``).
All other rows force the requested ``BlockSizes``/``num_stages``/``num_warps``
(the shrink is disabled by setting ``_SMEM_LIMIT_CACHE=None``); configurations
that do not fit in shared memory fail at compile time and are recorded with an
``error``.

Forward rows sweep ``block_q x block_k x num_stages x num_warps``. Backward rows
time ``jax.grad`` (forward + fused backward) with the forward tiles fixed at
``--bwd-fwd-config``; the fused backward runs one program per KV tile, so we
sweep ``outer`` (= ``block_kv_dkv`` = ``block_q_dq``, the tiles kept resident)
and ``inner`` (= ``block_q_dkv`` = ``block_kv_dq``, the tiles streamed through
the inner loops), ``num_stages`` and ``num_warps``.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import itertools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from docs.scripts._gpu_bench import run_safely, time_jax, write_results
from probjax.nn.pallas_kernels import BlockSizes, mha

# ``kernels.attention`` is shadowed by a re-exported function; import the module.
attn_mod = importlib.import_module("probjax.nn.pallas_kernels.kernels.attention")

_AUTO_LIMIT = attn_mod._query_triton_smem_limit()


def _set_forced(forced: bool):
    attn_mod._SMEM_LIMIT_CACHE = None if forced else _AUTO_LIMIT


def rel_err(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def _fns(block_sizes, num_stages, num_warps, scale):
    kw = dict(sm_scale=scale, num_stages=num_stages, num_warps=num_warps)
    if block_sizes is not None:
        kw["block_sizes"] = block_sizes

    def fwd(q, k, v):
        return mha(q, k, v, **kw)

    def loss(q, k, v, g):
        return jnp.sum(fwd(q, k, v).astype(jnp.float32) * g)

    return jax.jit(fwd), jax.jit(jax.grad(loss, argnums=(0, 1, 2)))


# Batch elements and heads are independent: errors are measured on this slice
# so the fp32 reference (which materializes [T, T] scores) stays small.
_REF = (slice(0, 1), slice(None), slice(0, 2))


def _reference(q, k, v, g, scale):
    f32 = [x[_REF].astype(jnp.float32) for x in (q, k, v)]

    def attn(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, scale=scale)

    def loss(q, k, v, g):
        return jnp.sum(attn(q, k, v) * g)

    with jax.default_matmul_precision("highest"):
        out = jax.jit(attn)(*f32)
        grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(*f32, g[_REF])
    return np.asarray(out), [np.asarray(x) for x in grads]


def _fwd_case(fwd, *, args, ref, repeats):
    timing = time_jax(fwd, *args, repeats=repeats)
    return dict(rel_err=rel_err(fwd(*args)[_REF], ref), **timing)


def _bwd_case(bwd, *, args, refs, repeats):
    timing = time_jax(bwd, *args, repeats=repeats)
    res = bwd(*args)
    err = max(rel_err(a[_REF], b) for a, b in zip(res, refs, strict=True))
    return dict(rel_err=err, **timing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true", help="append to --output")
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--modes", nargs="+", default=["fwd", "fwd_bwd"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--auto-only", action="store_true", help="only the auto rows")
    parser.add_argument("--tag", default=None, help="label stored in every row")
    parser.add_argument("--fwd-block-q", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--fwd-block-k", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--fwd-stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--fwd-warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--bwd-outer", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--bwd-inner", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--bwd-stages", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--bwd-warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--bwd-fwd-config",
        type=int,
        nargs=4,
        default=None,
        metavar=("BQ", "BK", "STAGES", "WARPS"),
        help="forward config used inside the backward sweep (default: auto's)",
    )
    args = parser.parse_args()
    rows = []
    if args.append and args.output.exists():
        rows = json.loads(args.output.read_text())["results"]
    config = vars(args) | {"output": str(args.output), "smem_limit_bytes": _AUTO_LIMIT}

    def record(row):
        rows.append(row)
        print(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
            flush=True,
        )

    for dtype_name, dim, seq in itertools.product(
        args.dtypes, args.head_dims, args.seq_lens
    ):
        dtype = jnp.dtype(dtype_name)
        keys = jax.random.split(jax.random.key(seq + dim), 4)
        shape = (args.batch, seq, args.heads, dim)
        q, k, v = (jax.random.normal(kk, shape, dtype) for kk in keys[:3])
        g = jax.random.normal(keys[3], shape, jnp.float32)
        scale = 1 / math.sqrt(dim)
        base = dict(
            dtype=dtype_name,
            head_dim=dim,
            seq=seq,
            batch=args.batch,
            heads=args.heads,
            tag=args.tag,
        )
        ref_out, ref_grads = _reference(q, k, v, g, scale)

        fwd_case = functools.partial(
            _fwd_case, args=(q, k, v), ref=ref_out, repeats=args.repeats
        )
        bwd_case = functools.partial(
            _bwd_case, args=(q, k, v, g), refs=ref_grads, repeats=args.repeats
        )

        _set_forced(False)
        fwd_auto, bwd_auto = _fns(None, 2, None, scale)
        if "fwd" in args.modes:
            record(
                dict(
                    **base,
                    mode="fwd",
                    config="auto",
                    **run_safely(functools.partial(fwd_case, fwd_auto)),
                )
            )
        if "fwd_bwd" in args.modes:
            record(
                dict(
                    **base,
                    mode="fwd_bwd",
                    config="auto",
                    **run_safely(functools.partial(bwd_case, bwd_auto)),
                )
            )

        if args.auto_only:
            write_results(args.output, config, rows)
            continue
        _set_forced(True)
        if "fwd" in args.modes:
            for bq, bk, stages, warps in itertools.product(
                args.fwd_block_q, args.fwd_block_k, args.fwd_stages, args.fwd_warps
            ):
                fwd, _ = _fns(BlockSizes(block_q=bq, block_k=bk), stages, warps, scale)
                record(
                    dict(
                        **base,
                        mode="fwd",
                        config="forced",
                        block_q=bq,
                        block_k=bk,
                        num_stages=stages,
                        num_warps=warps,
                        **run_safely(functools.partial(fwd_case, fwd)),
                    )
                )

        if "fwd_bwd" in args.modes:
            if args.bwd_fwd_config is None:
                _set_forced(False)
                fbs, fst = attn_mod._adjust_block_sizes_for_smem(
                    BlockSizes(),
                    2,
                    q_seq_len=seq,
                    kv_seq_len=seq,
                    head_dim=dim,
                    dtype_bytes=dtype.itemsize,
                )
                _set_forced(True)
                fbq, fbk, fwarps = fbs.block_q, fbs.block_k, None
            else:
                fbq, fbk, fst, fwarps = args.bwd_fwd_config
            for outer, inner, stages, warps in itertools.product(
                args.bwd_outer, args.bwd_inner, args.bwd_stages, args.bwd_warps
            ):
                bs = BlockSizes(
                    block_q=fbq,
                    block_k=fbk,
                    block_q_dkv=inner,
                    block_kv_dkv=outer,
                    block_q_dq=outer,
                    block_kv_dq=inner,
                )
                # num_stages / num_warps are shared by forward and backward kernels;
                # the forward keeps its own config through the custom_vjp only when
                # they agree, so we record what was actually used.
                _, bwd = _fns(bs, stages, warps, scale)
                record(
                    dict(
                        **base,
                        mode="fwd_bwd",
                        config="forced",
                        fwd_block_q=fbq,
                        fwd_block_k=fbk,
                        bwd_outer=outer,
                        bwd_inner=inner,
                        num_stages=stages,
                        num_warps=warps,
                        **run_safely(functools.partial(bwd_case, bwd)),
                    )
                )
        _set_forced(False)
        write_results(args.output, config, rows)


if __name__ == "__main__":
    main()
