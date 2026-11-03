"""GPU benchmark of probjax neural-network modules (forward and train step).

Run from the repository root (requires a CUDA GPU)::

    python -m docs.scripts.benchmark_gpu_nn --output docs/benchmarks/gpu_nn.json

Suites:

* ``transformer`` / ``diffusion`` -- ``Transformer`` / ``DiffusionTransformer``
  with each available ``attention_fn`` (probjax default = Flax/XLA einsum,
  ``flex_attention`` = Pallas kernel, ``jax.nn.dot_product_attention`` with the
  XLA or cuDNN implementation).
* ``moe``  -- dropless ``MoELayer`` with the Pallas grouped matmul vs the
  blocked and ``lax.ragged_dot`` fallbacks.
* ``ssm``  -- ``SSMModel`` with ``MambaCell`` / ``SSDCell`` / ``LRUCell``.
* ``unet`` -- ``UNet`` on image batches.

A train step is ``value_and_grad`` of an MSE loss followed by an AdamW update,
all under one ``jit``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from docs.scripts._gpu_bench import run_safely, time_jax, write_results
from probjax.nn import DiffusionTransformer, Transformer
from probjax.nn.layers.attention import flex_attention
from probjax.nn.layers.moe import MoELayer
from probjax.nn.layers.ssm import LRUCell, MambaCell, SSDCell
from probjax.nn.nets.ssm import SSMModel
from probjax.nn.nets.unets import UNet


def _jax_attention(implementation):
    def attention_fn(query, key, value, mask=None, bias=None, **kwargs):
        if mask is not None or bias is not None:
            raise ValueError("benchmark wrapper supports unmasked attention only")
        return jax.nn.dot_product_attention(
            query, key, value, implementation=implementation
        )

    return attention_fn


ATTENTION_FNS = {
    "probjax_default": None,
    "probjax_flex": flex_attention,
    "jax_xla": _jax_attention("xla"),
    "jax_cudnn": _jax_attention("cudnn"),
}


def _bench_module(rows, meta, model, inputs, predict):
    """Time forward and AdamW train step for ``predict(model, *inputs)``."""
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    tx = optax.adamw(1e-4)
    opt_state = tx.init(params)

    def loss_fn(params, rest, *inputs):
        out = predict(nnx.merge(graphdef, params, rest), *inputs)
        return jnp.mean(jnp.square(out.astype(jnp.float32)))

    @jax.jit
    def forward(params, rest, *inputs):
        return predict(nnx.merge(graphdef, params, rest), *inputs)

    @jax.jit
    def train_step(params, opt_state, rest, *inputs):
        loss, grads = jax.value_and_grad(loss_fn)(params, rest, *inputs)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    for op, fn, args in (
        ("forward", forward, (params, rest, *inputs)),
        ("train_step", train_step, (params, opt_state, rest, *inputs)),
    ):

        def case(fn=fn, args=args):
            timing = time_jax(fn, *args)  # first: so compile_s includes compilation
            out = fn(*args)
            finite = all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(out))
            return dict(finite=finite, **timing)

        row = dict(**meta, op=op, params=n_params, **run_safely(case))
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


def suite_transformer(rows, args, diffusion=False):
    for dtype_name in args.dtypes:
        dtype = jnp.dtype(dtype_name)
        for tokens in args.tokens:
            for attn_name, attn_fn in ATTENTION_FNS.items():
                if attn_name == "jax_cudnn" and dtype == jnp.float32:
                    continue
                kw = dict(
                    model_dim=args.model_dim, num_heads=8, num_layers=4, attn_size=args.model_dim // 8,
                    attention_fn=attn_fn, dtype=dtype, rngs=nnx.Rngs(0),
                )
                meta = dict(suite="diffusion" if diffusion else "transformer", attention=attn_name,
                            dtype=dtype_name, tokens=tokens, batch=args.batch, model_dim=args.model_dim)

                def build_and_run(kw=kw, meta=meta, tokens=tokens, dtype=dtype):
                    if diffusion:
                        model = DiffusionTransformer(8, **kw)
                        x = jax.random.normal(jax.random.key(1), (args.batch, tokens, 8), dtype)
                        t = jnp.full((args.batch,), 0.5, dtype)
                        _bench_module(rows, meta, model, (t, x), lambda m, t, x: m(t, x))
                    else:
                        model = Transformer(**kw)
                        x = jax.random.normal(jax.random.key(1), (args.batch, tokens, args.model_dim), dtype)
                        _bench_module(rows, meta, model, (x,), lambda m, x: m(x))
                    return {}

                err = run_safely(build_and_run)
                if err:
                    rows.append(dict(**meta, **err))
                    print(rows[-1], flush=True)
                jax.clear_caches()


def suite_moe(rows, args):
    for dtype_name in args.dtypes:
        dtype = jnp.dtype(dtype_name)
        for n_tokens in args.moe_tokens:
            for impl in ("pallas", "blocked", "ragged_dot"):
                meta = dict(suite="moe", impl=impl, dtype=dtype_name, tokens=n_tokens,
                            experts=8, top_k=2, d_model=512, d_hidden=1024)

                def build_and_run(impl=impl, dtype=dtype, n_tokens=n_tokens, meta=meta):
                    model = MoELayer(512, 1024, 8, top_k=2, routing="dropless", dropless_impl=impl,
                                     dtype=dtype, rngs=nnx.Rngs(0))
                    x = jax.random.normal(jax.random.key(1), (n_tokens, 512), dtype)
                    _bench_module(rows, meta, model, (x,), lambda m, x: m(x))
                    return {}

                err = run_safely(build_and_run)
                if err:
                    rows.append(dict(**meta, **err))
                    print(rows[-1], flush=True)


def suite_ssm(rows, args):
    # ``ssd`` uses SSDCell defaults (num_heads=1), which the Triton SSD kernel
    # rejects (it needs >=2 heads); with num_heads>1 SSDCell's output width is
    # model_dim // num_heads, so no SSDCell configuration runs on GPU today.
    cells = {
        "mamba": (MambaCell, None),
        "ssd": (SSDCell, None),
        "lru": (LRUCell, None),
    }
    for tokens in args.ssm_tokens:
        for name, (cell, cell_kwargs) in cells.items():
            meta = dict(suite="ssm", cell=name, dtype="float32", tokens=tokens, batch=args.batch, model_dim=128)

            def build_and_run(cell=cell, cell_kwargs=cell_kwargs, tokens=tokens, meta=meta):
                model = SSMModel(16, 128, 16, 4, recurrent_cls=cell, recurrent_kwargs=cell_kwargs,
                                 rngs=nnx.Rngs(0))
                x = jax.random.normal(jax.random.key(1), (args.batch, tokens, 16))
                _bench_module(rows, meta, model, (x,), lambda m, x: m(x))
                return {}

            err = run_safely(build_and_run)
            if err:
                rows.append(dict(**meta, **err))
                print(rows[-1], flush=True)


def suite_unet(rows, args):
    for dtype_name in args.dtypes:
        dtype = jnp.dtype(dtype_name)
        for res in (32, 64):
            meta = dict(suite="unet", dtype=dtype_name, resolution=res, batch=args.batch)

            def build_and_run(res=res, dtype=dtype, meta=meta):
                model = UNet(3, (64, 128, 256), dtype=dtype, rngs=nnx.Rngs(0))
                x = jax.random.normal(jax.random.key(1), (args.batch, res, res, 3), dtype)
                _bench_module(rows, meta, model, (x,), lambda m, x: m(x))
                return {}

            err = run_safely(build_and_run)
            if err:
                rows.append(dict(**meta, **err))
                print(rows[-1], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", default=["transformer", "diffusion", "moe", "ssm", "unet"])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--tokens", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--moe-tokens", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--ssm-tokens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=256)
    args = parser.parse_args()
    rows: list[dict] = []
    config = vars(args) | {"output": str(args.output)}
    for suite in args.suites:
        if suite in ("transformer", "diffusion"):
            suite_transformer(rows, args, diffusion=suite == "diffusion")
        elif suite == "moe":
            suite_moe(rows, args)
        elif suite == "ssm":
            suite_ssm(rows, args)
        elif suite == "unet":
            suite_unet(rows, args)
        write_results(args.output, config, rows)


if __name__ == "__main__":
    main()
