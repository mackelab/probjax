"""Data-parallel weak scaling of ``MoELayer``: per-shard vs global routing.

Builds a residual stack of ``--layers`` MoE layers, times one fwd+bwd step on
one GPU at ``--tokens`` tokens, then on all visible GPUs (a ``("data",
"model")`` mesh with ``model=1``) at ``--tokens`` per GPU, with
``local_routing`` off and on.  Efficiency is ``t_1gpu / t_ngpu`` (1.0 =
perfect weak scaling).  Needs at least two GPUs that are otherwise idle.

    python docs/scripts/benchmark_gpu_moe_data_parallel.py --routing dropless
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from probjax.nn import MoELayer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--routing", default="dropless", choices=("capacity", "dense", "dropless")
    )
    parser.add_argument("--tokens", type=int, default=32768, help="per GPU")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-hidden", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    n_dev = jax.device_count()
    if n_dev < 2:
        raise SystemExit("needs at least two devices")
    dtype = jnp.bfloat16 if args.bf16 else None
    seq = 1024
    batch = max(1, args.tokens // seq)

    class Stack(nnx.Module):
        def __init__(self, local: bool, rngs: nnx.Rngs):
            self.layers = nnx.List([
                MoELayer(
                    args.d_model,
                    args.d_hidden,
                    args.experts,
                    top_k=args.top_k,
                    routing=args.routing,
                    local_routing=local,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(args.layers)
            ])

        def __call__(self, x):
            for layer in self.layers:
                x = x + layer(x).astype(x.dtype)
            return x

    def step(local: bool):
        graphdef, state = nnx.split(Stack(local, nnx.Rngs(0)))

        def loss(state, x):
            return jnp.mean(nnx.merge(graphdef, state)(x) ** 2)

        return jax.jit(jax.value_and_grad(loss)), state

    def bench(f, *a) -> float:
        for _ in range(3):
            jax.block_until_ready(f(*a))
        start = time.perf_counter()
        for _ in range(args.iters):
            out = f(*a)
        jax.block_until_ready(out)
        return (time.perf_counter() - start) / args.iters * 1e3

    key = jax.random.key(0)
    f1, s1 = step(True)
    t1 = bench(f1, s1, jax.random.normal(key, (batch, seq, args.d_model)))
    print(
        f"{args.routing} layers={args.layers} tokens/GPU={batch * seq} "
        f"{'bf16' if dtype else 'fp32'}: 1 GPU {t1:.2f} ms"
    )
    mesh = jax.make_mesh((n_dev, 1), ("data", "model"), axis_types=(AxisType.Auto,) * 2)
    x = jax.random.normal(key, (n_dev * batch, seq, args.d_model))
    for local in (False, True):
        with jax.set_mesh(mesh):
            f, s = step(local)
            t = bench(f, s, jax.device_put(x, NamedSharding(mesh, P("data"))))
        label = "per-shard" if local else "global   "
        print(f"  {label} routing, {n_dev} GPUs: {t:.2f} ms, efficiency {t1 / t:.2f}")
        jax.clear_caches()


if __name__ == "__main__":
    main()
