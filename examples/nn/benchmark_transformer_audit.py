"""Synchronized forward/backward timing for DiffusionTransformer token scaling."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from probjax.nn import DiffusionTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dtype in (jnp.float32, jnp.bfloat16):
        for tokens in (32, 128, 512):
            model = DiffusionTransformer(
                8,
                model_dim=64,
                num_heads=4,
                num_layers=2,
                attn_size=16,
                time_embed_dim=64,
                fourier_dim=32,
                context_dim=4,
                dtype=dtype,
                rngs=nnx.Rngs(0),
            )
            x = jax.random.normal(jax.random.key(1), (8, tokens, 8)).astype(dtype)
            context = jnp.ones((8, 4), dtype=dtype)
            for operation in ('forward', 'backward'):
                if operation == 'forward':
                    fn = nnx.jit(lambda m, x, c: m(0.5, x, context=c))
                else:
                    fn = nnx.jit(
                        nnx.value_and_grad(
                            lambda m, x, c: jnp.mean(
                                m(0.5, x, context=c).astype(jnp.float32) ** 2
                            )
                        )
                    )
                start = time.perf_counter()
                result = jax.block_until_ready(fn(model, x, context))
                cold = time.perf_counter() - start
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    jax.block_until_ready(fn(model, x, context))
                    times.append(time.perf_counter() - start)
                rows.append(
                    dict(
                        dtype=str(jnp.dtype(dtype)),
                        tokens=tokens,
                        operation=operation,
                        cold_s=cold,
                        warm_ms=1000 * float(np.median(times)),
                        finite=all(
                            bool(jnp.isfinite(v).all()) for v in jax.tree.leaves(result)
                        ),
                    )
                )
                print(rows[-1], flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],
                batch=8,
                repeats=5,
                results=rows,
            ),
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
