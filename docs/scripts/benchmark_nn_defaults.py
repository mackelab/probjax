"""Small synchronized forward/gradient comparisons for transformer profiles.

Run from the repository root: python -m docs.scripts.benchmark_nn_defaults
These are capacity/runtime checks, not training-quality rankings.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from probjax.nn import DiffusionTransformer, Transformer
from probjax.nn.generative.nflows.autoregressive import AutoregressiveTransformer
from probjax.nn.layers.attention import dot_product_attention

PROFILES = {
    "small_reference": dict(model_dim=64, num_heads=4, attn_size=16),
    "compact": dict(model_dim=128, num_heads=8, attn_size=32),
    "standard": dict(model_dim=256, num_heads=8, attn_size=32),
}


def build(kind, profile, seed):
    kwargs = {**profile, "num_layers": 4, "widening_factor": 4, "rngs": nnx.Rngs(seed)}
    if kind == "diffusion":
        return DiffusionTransformer(2, **kwargs), 2
    if kind == "autoregressive":
        model = AutoregressiveTransformer(
            1, 2, None, attention_fn=dot_product_attention, **kwargs
        )
        # Exercise the trunk's gradients rather than a zero-initialized head.
        model.decoder.kernel[...] = 0.01 * jax.random.normal(
            jax.random.key(seed + 3), model.decoder.kernel.shape
        )
        return model, 1
    return Transformer(**kwargs), profile["model_dim"]


def run(args):
    rows = []
    for kind in ("transformer", "autoregressive", "diffusion"):
        for name, profile in PROFILES.items():
            for tokens in args.tokens:
                model, features = build(kind, profile, 0)
                data = jax.random.normal(
                    jax.random.key(1), (args.batch, tokens, features)
                )

                def predict(current, values, kind=kind):
                    if kind == "diffusion":
                        return current(0.5, values)
                    if kind == "autoregressive":
                        return current.predict_bij_params(values)
                    return current(values)

                operations = {
                    "forward": nnx.jit(predict),
                    "gradient": nnx.jit(
                        nnx.value_and_grad(
                            lambda current, values, predict=predict: jnp.mean(
                                (predict(current, values) - 0.2) ** 2
                            )
                        )
                    ),
                }
                parameters = sum(
                    value.size for value in jax.tree.leaves(nnx.state(model, nnx.Param))
                )
                for operation, call in operations.items():
                    start = time.perf_counter()
                    result = jax.block_until_ready(call(model, data))
                    cold = time.perf_counter() - start
                    durations = []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        jax.block_until_ready(call(model, data))
                        durations.append(time.perf_counter() - start)
                    row = {
                        "kind": kind,
                        "profile": name,
                        **profile,
                        "layers": 4,
                        "tokens": tokens,
                        "operation": operation,
                        "parameters": parameters,
                        "cold_seconds": cold,
                        "warm_ms": 1000 * float(np.median(durations)),
                        "finite": all(
                            bool(jnp.isfinite(value).all())
                            for value in jax.tree.leaves(result)
                        ),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(
                        json.dumps(
                            {
                                "jax": jax.__version__,
                                "platform": platform.platform(),
                                "devices": [str(device) for device in jax.devices()],
                                "batch": args.batch,
                                "repeats": args.repeats,
                                "rows": rows,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                jax.clear_caches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/benchmarks/nn_defaults.json")
    )
    run(parser.parse_args())
