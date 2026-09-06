"""Measure tracing, compilation, and execution separately (CPU by default).

Run with the project Python: ``python benchmarks/benchmark_inverse_recovery.py``.
Use ``--ordinary-only`` to compare against a revision without section recovery.
Times are medians in milliseconds; execution timings synchronize device work.
"""

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp

from probjax.core import inverse, inverse_and_logabsdet


def functions(depth, composition, with_logdet):
    def forward(x, matrix, bias):
        for _ in range(depth):
            if composition:
                x = jnp.log(jnp.exp(matrix @ x + x + bias))
            else:
                x = jnp.log(jnp.exp(1.01 * x + 0.001))
        return x

    def handwritten(y, matrix, bias):
        logdet = jnp.asarray(0.0)
        for _ in range(depth):
            if with_logdet:
                logdet += jnp.sum(y)
            y = jnp.exp(y)
            if with_logdet:
                logdet -= jnp.sum(jnp.log(y))
            y = jnp.log(y)
            if composition:
                coefficient = matrix + jnp.eye(y.size)
                y = jnp.linalg.solve(coefficient, y - bias)
                if with_logdet:
                    logdet -= jnp.linalg.slogdet(coefficient)[1]
            else:
                y = (y - 0.001) / 1.01
                if with_logdet:
                    logdet -= y.size * jnp.log(1.01)
        return (y, logdet) if with_logdet else y

    return forward, handwritten


def benchmark(depth, composition, with_logdet, repeats):
    forward, handwritten = functions(depth, composition, with_logdet)
    transform = inverse_and_logabsdet if with_logdet else inverse
    args = (jnp.full(8, 0.2), 0.01 * jnp.eye(8), jnp.full(8, 0.001))
    # Warm library initialization; every measured wrapper still gets a fresh
    # forward trace and recovery-analysis cache.
    jax.make_jaxpr(transform(forward))(*args)
    traces = []
    for _ in range(repeats):
        generated = transform(forward)
        start = time.perf_counter()
        jax.make_jaxpr(generated)(*args)
        traces.append(1000 * (time.perf_counter() - start))

    timings = {}
    for name, fn in ("generated", transform(forward)), ("handwritten", handwritten):
        start = time.perf_counter()
        lowered = jax.jit(fn).lower(*args)
        timings[name + "_lower_ms"] = 1000 * (time.perf_counter() - start)
        start = time.perf_counter()
        compiled = lowered.compile()
        timings[name + "_compile_ms"] = 1000 * (time.perf_counter() - start)
        jax.block_until_ready(compiled(*args))
        samples = []
        for _ in range(max(40, repeats)):
            start = time.perf_counter()
            jax.block_until_ready(compiled(*args))
            samples.append(1000 * (time.perf_counter() - start))
        timings[name + "_execute_ms"] = statistics.median(samples)
    return {
        "depth": depth,
        "composition": composition,
        "logdet": with_logdet,
        "trace_ms": statistics.median(traces),
        **timings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--ordinary-only", action="store_true")
    options = parser.parse_args()
    for composition in [False] if options.ordinary_only else [False, True]:
        for logdet in (False, True):
            for depth in options.depths:
                print(
                    json.dumps(benchmark(depth, composition, logdet, options.repeats)),
                    flush=True,
                )
