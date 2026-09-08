"""Compile/warm-time and compiled-memory scaling for temporal inference.

Run from the repository: python benchmarks/temporal_smc.py
Memory figures are compiler estimates, not measured process RSS. Backend/model
costs and input arrays are included separately from retained output storage.
"""

import json
import time

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from probjax.inference import (
    particle_backend,
    run_temporal_filter,
    run_temporal_smc,
    temporal_smc,
)


def benchmark():
    key = jax.random.key(0)
    for n in (64, 256):
        backend = particle_backend(
            lambda k, p, t, n=n: jax.random.normal(k, (n, 1)),
            lambda k, p, x, a, b: (
                x + jnp.sqrt(jnp.exp(p) * (b - a)) * jax.random.normal(k, x.shape)
            ),
            lambda p, x, y, t: norm.logpdf(y, x, 0.5).sum(-1),
        )
        for horizon in (32, 128):
            ts = jnp.arange(1, horizon + 1, dtype=float) * 0.2
            ys = jnp.sin(ts)[:, None]
            for history in ('none', 8, 'full'):
                state = backend.init(key, jnp.array(-1.0), 0.0)
                fn = jax.jit(
                    lambda k, s, ts, ys, history=history, backend=backend: (
                        run_temporal_filter(
                            backend, k, s, jnp.array(-1.0), ts, ys, history=history
                        )
                    )
                )
                measure(fn, key, state, ts, ys, n=n, horizon=horizon, history=history)
            # Compare normal outer updates with forced resample/replay moves.
            for moves in (False, True):
                kernel = temporal_smc(
                    backend,
                    lambda p: norm.logpdf(p),
                    ess_threshold=1.0 if moves else 0.0,
                    proposal_fn=(lambda k, p: (p + 0.1 * jax.random.normal(k), 0.0))
                    if moves
                    else None,
                )
                state = kernel.init(key, jnp.linspace(-2.0, 0.0, 16), 0.0)
                fn = jax.jit(
                    lambda k, s, ts, ys, kernel=kernel: run_temporal_smc(
                        kernel, k, s, ts, ys
                    )
                )
                measure(
                    fn, key, state, ts, ys, n=n, horizon=horizon, outer=16, moves=moves
                )


def measure(fn, key, state, ts, ys, **labels):
    start = time.perf_counter()
    executable = fn.lower(key, state, ts, ys).compile()
    compile_seconds = time.perf_counter() - start
    executable(key, state, ts, ys).log_likelihood.block_until_ready()
    start = time.perf_counter()
    for _ in range(3):
        executable(key, state, ts, ys).log_likelihood.block_until_ready()
    ms = (time.perf_counter() - start) / 3 * 1000
    memory = executable.memory_analysis()
    print(
        json.dumps(
            dict(
                **labels,
                compile_seconds=compile_seconds,
                warm_ms=ms,
                temporary_bytes=memory.temp_size_in_bytes,
                output_bytes=memory.output_size_in_bytes,
            )
        ),
        flush=True,
    )


if __name__ == '__main__':
    benchmark()
