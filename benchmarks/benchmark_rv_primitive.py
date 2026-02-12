import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import jax
import jax.numpy as jnp

from probjax.core.custom_primitives.random_variable import rv_p
from probjax.stats import norm


EAGER_ITERS = 3000
JAXPR_ITERS = 250
JIT_ITERS = 1200
JIT_VMAP_ITERS = 250
VMAP_BATCH = 256
REPEATS = 3


def _per_iter_us(total_seconds: float, iters: int) -> float:
    return (total_seconds / max(iters, 1)) * 1e6


def _run_eager() -> float:
    keys = jax.random.split(jax.random.PRNGKey(0), EAGER_ITERS)
    loc = jnp.asarray(0.0)
    scale = jnp.asarray(1.0)

    start = time.perf_counter()
    for key in keys:
        out = rv_p.bind(key, loc, scale, dist=norm, name="x")
        _ = jax.block_until_ready(out)
    elapsed = time.perf_counter() - start
    return _per_iter_us(elapsed, EAGER_ITERS)


def _run_make_jaxpr() -> float:
    loc = jnp.asarray(0.0)
    scale = jnp.asarray(1.0)
    key = jax.random.PRNGKey(1)

    def fn(rng, loc_, scale_):
        return rv_p.bind(rng, loc_, scale_, dist=norm, name="x")

    make = jax.make_jaxpr(fn)
    _ = make(key, loc, scale)

    start = time.perf_counter()
    for _ in range(JAXPR_ITERS):
        _ = make(key, loc, scale)
    elapsed = time.perf_counter() - start
    return _per_iter_us(elapsed, JAXPR_ITERS)


def _run_jit():
    loc = jnp.asarray(0.0)
    scale = jnp.asarray(1.0)
    keys = jax.random.split(jax.random.PRNGKey(2), JIT_ITERS + 1)

    def fn(rng, loc_, scale_):
        return rv_p.bind(rng, loc_, scale_, dist=norm, name="x")

    jitted = jax.jit(fn)

    start = time.perf_counter()
    out = jitted(keys[0], loc, scale)
    _ = jax.block_until_ready(out)
    first_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    for key in keys[1:]:
        out = jitted(key, loc, scale)
        _ = jax.block_until_ready(out)
    steady_us = _per_iter_us(time.perf_counter() - start, JIT_ITERS)

    return first_ms, steady_us


def _run_jit_vmap():
    locs = jnp.linspace(-1.0, 1.0, VMAP_BATCH)
    scale = jnp.asarray(1.0)
    key_batches = jax.random.split(jax.random.PRNGKey(3), JIT_VMAP_ITERS + 1)
    keys = jax.vmap(lambda k: jax.random.split(k, VMAP_BATCH))(key_batches)

    def single(rng, loc_):
        return rv_p.bind(rng, loc_, scale, dist=norm, name="x")

    vmapped = jax.vmap(single)
    jitted = jax.jit(vmapped)

    start = time.perf_counter()
    out = jitted(keys[0], locs)
    _ = jax.block_until_ready(out)
    first_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    for key_block in keys[1:]:
        out = jitted(key_block, locs)
        _ = jax.block_until_ready(out)
    steady_us = _per_iter_us(time.perf_counter() - start, JIT_VMAP_ITERS)

    return first_ms, steady_us


def benchmark_once():
    jit_first_ms, jit_steady_us = _run_jit()
    jit_vmap_first_ms, jit_vmap_steady_us = _run_jit_vmap()
    return {
        "eager_us": _run_eager(),
        "make_jaxpr_us": _run_make_jaxpr(),
        "jit_first_ms": jit_first_ms,
        "jit_steady_us": jit_steady_us,
        "jit_vmap_first_ms": jit_vmap_first_ms,
        "jit_vmap_steady_us": jit_vmap_steady_us,
    }


def summarize(result):
    print("rv_p:")
    print(f"  eager bind        : {result['eager_us']:.2f} us")
    print(f"  make_jaxpr        : {result['make_jaxpr_us']:.2f} us")
    print(f"  jit first         : {result['jit_first_ms']:.2f} ms")
    print(f"  jit steady        : {result['jit_steady_us']:.2f} us")
    print(f"  jit(vmap) first   : {result['jit_vmap_first_ms']:.2f} ms")
    print(f"  jit(vmap) steady  : {result['jit_vmap_steady_us']:.2f} us")


def _run_worker():
    print(json.dumps(benchmark_once()))


def _run_worker_subprocess():
    cmd = [
        sys.executable,
        __file__,
        "--worker",
    ]
    env = dict(os.environ)
    out = subprocess.check_output(cmd, text=True, env=env)
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("No benchmark output")
    return json.loads(lines[-1])


def _aggregate_results(runs):
    keys = runs[0].keys()
    return {key: statistics.median(run[key] for run in runs) for key in keys}


def main(repeats: int):
    print("Benchmarking random variable primitive...")
    print(f"Using median over {repeats} fresh worker runs")
    runs = [_run_worker_subprocess() for _ in range(repeats)]
    summary = _aggregate_results(runs)
    print()
    summarize(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()

    if args.worker:
        _run_worker()
    else:
        main(args.repeats)
