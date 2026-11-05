"""Shared GPU timing helpers for the device benchmarks in ``docs/scripts``.

Timings are wall-clock over ``inner`` back-to-back calls bracketed by a device
synchronization, repeated ``repeats`` times; we report the median and the
inter-quartile range per call. Compilation (first call) is timed separately.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = dict(
        jax=jax.__version__,
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        python=platform.python_version(),
    )
    try:
        info["device_kind"] = jax.devices()[0].device_kind
    except Exception:  # pragma: no cover - device introspection is best effort
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["gpu"] = pynvml.nvmlDeviceGetName(handle)
        info["driver"] = pynvml.nvmlSystemGetDriverVersion()
        info["gpu_memory_gb"] = round(
            pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**30, 1
        )
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def _summarize(samples: list[float], inner: int) -> dict[str, float]:
    per_call = np.asarray(samples) / inner * 1e3
    q1, med, q3 = np.percentile(per_call, [25, 50, 75])
    return dict(median_ms=float(med), q1_ms=float(q1), q3_ms=float(q3))


def time_jax(
    fn: Callable[..., Any],
    *args: Any,
    repeats: int = 7,
    inner: int | None = None,
    target_s: float = 0.05,
) -> dict[str, float]:
    """Time an already-jitted JAX callable; returns compile + warm statistics."""
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - start
    # Second call: steady state, used to size the inner loop.
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    single = max(time.perf_counter() - start, 1e-6)
    if inner is None:
        inner = int(min(max(target_s / single, 1), 200))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = None
        for _ in range(inner):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - start)
    return dict(compile_s=compile_s, inner=inner, **_summarize(samples, inner))


def time_torch(
    fn: Callable[..., Any],
    *args: Any,
    repeats: int = 7,
    inner: int | None = None,
    target_s: float = 0.05,
) -> dict[str, float]:
    """Time a torch callable on CUDA with explicit synchronization."""
    import torch

    start = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - start
    start = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    single = max(time.perf_counter() - start, 1e-6)
    if inner is None:
        inner = int(min(max(target_s / single, 1), 200))
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(inner):
            fn(*args)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return dict(compile_s=compile_s, inner=inner, **_summarize(samples, inner))


def max_abs_err(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def run_safely(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one benchmark case; record failures (e.g. OOM) instead of aborting."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - failures are data here
        msg = str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__
        return dict(error=f"{type(exc).__name__}: {msg}")
    finally:
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass


def write_results(path: Path, config: dict[str, Any], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(device=device_info(), config=config, results=rows)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path} ({len(rows)} rows)", flush=True)
