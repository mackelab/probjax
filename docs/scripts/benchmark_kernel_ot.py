# ruff: noqa: B023
"""Benchmark the streaming log-domain kernel reduction and its OT consumers.

Run from the repository root (hold the shared bench lock for timings)::

    python -m docs.scripts.benchmark_kernel_ot --device gpu --suite lse
    python -m docs.scripts.benchmark_kernel_ot --device gpu --suite sinkhorn
    python -m docs.scripts.benchmark_kernel_ot --device gpu --suite scale
    python -m docs.scripts.benchmark_kernel_ot --device gpu --suite peak
    python -m docs.scripts.benchmark_kernel_ot --device gpu --suite cfm
    JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_ot --device cpu --suite lse

Suites (all rows go to ``docs/benchmarks/kernel_ot_{cpu,gpu}.json``; ``--append``
merges into an existing file, replacing rows with the same key):

* ``lse``      -- ``kernel_logsumexp`` (auto backend: Pallas on GPU, blocked XLA
                  on CPU) vs the dense ``jax.nn.logsumexp`` over the materialized
                  N x M cost matrix; N = M in 1k..64k, D in {2, 8, 32}; fwd and
                  fwd+bwd; time and XLA memory analysis (temp/arg/out MB).
* ``scale``    -- largest problems: kernel up to N = M = 1M (GPU), dense until OOM.
* ``sinkhorn`` -- 50 log-domain Sinkhorn iterations, ours vs a dense log-domain
                  solver on the materialized cost (ott-jax is not installed).
* ``peak``     -- true device peak (``peak_bytes_in_use``) of headline cases,
                  each measured in a fresh subprocess (GPU only).
* ``cfm``      -- OT-CFM demo: N(0, I) -> 8 Gaussians with ``LinearFlow``,
                  independent vs exact minibatch-OT pairing; sample quality
                  (Sinkhorn divergence to held-out target samples) vs Euler NFE
                  and path straightness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from docs.scripts._gpu_bench import device_info, run_safely, time_jax
from probjax.nn.ot import sinkhorn
from probjax.nn.pallas_kernels import kernel_logsumexp, kernel_logsumexp_dense
from probjax.nn.pallas_kernels.kernels.kernel_lse import sqeuclidean_cost

EPS = 0.5


def _mem(jfn, *args):
    try:
        m = jfn.lower(*args).compile().memory_analysis()
        return dict(
            xla_temp_mb=m.temp_size_in_bytes / 2**20,
            xla_arg_mb=m.argument_size_in_bytes / 2**20,
            xla_out_mb=m.output_size_in_bytes / 2**20,
        )
    except Exception as exc:  # noqa: BLE001
        return dict(mem_error=str(exc).splitlines()[0][:200])


def _inputs(n, m, d, dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k[0], (n, d), dtype)
    y = jax.random.normal(k[1], (m, d), dtype)
    lw = jax.random.normal(k[2], (m,), jnp.float32) * 0.1
    return x, y, lw


def _lse_fns(impl):
    if impl == "dense":
        f = lambda x, y, lw: kernel_logsumexp_dense(  # noqa: E731
            x, y, log_weights=lw, epsilon=EPS
        )
    else:
        f = lambda x, y, lw: kernel_logsumexp(  # noqa: E731
            x, y, log_weights=lw, epsilon=EPS, impl=impl
        )
    fwd = jax.jit(f)
    bwd = jax.jit(
        jax.grad(lambda x, y, lw: jnp.sum(jnp.sin(f(x, y, lw))), argnums=(0, 1, 2))
    )
    return fwd, bwd


def _dense_bytes(n, m):
    return 4 * n * m


def _time_case(jfn, args, repeats, budget_s):
    """Time with the shared helper; very slow cases get a single sample."""
    t0 = time.perf_counter()
    jax.block_until_ready(jfn(*args))
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax.block_until_ready(jfn(*args))
    single = time.perf_counter() - t0
    if single * (repeats + 1) > budget_s:
        return dict(
            compile_s=first - single,
            median_ms=single * 1e3,
            q1_ms=single * 1e3,
            q3_ms=single * 1e3,
            inner=1,
            repeats=1,
        )
    return time_jax(jfn, *args, repeats=repeats)


def lse_rows(device, sizes, dims, impls, repeats, max_dense_bytes, budget_s):
    rows = []
    for d in dims:
        for n in sizes:
            x, y, lw = _inputs(n, n, d)
            for impl in impls:
                fwd, bwd = _lse_fns(impl)
                for mode, jfn in (("fwd", fwd), ("fwd_bwd", bwd)):
                    key = dict(
                        suite="lse",
                        impl=impl,
                        mode=mode,
                        n=n,
                        m=n,
                        d=d,
                        dtype="float32",
                    )
                    mem = _mem(jfn, x, y, lw)
                    if impl == "dense" and _dense_bytes(n, n) > max_dense_bytes:
                        res = dict(
                            skipped=f"dense N*M*4 > {max_dense_bytes / 2**30:.0f} GiB"
                        )
                    else:
                        res = run_safely(
                            lambda: _time_case(jfn, (x, y, lw), repeats, budget_s)
                        )
                    row = {**key, **mem, **res}
                    print(json.dumps(row), flush=True)
                    rows.append(row)
    return rows


def lse_accuracy_rows(device):
    """Relative error vs a float64 host reference (fp32 inputs)."""
    from scipy.special import logsumexp, softmax

    rows = []
    for n, d, eps in [(2048, 8, 0.5), (2048, 8, 0.01), (2048, 32, 0.05)]:
        x, y, lw = _inputs(n, n, d)
        xn, yn, lwn = (np.asarray(a, np.float64) for a in (x, y, lw))
        c = (xn**2).sum(1)[:, None] + (yn**2).sum(1)[None] - 2 * xn @ yn.T
        s = lwn[None] - c / eps
        ref = logsumexp(s, axis=1)
        p = softmax(s, axis=1)
        gref = -2 / eps * (xn - p @ yn)
        for impl in ["dense", "pallas" if device == "gpu" else "xla"]:
            if impl == "dense":
                f = lambda x: kernel_logsumexp_dense(x, y, log_weights=lw, epsilon=eps)  # noqa: E731
            else:
                f = lambda x, impl=impl: kernel_logsumexp(  # noqa: E731
                    x, y, log_weights=lw, epsilon=eps, impl=impl
                )
            out = np.asarray(jax.jit(f)(x), np.float64)
            g = np.asarray(jax.jit(jax.grad(lambda x: jnp.sum(f(x))))(x), np.float64)
            row = dict(
                suite="accuracy",
                impl=impl,
                n=n,
                d=d,
                eps=eps,
                lse_max_abs_err=float(np.max(np.abs(out - ref))),
                grad_rel_err=float(np.max(np.abs(g - gref)) / np.max(np.abs(gref))),
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
    # Default-precision dense on GPU (TF32 cross term) for contrast.
    if device == "gpu":
        n, d, eps = 2048, 8, 0.01
        x, y, lw = _inputs(n, n, d)
        xn, yn, lwn = (np.asarray(a, np.float64) for a in (x, y, lw))
        c = (xn**2).sum(1)[:, None] + (yn**2).sum(1)[None] - 2 * xn @ yn.T
        ref = logsumexp(lwn[None] - c / eps, axis=1)
        out = jax.jit(
            lambda x: kernel_logsumexp_dense(
                x, y, log_weights=lw, epsilon=eps, precision=None
            )
        )(x)
        row = dict(
            suite="accuracy",
            impl="dense_default_precision",
            n=n,
            d=d,
            eps=eps,
            lse_max_abs_err=float(np.max(np.abs(np.asarray(out, np.float64) - ref))),
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def scale_rows(device, kernel_sizes, dense_sizes, d, budget_s):
    rows = []
    for n in kernel_sizes:
        x, y, lw = _inputs(n, n, d)
        fwd, bwd = _lse_fns(None)
        for mode, jfn in (("fwd", fwd), ("fwd_bwd", bwd)):
            row = dict(
                suite="scale",
                impl="kernel",
                mode=mode,
                n=n,
                m=n,
                d=d,
                **_mem(jfn, x, y, lw),
                **run_safely(lambda: _time_case(jfn, (x, y, lw), 1, budget_s)),
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
    for n in dense_sizes:
        x, y, lw = _inputs(n, n, d)
        fwd, bwd = _lse_fns("dense")
        for mode, jfn in (("fwd", fwd), ("fwd_bwd", bwd)):
            row = dict(
                suite="scale",
                impl="dense",
                mode=mode,
                n=n,
                m=n,
                d=d,
                **_mem(jfn, x, y, lw),
                **run_safely(lambda: _time_case(jfn, (x, y, lw), 1, budget_s)),
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
    return rows


def _dense_sinkhorn(x, y, eps, iters):
    n, m = x.shape[0], y.shape[0]
    c = sqeuclidean_cost(x, y, lax.Precision.HIGHEST)
    la = jnp.full((n,), -jnp.log(n))
    lb = jnp.full((m,), -jnp.log(m))

    def body(f, _):
        g = -eps * jax.nn.logsumexp(la[:, None] + (f[:, None] - c) / eps, axis=0)
        f = -eps * jax.nn.logsumexp(lb[None, :] + (g[None, :] - c) / eps, axis=1)
        return f, g

    f, gs = lax.scan(body, jnp.zeros((n,)), None, length=iters)
    g = gs[-1]
    return jnp.exp(la) @ f + jnp.exp(lb) @ g


def sinkhorn_rows(device, sizes, d, iters, repeats, max_dense_bytes, budget_s):
    rows = []
    for n in sizes:
        k = jax.random.split(jax.random.key(1), 2)
        x = jax.random.normal(k[0], (n, d))
        y = jax.random.normal(k[1], (n, d)) * 0.5 + 1.0
        eps = 0.05 * float(
            jnp.mean(jnp.sum(x**2, 1))
            + jnp.mean(jnp.sum(y**2, 1))
            - 2 * jnp.mean(x, 0) @ jnp.mean(y, 0)
        )
        impls = {
            "kernel": lambda x, y: (
                sinkhorn(x, y, epsilon=eps, num_iters=iters).reg_ot_cost
            ),
            "dense": lambda x, y: _dense_sinkhorn(x, y, eps, iters),
        }
        grads = {
            "kernel": jax.grad(
                lambda x, y: sinkhorn(x, y, epsilon=eps, num_iters=iters).reg_ot_cost
            ),
        }
        vals = {}
        for name, f in impls.items():
            jfn = jax.jit(f)
            if name == "dense" and _dense_bytes(n, n) > max_dense_bytes:
                res = dict(skipped="dense cost matrix too large")
            else:
                res = run_safely(lambda: _time_case(jfn, (x, y), repeats, budget_s))
                if "error" not in res:
                    vals[name] = float(jfn(x, y))
            row = dict(
                suite="sinkhorn",
                impl=name,
                mode="fwd",
                n=n,
                m=n,
                d=d,
                iters=iters,
                eps=eps,
                **_mem(jfn, x, y),
                **res,
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
        for name, f in grads.items():
            jfn = jax.jit(f)
            res = run_safely(lambda: _time_case(jfn, (x, y), repeats, budget_s))
            row = dict(
                suite="sinkhorn",
                impl=name,
                mode="fwd_bwd_implicit",
                n=n,
                m=n,
                d=d,
                iters=iters,
                eps=eps,
                **_mem(jfn, x, y),
                **res,
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
        if len(vals) == 2:
            rows.append(
                dict(
                    suite="sinkhorn",
                    impl="agreement",
                    n=n,
                    d=d,
                    cost_kernel=vals["kernel"],
                    cost_dense=vals["dense"],
                    abs_diff=abs(vals["kernel"] - vals["dense"]),
                )
            )
            print(json.dumps(rows[-1]), flush=True)
    return rows


PEAK_CASES = [
    ("kernel", "fwd", 65536, 8),
    ("kernel", "fwd_bwd", 65536, 8),
    ("dense", "fwd", 32768, 8),
    ("dense", "fwd_bwd", 32768, 8),
    ("kernel", "fwd_bwd", 1048576, 2),
]


def _peak_child(impl, mode, n, d):
    x, y, lw = _inputs(n, n, d)
    fwd, bwd = _lse_fns(None if impl == "kernel" else "dense")
    jfn = fwd if mode == "fwd" else bwd
    base = jax.devices()[0].memory_stats()["peak_bytes_in_use"]
    jax.block_until_ready(jfn(x, y, lw))
    peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"]
    print(json.dumps(dict(peak_mb=peak / 2**20, inputs_peak_mb=base / 2**20)))


def peak_rows():
    rows = []
    for impl, mode, n, d in PEAK_CASES:
        cmd = [
            sys.executable,
            "-m",
            "docs.scripts.benchmark_kernel_ot",
            "--peak-child",
            impl,
            mode,
            str(n),
            str(d),
        ]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=dict(os.environ)
            )
            line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
            res = (
                json.loads(line[-1])
                if line
                else dict(error=out.stderr.strip().splitlines()[-1][:300])
            )
        except Exception as exc:  # noqa: BLE001
            res = dict(error=str(exc)[:300])
        row = dict(suite="peak", impl=impl, mode=mode, n=n, m=n, d=d, **res)
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# OT-CFM demo
# ---------------------------------------------------------------------------


def _eight_gaussians(key, n, radius=4.0, std=0.1):
    k1, k2 = jax.random.split(key)
    ang = jax.random.randint(k1, (n,), 0, 8) * (jnp.pi / 4)
    centers = radius * jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)
    return centers + std * jax.random.normal(k2, (n, 2))


def cfm_rows(steps, batch, nfes, seeds, eval_n):
    import optax
    from flax import nnx

    from probjax.nn import TimeMLP
    from probjax.nn.generative.flow_matching import (
        LinearFlow,
        UniformFlowTrainingConfig,
    )
    from probjax.nn.ot import sinkhorn_divergence

    rows = []
    target = _eight_gaussians(jax.random.key(123), eval_n)
    noise = jax.random.normal(jax.random.key(321), (eval_n, 2))
    sdiv = jax.jit(lambda a, b: sinkhorn_divergence(a, b, epsilon=0.01, num_iters=300))
    floor = float(sdiv(_eight_gaussians(jax.random.key(7), eval_n), target))
    for seed in seeds:
        for coupling in [None, "exact", "sinkhorn"]:
            net = TimeMLP(2, hidden_dim=128, depth=3, rngs=nnx.Rngs(seed))
            model = LinearFlow(
                net,
                event_spec=2,
                train_cfg=UniformFlowTrainingConfig(),
                ot_coupling=coupling,
            )
            opt = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

            @nnx.jit
            def train_step(model, opt, key):
                kd, kl = jax.random.split(key)
                data = _eight_gaussians(kd, batch)
                loss, grads = nnx.value_and_grad(lambda m: m.loss(kl, data))(model)
                opt.update(model, grads)
                return loss

            key = jax.random.key(seed)
            jax.block_until_ready(train_step(model, opt, key))
            t0 = time.perf_counter()
            for i in range(steps):
                loss = train_step(model, opt, jax.random.fold_in(key, i))
            jax.block_until_ready(loss)
            train_s = time.perf_counter() - t0
            row = dict(
                suite="cfm",
                coupling=coupling or "independent",
                seed=seed,
                steps=steps,
                batch=batch,
                train_s=train_s,
                ms_per_step=train_s / steps * 1e3,
                final_loss=float(loss),
                sdiv_floor=floor,
            )
            sample = nnx.jit(
                lambda model, z, n: model.sample_from_noise(
                    z, num_steps=n, method="euler"
                ),
                static_argnums=(2,),
            )
            for nfe in nfes:
                xs = sample(
                    model, noise, nfe + 1
                )  # nfe Euler steps = nfe + 1 grid points
                row[f"sdiv_nfe{nfe}"] = float(sdiv(xs, target))
            path = nnx.jit(
                lambda model, z: model.sample_from_noise(
                    z, num_steps=129, method="euler", trace=True
                )
            )(model, noise)
            path = jnp.asarray(path)
            if path.shape[0] != noise.shape[0]:
                vel = (path[1:] - path[:-1]) * (path.shape[0] - 1)
                disp = path[-1] - path[0]
                row["straightness"] = float(
                    jnp.mean(jnp.sum((vel - disp[None]) ** 2, -1))
                )
            print(json.dumps(row), flush=True)
            rows.append(row)
    return rows


def _row_key(r):
    return tuple(
        r.get(k)
        for k in ("suite", "impl", "mode", "n", "d", "coupling", "seed", "eps", "iters")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", choices=["cpu", "gpu"], default=None)
    p.add_argument(
        "--suite",
        nargs="+",
        default=["lse"],
        choices=["lse", "accuracy", "scale", "sinkhorn", "peak", "cfm"],
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--append", action="store_true")
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument("--dims", type=int, nargs="+", default=[2, 8, 32])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cfm-steps", type=int, default=3000)
    p.add_argument("--cfm-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--peak-child", nargs=4, default=None, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.peak_child:
        impl, mode, n, d = a.peak_child
        _peak_child(impl, mode, int(n), int(d))
        return
    device = a.device or ("cpu" if jax.default_backend() == "cpu" else "gpu")
    out = a.output or Path(f"docs/benchmarks/kernel_ot_{device}.json")
    gpu = device == "gpu"
    max_dense = (20 if gpu else 4) * 2**30
    budget = 60.0
    rows = []
    for suite in a.suite:
        if suite == "lse":
            sizes = a.sizes or (
                [1024, 4096, 16384, 65536] if gpu else [1024, 4096, 16384]
            )
            impls = ["pallas" if gpu else "xla", "dense"]
            rows += lse_rows(device, sizes, a.dims, impls, a.repeats, max_dense, budget)
        elif suite == "accuracy":
            rows += lse_accuracy_rows(device)
        elif suite == "scale":
            ks = a.sizes or ([131072, 262144, 1048576] if gpu else [65536])
            ds = [32768, 49152, 65536] if gpu else [32768]
            rows += scale_rows(device, ks, ds, 2, 600.0)
        elif suite == "sinkhorn":
            sizes = a.sizes or ([4096, 16384, 65536] if gpu else [4096, 16384])
            rows += sinkhorn_rows(device, sizes, 8, 50, a.repeats, max_dense, budget)
        elif suite == "peak":
            rows += peak_rows()
        elif suite == "cfm":
            rows += cfm_rows(a.cfm_steps, 256, [1, 2, 4, 8, 32], a.cfm_seeds, 4096)
    if a.append and out.exists():
        old = json.loads(out.read_text())["results"]
        new_keys = {_row_key(r) for r in rows}
        rows = [r for r in old if _row_key(r) not in new_keys] + rows
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(device=device_info(), config=dict(epsilon=EPS), results=rows), indent=2
        )
        + "\n"
    )
    print(f"wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
