"""CPU/GPU benchmark of probjax neural-network modules (forward and train step).

Run from the repository root::

    python -m docs.scripts.benchmark_kernel_nn --device gpu \
        --output docs/benchmarks/kernel_nn_gpu.json
    python -m docs.scripts.benchmark_kernel_nn --device cpu \
        --output docs/benchmarks/kernel_nn_cpu.json

Every row reports, next to the steady-state time, the cost of getting there:
``trace_s`` (Python tracing + lowering to StableHLO), ``compile_s`` (trace +
XLA compile) and XLA's static memory plan (``xla_temp_mb`` / ``xla_arg_mb`` /
``xla_out_mb`` from ``compiled.memory_analysis()``). ``--peak`` re-runs the
headline cases in a fresh subprocess and records the device's
``peak_bytes_in_use`` (GPU only).

Suites:

* ``transformer`` / ``diffusion`` -- ``Transformer`` / ``DiffusionTransformer``
  (4 layers, width 256, 8 heads) with each available ``attention_fn``. The
  ``no_attention`` row replaces attention by ``value`` (an ablation, not a
  model) so ``1 - no_attention / probjax_default`` is attention's share.
* ``ssm``  -- ``SSMModel`` with ``MambaCell`` / ``SSDCell`` / ``LRUCell``.
* ``unet`` -- ``UNet((64, 128, 256))`` on image batches.
* ``train_loop`` -- per-update cost of ``probjax.nn.training.fit_module``
  (array data, i.e. the on-device ``lax.scan`` loop, and a host stream)
  against a bare ``jax.jit`` AdamW step, plus the cost of a second ``fit``
  call on the same model (re-tracing / re-compiling).
* ``remat`` -- memory/time of the train step with ``jax.checkpoint`` around
  the whole loss vs none, and with parameter/optimizer buffer donation.
* ``moe`` -- dropless ``MoELayer`` implementations (owned by the gmm lane;
  kept for completeness, not run by default).

A train step is ``value_and_grad`` of an MSE loss followed by an AdamW update,
all under one ``jit``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULTS = {
    "gpu": dict(
        tokens=[256, 1024, 4096],
        batch=8,
        ssm_tokens=[1024, 4096],
        unet_res=[32, 64],
        moe_tokens=[4096, 32768],
        scan_tokens=4096,
        scan_batch=2,
    ),
    "cpu": dict(
        tokens=[128, 512],
        batch=4,
        ssm_tokens=[256, 1024],
        unet_res=[32],
        moe_tokens=[1024],
        scan_tokens=512,
        scan_batch=2,
    ),
}


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["transformer", "diffusion", "ssm", "unet", "train_loop", "remat"],
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--tokens", type=int, nargs="+", default=None)
    parser.add_argument("--ssm-tokens", type=int, nargs="+", default=None)
    parser.add_argument("--unet-res", type=int, nargs="+", default=None)
    parser.add_argument("--moe-tokens", type=int, nargs="+", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--attention", nargs="+", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--scan-tokens", type=int, default=None)
    parser.add_argument("--scan-batch", type=int, default=None)
    parser.add_argument("--cells", nargs="+", default=["mamba", "ssd", "lru"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--peak",
        action="store_true",
        help="also measure the device peak of headline cases in subprocesses",
    )
    # Internal: run one case in this (fresh) process and print its peak.
    parser.add_argument("--peak-case", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key, value in DEFAULTS[args.device].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.output is None and args.peak_case is None:
        parser.error("--output is required")
    return args


ARGS = _parse()
if ARGS.device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402

from docs.scripts._gpu_bench import run_safely, time_jax, write_results  # noqa: E402
from probjax.nn import DiffusionTransformer, Transformer  # noqa: E402
from probjax.nn.layers.attention import flex_attention  # noqa: E402
from probjax.nn.layers.ssm import LRUCell, MambaCell, SSDCell  # noqa: E402
from probjax.nn.nets.ssm import SSMModel  # noqa: E402
from probjax.nn.nets.unets import UNet  # noqa: E402

MB = 2**20


def _jax_attention(implementation):
    def attention_fn(query, key, value, mask=None, bias=None, **kwargs):
        if mask is not None or bias is not None:
            raise ValueError("benchmark wrapper supports unmasked attention only")
        return jax.nn.dot_product_attention(
            query, key, value, implementation=implementation
        )

    return attention_fn


def _no_attention(query, key, value, **kwargs):
    """Ablation: skip the attention kernel (keeps the q/k/v/out projections)."""
    del query, key, kwargs
    return value


ATTENTION_FNS = {
    "probjax_default": None,
    "probjax_flex": flex_attention,
    "jax_xla": _jax_attention("xla"),
    "jax_cudnn": _jax_attention("cudnn"),
    "no_attention": _no_attention,
}
DEFAULT_ATTENTION = {
    "gpu": ["probjax_default", "probjax_flex", "jax_xla", "jax_cudnn", "no_attention"],
    "cpu": ["probjax_default", "jax_xla", "no_attention"],
}


# ---------------------------------------------------------------------------
# Timing + memory of one jitted function
# ---------------------------------------------------------------------------


def _memory(compiled) -> dict:
    try:
        mem = compiled.memory_analysis()
    except Exception:  # pragma: no cover - backend dependent
        return {}
    if mem is None:
        return {}
    return dict(
        xla_temp_mb=mem.temp_size_in_bytes / MB,
        xla_arg_mb=mem.argument_size_in_bytes / MB,
        xla_out_mb=mem.output_size_in_bytes / MB,
        xla_alias_mb=getattr(mem, "alias_size_in_bytes", 0) / MB,
    )


def time_compiled(jitted, *args, repeats: int | None = None) -> dict:
    """AOT trace/compile timing, XLA memory plan, then steady-state timing."""
    start = time.perf_counter()
    lowered = jitted.lower(*args)
    trace_s = time.perf_counter() - start
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start
    timing = time_jax(compiled, *args, repeats=repeats or ARGS.repeats)
    timing["first_call_s"] = timing.pop("compile_s")
    return dict(trace_s=trace_s, compile_s=compile_s, **_memory(compiled), **timing)


def make_steps(model, predict, *, remat: bool = False):
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    tx = optax.adamw(1e-4)
    opt_state = tx.init(params)

    def loss_fn(params, rest, *inputs):
        out = predict(nnx.merge(graphdef, params, rest), *inputs)
        return jnp.mean(jnp.square(out.astype(jnp.float32)))

    if remat:
        loss_fn = jax.checkpoint(loss_fn)

    def forward(params, rest, *inputs):
        return predict(nnx.merge(graphdef, params, rest), *inputs)

    def train_step(params, opt_state, rest, *inputs):
        loss, grads = jax.value_and_grad(loss_fn)(params, rest, *inputs)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return params, opt_state, rest, forward, train_step


def bench_module(rows, meta, model, inputs, predict, ops=("forward", "train_step")):
    params, opt_state, rest, forward, train_step = make_steps(model, predict)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    cases = dict(
        forward=(jax.jit(forward), (params, rest, *inputs)),
        train_step=(jax.jit(train_step), (params, opt_state, rest, *inputs)),
    )
    for op in ops:
        fn, args = cases[op]

        def case(fn=fn, args=args):
            timing = time_compiled(fn, *args)
            out = fn(*args)
            finite = all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(out))
            return dict(finite=finite, **timing)

        _emit(rows, dict(**meta, op=op, params=n_params, **run_safely(case)))


def _emit(rows, row):
    rows.append(row)
    print(
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
        flush=True,
    )


def _guard(rows, meta, fn):
    err = run_safely(fn)
    if err:
        _emit(rows, dict(**meta, **err))
    jax.clear_caches()


# ---------------------------------------------------------------------------
# Model builders (shared with --peak subprocesses)
# ---------------------------------------------------------------------------


def build_transformer(
    diffusion, attention, dtype_name, tokens, batch, model_dim, **extra
):
    dtype = jnp.dtype(dtype_name)
    kw = dict(
        model_dim=model_dim,
        num_heads=8,
        num_layers=4,
        attn_size=model_dim // 8,
        attention_fn=ATTENTION_FNS[attention],
        dtype=dtype,
        rngs=nnx.Rngs(0),
    )
    kw.update(extra)
    if diffusion:
        model = DiffusionTransformer(8, **kw)
        x = jax.random.normal(jax.random.key(1), (batch, tokens, 8), dtype)
        t = jnp.full((batch,), 0.5, dtype)
        return model, (t, x), lambda m, t, x: m(t, x)
    model = Transformer(**kw)
    x = jax.random.normal(jax.random.key(1), (batch, tokens, model_dim), dtype)
    return model, (x,), lambda m, x: m(x)


SSM_CELLS = {"mamba": MambaCell, "ssd": SSDCell, "lru": LRUCell}


def build_ssm(cell, tokens, batch):
    model = SSMModel(16, 128, 16, 4, recurrent_cls=SSM_CELLS[cell], rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (batch, tokens, 16))
    return model, (x,), lambda m, x: m(x)


def build_unet(dtype_name, res, batch):
    dtype = jnp.dtype(dtype_name)
    model = UNet(3, (64, 128, 256), dtype=dtype, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (batch, res, res, 3), dtype)
    return model, (x,), lambda m, x: m(x)


def build(spec: dict):
    kind = spec["suite"]
    if kind in ("transformer", "diffusion"):
        return build_transformer(
            kind == "diffusion",
            spec["attention"],
            spec["dtype"],
            spec["tokens"],
            spec["batch"],
            spec["model_dim"],
        )
    if kind == "scan_remat":
        extra = dict(num_layers=spec["layers"], scan_layers=spec["scan"])
        if spec["remat"] != "none":
            extra["remat_policy"] = spec["remat"]
        return build_transformer(
            spec["model"] == "diffusion",
            spec["attention"],
            spec["dtype"],
            spec["tokens"],
            spec["batch"],
            spec["model_dim"],
            **extra,
        )
    if kind == "ssm":
        return build_ssm(spec["cell"], spec["tokens"], spec["batch"])
    if kind == "unet":
        return build_unet(spec["dtype"], spec["resolution"], spec["batch"])
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def suite_transformer(rows, args, diffusion=False):
    attention = args.attention or DEFAULT_ATTENTION[args.device]
    for dtype_name in args.dtypes:
        for tokens in args.tokens:
            for attn_name in attention:
                if attn_name == "jax_cudnn" and dtype_name == "float32":
                    continue
                meta = dict(
                    suite="diffusion" if diffusion else "transformer",
                    attention=attn_name,
                    dtype=dtype_name,
                    tokens=tokens,
                    batch=args.batch,
                    model_dim=args.model_dim,
                )

                def run(meta=meta):
                    bench_module(rows, meta, *build(meta))
                    return {}

                _guard(rows, meta, run)


def suite_ssm(rows, args):
    # SSDCell's defaults (num_heads=1) are rejected by the GPU SSD kernel
    # (needs >= 2 heads); recorded as an error row (maintainer question).
    for tokens in args.ssm_tokens:
        for cell in args.cells:
            meta = dict(
                suite="ssm",
                cell=cell,
                dtype="float32",
                tokens=tokens,
                batch=args.batch,
                model_dim=128,
            )

            def run(meta=meta):
                bench_module(rows, meta, *build(meta))
                return {}

            _guard(rows, meta, run)


def suite_unet(rows, args):
    for dtype_name in args.dtypes:
        for res in args.unet_res:
            meta = dict(
                suite="unet", dtype=dtype_name, resolution=res, batch=args.batch
            )

            def run(meta=meta):
                bench_module(rows, meta, *build(meta))
                return {}

            _guard(rows, meta, run)


def suite_moe(rows, args):
    from probjax.nn.layers.moe import MoELayer

    for dtype_name in args.dtypes:
        dtype = jnp.dtype(dtype_name)
        for n_tokens in args.moe_tokens:
            for impl in ("pallas", "blocked", "ragged_dot"):
                meta = dict(
                    suite="moe", impl=impl, dtype=dtype_name, tokens=n_tokens,
                    experts=8, top_k=2, d_model=512, d_hidden=1024,
                )  # fmt: skip

                def run(impl=impl, dtype=dtype, n_tokens=n_tokens, meta=meta):
                    model = MoELayer(
                        512, 1024, 8, top_k=2, routing="dropless",
                        dropless_impl=impl, dtype=dtype, rngs=nnx.Rngs(0),
                    )  # fmt: skip
                    x = jax.random.normal(jax.random.key(1), (n_tokens, 512), dtype)
                    bench_module(rows, meta, model, (x,), lambda m, x: m(x))
                    return {}

                _guard(rows, meta, run)


def suite_train_loop(rows, args):
    """Per-update overhead of ``fit_module`` vs a bare jitted AdamW step."""
    from probjax.nn.nets.simple import MLP
    from probjax.nn.training import fit_module

    num_steps = 200
    for width, batch in ((64, 256), (512, 4096)):
        meta = dict(suite="train_loop", model="MLP", width=width, batch=batch)
        x = jax.random.normal(jax.random.key(1), (batch, 16))

        def predict(m, x):
            return m(x)

        def loss(model, rng, batch):
            del rng
            return jnp.mean(jnp.square(model(batch)))

        def fresh(width=width):
            return MLP([16, width, width, 16], rngs=nnx.Rngs(0))

        def bare(meta=meta, x=x):
            model = fresh()
            params, opt_state, rest, _, train_step = make_steps(model, predict)
            step = jax.jit(train_step)
            timing = time_compiled(step, params, opt_state, rest, x)
            return dict(per_step_ms=timing["median_ms"], **timing)

        def fit_array(meta=meta, x=x):
            out = {}
            for call in ("first", "second"):
                model = fresh()
                jax.block_until_ready(nnx.state(model))
                start = time.perf_counter()
                res = fit_module(
                    model, jax.random.key(0), x, loss=loss, num_steps=num_steps,
                    batch_size=None,
                )  # fmt: skip
                jax.block_until_ready(res.losses)
                out[f"{call}_call_s"] = time.perf_counter() - start
            # steady state: time a longer run minus the fixed cost
            model = fresh()
            start = time.perf_counter()
            res = fit_module(
                model, jax.random.key(0), x, loss=loss, num_steps=num_steps * 5,
                batch_size=None,
            )  # fmt: skip
            jax.block_until_ready(res.losses)
            long_s = time.perf_counter() - start
            out["per_step_ms"] = (long_s - out["second_call_s"]) / (4 * num_steps) * 1e3
            return out

        def fit_stream(meta=meta, x=x):
            out = {}
            for call in ("first", "second"):
                model = fresh()
                start = time.perf_counter()
                res = fit_module(
                    model, jax.random.key(0), [x] * num_steps, loss=loss,
                    num_steps=num_steps,
                )  # fmt: skip
                jax.block_until_ready(res.losses)
                out[f"{call}_call_s"] = time.perf_counter() - start
            model = fresh()
            start = time.perf_counter()
            res = fit_module(
                model, jax.random.key(0), [x] * (5 * num_steps), loss=loss,
                num_steps=5 * num_steps,
            )  # fmt: skip
            jax.block_until_ready(res.losses)
            long_s = time.perf_counter() - start
            out["per_step_ms"] = (long_s - out["second_call_s"]) / (4 * num_steps) * 1e3
            return out

        for name, fn in (("bare_jit", bare), ("fit_array", fit_array),
                         ("fit_stream", fit_stream)):  # fmt: skip
            _emit(rows, dict(**meta, op=name, **run_safely(fn)))


SCAN_REMAT_VARIANTS = (
    (False, "none"),
    (False, "full"),
    (False, "dots_saveable"),
    (True, "none"),
    (True, "full"),
)


def suite_scan_remat(rows, args):
    """Opt-in ``scan_layers`` / ``remat_policy`` of a deep Transformer."""
    gpu = args.device == "gpu"
    for model in ("transformer", "diffusion"):
        for attention in args.attention or ["probjax_default", "probjax_flex"]:
            if not gpu and attention != "probjax_default":
                continue
            for layers in args.layers:
                for scan, remat in SCAN_REMAT_VARIANTS:
                    spec = dict(
                        suite="scan_remat",
                        model=model,
                        attention=attention,
                        dtype="bfloat16" if gpu else "float32",
                        tokens=args.scan_tokens,
                        batch=args.scan_batch,
                        model_dim=512,
                        layers=layers,
                        scan=scan,
                        remat=remat,
                    )

                    def run(spec=spec):
                        bench_module(rows, spec, *build(spec), ops=("train_step",))
                        return {}

                    _guard(rows, spec, run)
                    if gpu and args.peak:
                        row = dict(spec, op="train_step", measure="device_peak")
                        _emit(rows, dict(row, **peak_subprocess(row)))


def suite_remat(rows, args):
    """Train-step memory/time: plain vs jax.checkpoint(loss) vs donation."""
    if args.device == "gpu":
        specs = [
            dict(suite="transformer", attention="probjax_default", dtype="bfloat16",
                 tokens=1024, batch=8, model_dim=256),
            dict(suite="diffusion", attention="probjax_flex", dtype="bfloat16",
                 tokens=4096, batch=8, model_dim=256),
            dict(suite="unet", dtype="bfloat16", resolution=64, batch=8),
            dict(suite="ssm", cell="lru", dtype="float32", tokens=4096, batch=8),
        ]  # fmt: skip
    else:
        specs = [
            dict(suite="transformer", attention="probjax_default", dtype="float32",
                 tokens=512, batch=4, model_dim=256),
            dict(suite="unet", dtype="float32", resolution=32, batch=4),
            dict(suite="ssm", cell="lru", dtype="float32", tokens=1024, batch=4),
        ]  # fmt: skip
    for spec in specs:
        for variant in ("plain", "remat", "donate"):
            meta = dict(spec, op="train_step", variant=variant)

            def run(spec=spec, variant=variant, meta=meta):
                model, inputs, predict = build(spec)
                params, opt_state, rest, _, train_step = make_steps(
                    model, predict, remat=variant == "remat"
                )
                if variant == "donate":
                    step = jax.jit(train_step, donate_argnums=(0, 1))

                    # Donated buffers are consumed: thread the state through.
                    state = [params, opt_state]

                    def call(*inputs):
                        p, o, loss = step(state[0], state[1], rest, *inputs)
                        state[0], state[1] = p, o
                        return loss

                    start = time.perf_counter()
                    compiled = step.lower(params, opt_state, rest, *inputs).compile()
                    compile_s = time.perf_counter() - start
                    timing = time_jax(call, *inputs)
                    timing.pop("compile_s")
                    _emit(rows, dict(**meta, compile_s=compile_s,
                                     **_memory(compiled), **timing))  # fmt: skip
                else:
                    step = jax.jit(train_step)
                    timing = time_compiled(step, params, opt_state, rest, *inputs)
                    _emit(rows, dict(**meta, **timing))
                return {}

            _guard(rows, meta, run)


# ---------------------------------------------------------------------------
# Device peak in a fresh process
# ---------------------------------------------------------------------------


def _peak_case(spec_json: str) -> None:
    spec = json.loads(spec_json)
    model, inputs, predict = build(spec)
    params, opt_state, rest, forward, train_step = make_steps(model, predict)
    if spec["op"] == "forward":
        out = jax.jit(forward)(params, rest, *inputs)
    else:
        out = jax.jit(train_step)(params, opt_state, rest, *inputs)
    jax.block_until_ready(out)
    stats = jax.devices()[0].memory_stats() or {}
    print(
        "PEAK_JSON " + json.dumps({"peak_mb": stats.get("peak_bytes_in_use", 0) / MB})
    )


def peak_subprocess(spec: dict) -> dict:
    cmd = [sys.executable, "-m", "docs.scripts.benchmark_kernel_nn",
           "--device", ARGS.device, "--peak-case", json.dumps(spec)]  # fmt: skip
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    for line in proc.stdout.splitlines():
        if line.startswith("PEAK_JSON "):
            return json.loads(line[len("PEAK_JSON ") :])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
    return dict(peak_error=tail[0][:300] if tail else f"exit {proc.returncode}")


def headline_peaks(rows, args):
    if args.device != "gpu":
        return
    specs = []
    for diffusion in (False, True):
        for attention in ("probjax_default", "probjax_flex", "jax_cudnn"):
            for tokens in (1024, 4096):
                specs.append(dict(suite="diffusion" if diffusion else "transformer",
                                  attention=attention, dtype="bfloat16", tokens=tokens,
                                  batch=8, model_dim=256))  # fmt: skip
    for cell in ("mamba", "lru"):
        specs.append(
            dict(suite="ssm", cell=cell, dtype="float32", tokens=1024, batch=8)
        )
    for dtype in ("float32", "bfloat16"):
        specs.append(dict(suite="unet", dtype=dtype, resolution=64, batch=8))
    for spec in specs:
        for op in ("forward", "train_step"):
            row = dict(spec, op=op, measure="device_peak")
            _emit(rows, dict(row, **peak_subprocess(row)))


def main():
    args = ARGS
    if args.peak_case is not None:
        _peak_case(args.peak_case)
        return
    rows: list[dict] = []
    config = {k: v for k, v in vars(args).items() if k != "peak_case"}
    config["output"] = str(args.output)
    suites = dict(
        transformer=lambda: suite_transformer(rows, args),
        diffusion=lambda: suite_transformer(rows, args, diffusion=True),
        ssm=lambda: suite_ssm(rows, args),
        unet=lambda: suite_unet(rows, args),
        moe=lambda: suite_moe(rows, args),
        train_loop=lambda: suite_train_loop(rows, args),
        remat=lambda: suite_remat(rows, args),
        scan_remat=lambda: suite_scan_remat(rows, args),
    )
    for suite in args.suites:
        suites[suite]()
        write_results(args.output, config, rows)
    if args.peak:
        headline_peaks(rows, args)
        write_results(args.output, config, rows)


if __name__ == "__main__":
    main()
