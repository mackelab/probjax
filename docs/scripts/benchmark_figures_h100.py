"""The H100 page (``docs/benchmarks/gpu_h100.md``) and its figures.

Called by :mod:`docs.scripts.benchmark_figures` (``python -m
docs.scripts.benchmark_figures`` regenerates every page).  Every number and
figure comes from a JSON row:

* ``gpu_attention_h100``, ``gpu_attention_tiles_h100``, ``gpu_kernels_h100``,
  ``gpu_nn_h100`` -- the suites of the RTX 3090 pages, run on an H100;
* ``gpu_attention_h100_device`` -- implementations by shape
  (``docs/scripts/benchmark_gpu_attention_device.py``), incl. masked
  cross-attention, PyTorch and the Mosaic GPU kernel;
* ``gpu_moe_h100`` -- MoE dropless dispatch (``benchmark_gpu_moe_dispatch.py``).

The sm_90 routing rules quoted on the page are read from
``probjax.nn.layers.attention_device.DEVICE_ATTENTION_DEFAULTS``.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from docs.scripts.benchmark_figures import (
    AQUA,
    DT,
    FOOT_STATUS,
    GREEN,
    MAGENTA,
    ORANGE,
    P1,
    P2,
    P3,
    STY,
    VIOLET,
    YELLOW,
    S,
    attention_time_figures,
    bar_panel,
    dot_panel,
    f_ms,
    f_x,
    get,
    img,
    isnum,
    line_panel,
    load,
    rows,
    save_figure,
    table,
)

GAH = "gpu_attention_h100"
DEV = "gpu_attention_h100_device"
KH, NH, MOEH = "gpu_kernels_h100", "gpu_nn_h100", "gpu_moe_h100"
RTX = {"attention": "gpu_attention", "kernels": "gpu_kernels", "nn": "gpu_nn"}

STY.update(
    {
        "dev:auto": S("generic auto (before)", P3, "^", ":", md="generic `auto`"),
        "dev:split": S("Triton split backward", P2, "s", "--", md="Triton split"),
        "dev:onepass": S("Triton single-pass backward", AQUA, "D", "-.", md="single-pass"),
        "dev:cudnn": S("cuDNN", ORANGE, "v", md="cuDNN"),
        "dev:default": S("H100 default", P1, md="H100 default"),
        "dev:mosaic": S("Mosaic GPU", GREEN, "p", "-.", md="Mosaic GPU"),
        "dev:torch_flex": S("torch flex_attention", MAGENTA, "P", md="torch `flex_attention`"),
        "dev:torch_flash": S("torch SDPA flash", VIOLET, "X", md="torch SDPA flash"),
        "moe:h100:triton/sorted": S("triton / sorted", P3, "^", ":"),
        "moe:h100:triton/gather": S("triton / gather", P2, "D", "-."),
        "moe:h100:triton/fused": S("triton / fused", P2, "s", "--"),
        "moe:h100:triton/fused_gsum": S("triton / fused_gsum", P1),
        "moe:h100:mosaic/sorted": S("mosaic / sorted (Hopper ragged dot)", ORANGE, "v"),
        "moe:h100:ragged_dot/sorted": S("XLA ragged_dot", YELLOW, "h", "--"),
    }
)


# --------------------------------------------------------------------------- data
def dev(**filt):
    return [r for r in rows(DEV, **filt) if r.get("status") == "ok"]


def dev_time(shape, impl, labels):
    """fwd+bwd ms of ``impl`` on ``shape`` from the first label that has it."""
    for lbl in labels:
        for r in rows(DEV, shape=shape, impl=impl, label=lbl):
            if r.get("status") == "ok":
                return r["fwd_bwd_ms"]
            if r.get("error"):
                e = r["error"]
                return "no mask support" if "no attention mask" in e else (
                    "OOM" if "memory" in e.lower() or "RESOURCE_EXHAUSTED" in e else "error")
    return None


def shape_meta(shape):
    for r in rows(DEV, shape=shape):
        return r["B"], r["Lq"], r["Lkv"], r["H"], r["D"], r["masked"]
    raise KeyError(shape)


def shape_name(shape):
    B, lq, lkv, h, d, m = shape_meta(shape)
    return f"{B} × {lq} × {lkv}, {h}, {d}" + (" (KV lengths)" if m else "")


def sm90():
    from probjax.nn.layers.attention_device import DEVICE_ATTENTION_DEFAULTS

    return DEVICE_ATTENTION_DEFAULTS[9]


def routed(shape) -> str:
    """The sm_90 default for a bf16 call of this shape: "cudnn" or the Triton backward."""
    d = sm90()
    B, lq, lkv, h, hd, _ = shape_meta(shape)
    bd = max(32, 1 << (hd - 1).bit_length())
    min_work = d.cudnn_min_work.get(bd)
    fused = (
        hd % 8 == 0
        and min_work is not None
        and B * h * lq * lkv >= min_work
        and not (lkv <= d.cudnn_short_kv and lq >= d.cudnn_short_kv_ratio * lkv)
        and not (lq <= getattr(d, "cudnn_min_q", 0) and lkv >= d.onepass_kv_ratio * lq)
    )
    if fused:
        return "cudnn"
    if lq <= d.onepass_max_q and lkv >= d.onepass_kv_ratio * lq:
        return "triton_onepass"
    return "triton_split"


def has_mosaic_backend() -> bool:
    return bool(rows(DEV, impl="flex:mosaic"))


GENERIC = ["pr123_mla", "pr123_isab", "pr123_generic_d32", "pr123_generic_d64", "pr123_generic_d128"]
CUDNN_ANY = GENERIC + ["main_mla", "main_isab"]
SHAPES_DEFAULT = [
    "mla_32k84", "mla_128k36", "mla_1k756", "isab_ind_rows", "isab_rows_ind", "isab_1k_ind",
    "self_d32_t8192", "self_d64_t2048", "self_d64_t8192", "self_d128_t2048", "self_d128_t8192",
    "self_d128_t16384", "selfm_d64_t8192", "selfm_d128_t16384", "self_d64_t512",
    "cross_d64_256x16384", "cross_d128_256x16384", "cross_d64_16384x256", "cross_d128_16384x256",
]
SHAPES_SMALLQ = [
    "sq_isab_256x131072", "sq_isab_256x32768", "sq_128x262144_d64", "sq_64x131072_d64",
    "sq_16x65536_d128", "sq_512x131072_d32",
]
SHAPES_TORCH = [
    "mla_32k84", "mla_128k36", "mla_1k756", "selfm_d128_t16384", "self_d128_t8192",
    "selfm_d64_t8192", "cross_d128_256x16384", "cross_d64_16384x256", "sq_128x262144_d64",
    "isab_ind_rows", "sq_isab_256x131072", "isab_1k_ind",
]
TORCH_LABELS = ["h100_torch_A", "h100_torch_B"]


def cudnn_speedup_range() -> tuple[float, float] | None:
    """min/max fwd+bwd speed-up of the cuDNN route over the generic dispatch."""
    sp = [auto / dflt for _, auto, _, _, _, dflt, path in default_rows()
          if path == "cudnn" and isnum(auto) and isnum(dflt)]
    return (min(sp), max(sp)) if sp else None


def default_rows():
    """(shape, generic auto, split, single-pass, cuDNN, default, path) per shape."""
    out = []
    for s in SHAPES_DEFAULT:
        if not rows(DEV, shape=s):
            continue
        auto = dev_time(s, "flex:auto", GENERIC)
        split = dev_time(s, "flex:triton_split", GENERIC)
        one = dev_time(s, "flex:triton_onepass", GENERIC)
        cud = dev_time(s, "cudnn", CUDNN_ANY)
        path = routed(s)
        dflt = cud if path == "cudnn" else (one if path == "triton_onepass" else split)
        out.append((s, auto, split, one, cud, dflt, path))
    return out


def speedups(rtx_src, h_src, key_fn, group):
    a = load(h_src)["results"]
    b = load(rtx_src)["results"]
    skip = {"median_ms", "q1_ms", "q3_ms", "compile_s", "inner", "max_abs_err", "finite",
            "tag", "rel_err", "params"}
    keyf = lambda r: tuple(sorted((k, str(v)) for k, v in r.items() if k not in skip))  # noqa: E731
    ref = {keyf(r): r.get("median_ms") for r in b}
    g = defaultdict(list)
    for r in a:
        rv = ref.get(keyf(r))
        if isnum(rv) and isnum(r.get("median_ms")):
            g[r[group]].append(rv / r["median_ms"])
    return {k: sorted(v) for k, v in g.items()}


# --------------------------------------------------------------------------- figures
def h100_figures() -> dict[str, str]:
    figs = attention_time_figures(GAH, "_h100")

    # implementations per shape (fwd+bwd), the measurements behind the defaults
    cats, ser = [], defaultdict(dict)
    for s, auto, split, _one, cud, dflt, _ in default_rows():
        cats.append(shape_name(s))
        for key, v in (("dev:auto", auto), ("dev:split", split), ("dev:cudnn", cud),
                       ("dev:default", dflt)):
            ser[key][cats[-1]] = v
        tf = dev_time(s, "torch:flex", TORCH_LABELS)
        if tf is not None:
            ser["dev:torch_flex"][cats[-1]] = tf
        if has_mosaic_backend():
            mo = dev_time(s, "flex:mosaic", TORCH_LABELS)
            if mo is not None:
                ser["dev:mosaic"][cats[-1]] = mo
    order = ["dev:auto", "dev:split", "dev:cudnn", "dev:mosaic", "dev:torch_flex", "dev:default"]
    series = [(k, ser[k]) for k in order if ser.get(k)]
    figs["defaults"] = save_figure(
        "h100_attention_shapes",
        [lambda ax: dot_panel(ax, cats, series, title="fwd+bwd by shape (bf16)", xlabel="time (ms)")],
        1, width=8.5, panel_h=0.26 * len(cats) + 0.9, legend_ncol=3,
    )

    # H100 vs RTX 3090: median speed-up per family / suite
    fam = speedups(RTX["kernels"], KH, None, "family")
    suite = speedups(RTX["nn"], NH, None, "suite")
    labels = [f"kernels · {k}" for k in fam] + [f"models · {k}" for k in suite]
    vals = [statistics.median(v) for v in fam.values()] + [statistics.median(v) for v in suite.values()]
    figs["vs_rtx"] = save_figure(
        "h100_vs_rtx3090",
        [lambda ax: bar_panel(ax, labels, vals, title="RTX 3090 time ÷ H100 time (median over rows)",
                              xlabel="speed-up")],
        1, width=7.0, panel_h=0.3 * len(labels) + 0.9, legend=False,
    )

    # MoE dispatch
    ser = []
    for impl, disp in (("triton", "sorted"), ("triton", "gather"), ("triton", "fused"),
                       ("triton", "fused_gsum"), ("mosaic", "sorted"), ("ragged_dot", "sorted")):
        data = {}
        for r in rows(MOEH, impl=impl, dispatch=disp):
            data[r["tokens"]] = r.get("train_ms") if r.get("status") == "ok" else (
                "OOM" if "RESOURCE_EXHAUSTED" in r.get("error", "") else "fail†")
        if data:
            ser.append((f"moe:h100:{impl}/{disp}", data))
    figs["moe"] = save_figure(
        "kernel_moe_h100",
        [lambda ax: line_panel(ax, ser, title="MoE train step (bf16)", xlabel="tokens",
                               ylabel="time (ms)", xticks=sorted({r["tokens"] for r in rows(MOEH)}))],
        1, width=5.5, panel_h=3.0, legend_ncol=2,
    )
    return figs


# --------------------------------------------------------------------------- page
def _t(v):
    return "—" if v is None else (v if isinstance(v, str) else f_ms(v))


def _bold_min(cells, values):
    nums = [(v, i) for i, v in enumerate(values) if isnum(v)]
    if nums:
        i = min(nums)[1]
        cells[i] = f"**{cells[i]}**"
    return cells


def page_h100(figs) -> str:
    o = []
    A = o.append
    d = sm90()
    dk = load(GAH)["device"]["device_kind"]
    A("# H100 (Hopper)\n")
    A(
        "The same suites as on the [RTX 3090 pages](gpu_overview.md), run on an NVIDIA H100, "
        "plus the measurements behind the device-specific defaults for Hopper. Times are medians "
        "in milliseconds; inputs are `[batch, seq, heads, head_dim]` and bf16 unless stated "
        "otherwise. " + FOOT_STATUS + "\n"
    )
    A("## Setup\n")
    A(table(["", ""], [
        ["GPU", f"{dk} NVL, 94 GB (sm_90, 227 KB opt-in shared memory per block)"],
        ["Software", f"JAX {load(GAH)['device']['jax']}, cuDNN 9.27, CUDA 13.4 (pip wheels), "
                     "Python 3.13; PyTorch 2.14.1+cu130 for the torch rows"],
        ["PyTorch rows", "measured in a separate process with JAX's GPU preallocation off "
                         "(`XLA_PYTHON_CLIENT_PREALLOCATE=false`, platform allocator), so torch "
                         "has the whole GPU; JAX only builds the inputs and the reference"],
        ["Results", "[`gpu_attention_h100.json`](gpu_attention_h100.json), "
                    "[`gpu_attention_tiles_h100.json`](gpu_attention_tiles_h100.json), "
                    "[`gpu_kernels_h100.json`](gpu_kernels_h100.json), "
                    "[`gpu_nn_h100.json`](gpu_nn_h100.json) (the suites of the RTX 3090 pages), "
                    "[`gpu_attention_h100_device.json`](gpu_attention_h100_device.json) "
                    "(`docs/scripts/benchmark_gpu_attention_device.py`), "
                    "[`gpu_moe_h100.json`](gpu_moe_h100.json) "
                    "(`docs/scripts/benchmark_gpu_moe_dispatch.py`)"],
    ], align=["l", "l"]) + "\n")

    # ---------------------------------------------------------------- defaults
    A("## Device-specific defaults\n")
    A(
        "`flex_attention` and `dot_product_attention` look up the GPU's compute capability in "
        "`probjax.nn.layers.attention_device` when every kernel argument is at its default. GPUs "
        "without an entry (RTX 3090 and other non-sm_90 devices) keep the generic rules. On sm_90:\n"
    )
    w = d.cudnn_min_work
    rule_rows = [
        ["bf16/fp16; no mask, `KVLenMask` or causal with Lq = Lkv; no bias or dropout; head_dim a "
         f"multiple of 8, ≤ 128; B·H·Lq·Lkv ≥ 2^{w[64].bit_length() - 1} (head_dim ≥ 64) or "
         f"2^{w[32].bit_length() - 1} (head_dim 32)",
         "cuDNN (`jax.nn.dot_product_attention(implementation=\"cudnn\")`)"],
        [f"…except Lkv ≤ {d.cudnn_short_kv} with Lq ≥ {d.cudnn_short_kv_ratio}·Lkv"
         + (f", and Lq ≤ {d.cudnn_min_q} with Lkv ≥ {d.onepass_kv_ratio}·Lq"
            if hasattr(d, "cudnn_min_q") else ""),
         "Triton (faster there)"],
    ]
    if has_mosaic_backend():
        rule_rows.append([
            "cuDNN cannot take the call (head_dim not a multiple of 8, or "
            "`PROBJAX_ATTENTION_CUDNN=0`), same masks and thresholds, head_dim 33–128",
            "Mosaic GPU ([below](#mosaic-gpu))"])
    rule_rows.append([
        "Everything else, `backward_pass_impl=\"auto\"`",
        f"Triton split backward; single-pass for 16-bit Lq ≤ {d.onepass_max_q}, "
        f"Lkv ≥ {d.onepass_kv_ratio}·Lq"])
    A(table(["Case", "Kernel"], rule_rows, align=["l", "l"]) + "\n")
    A(
        "`implementation=\"triton\"` (per call)"
        + (", `\"cudnn\"` or `\"mosaic\"` (raise if they cannot compute the call exactly)"
           if has_mosaic_backend() else "")
        + ", `PROBJAX_ATTENTION_CUDNN=0`"
        + (", `PROBJAX_ATTENTION_MOSAIC=0`" if has_mosaic_backend() else "")
        + " or `PROBJAX_ATTENTION_DEVICE_DEFAULTS=0` (generic rules) opt out. The cuDNN path is "
        "not bitwise identical to the Triton kernels; outputs and gradients agree to bf16 "
        "precision (max |difference| ≤ 0.004 in `tests/test_attention_device_defaults.py`).\n"
    )
    A("### Why\n")
    dr = default_rows()
    fused_gap = [r[1] / r[2] for r in dr if isnum(r[1]) and isnum(r[2]) and r[1] / r[2] > 1.05]
    split_loss = max(r[2] / r[1] - 1 for r in dr if isnum(r[1]) and isnum(r[2]))
    A(
        "The generic `auto` rule picks the fused backward whenever the two lengths differ by at most "
        f"4×; on H100 that is up to {f_x(max(fused_gap))} slower than the split backward, which is "
        f"never more than {max(split_loss, 0):.0%} slower than `auto`. cuDNN, "
        "which supports per-batch KV lengths, is faster than every Triton variant above the work "
        "thresholds. fwd+bwd, ms (\"generic `auto`\" is the dispatch before the sm_90 entry):\n"
    )
    A(img(figs["defaults"], "Attention implementations by shape on H100",
          "fwd+bwd per implementation and shape, bf16; the H100 default is the filled blue dot."))
    body = []
    for s, auto, split, one, cud, dflt, path in dr:
        cells = [_t(auto), _t(split), _t(one), _t(cud)]
        sp = f_x(auto / dflt) if isnum(auto) and isnum(dflt) else "—"
        body.append([shape_name(s), *cells, f"**{_t(dflt)}**",
                     "cuDNN" if path == "cudnn" else path, sp])
    A(table(["B × Lq × Lkv, H, d", "generic `auto`", "split", "single-pass", "cuDNN",
             "H100 default", "path", "speed-up"], body,
            align=["l", "r", "r", "r", "r", "r", "l", "r"]) + "\n")
    A("Few queries over long KV, measured with the H100 default:\n")
    body = []
    for s in SHAPES_SMALLQ:
        if not rows(DEV, shape=s):
            continue
        vals = [dev_time(s, i, ["h100br_smallq"]) for i in
                ("flex:triton_generic", "flex:triton_onepass", "cudnn", "flex:default")]
        body.append([shape_name(s), *[_t(v) for v in vals[:3]], f"**{_t(vals[3])}**"])
    A(table(["B × Lq × Lkv, H, d", "generic `auto`", "single-pass", "cuDNN", "H100 default"],
            body) + "\n")

    # ---------------------------------------------------------------- mosaic
    A("## Mosaic GPU\n")
    um = defaultdict(dict)
    for r in dev(label="h100br_mosaic_sweep") + dev(label="h100br_mosaic_unmasked"):
        um[r["shape"]][r["impl"]] = r["fwd_bwd_ms"]
    best = "mosaic:64,128,2,2,128,64,64,128"
    A(
        "`mha_flash` runs JAX's Hopper FlashAttention-3 kernel (warpgroup MMA, TMA pipelining), "
        "vendored as `pallas_kernels/kernels/attention_mgpu.py` because upstream hard-codes float16 "
        "shared-memory buffers and rejects every bf16 call. The wrapper takes Lq ≠ Lkv and GQA and "
        "pads query length and head_dim exactly"
        + ("; with per-batch KV lengths it masks and skips padded keys, so any length works"
           if has_mosaic_backend() else
           "; KV lengths that are not a multiple of the KV tile raise (padded keys need an "
           "in-kernel length mask)")
        + ". Its default tiles are the best H100 configuration found (64 × 128 forward, backward "
        "dK/dV 128 × 64 and dQ 64 × 128, two compute warpgroups); deeper pipelines and 128 × 128 "
        "tiles with two warpgroups exceed shared memory at head_dim 128. Unmasked, fwd+bwd ms:\n"
    )
    body = []
    for s in ("self_d64_t8192", "self_d128_t8192", "mla_32k84"):
        v = um.get(s, {})
        old = v.get("mosaic:128,128,2,1", v.get("mosaic_fa3"))
        cells = [_t(old), f"**{_t(v.get(best))}**", _t(v.get("cudnn")), _t(v.get("flex:triton_split"))]
        body.append([shape_name(s).replace(" (KV lengths)", ""), *cells])
    A(table(["B × Lq × Lkv, H, d", "128 × 128 tiles (old default)", "tuned (default)", "cuDNN",
             "Triton split"], body) + "\n")
    if has_mosaic_backend():
        A(
            "`flex_attention(implementation=\"mosaic\")` uses it as a backend with KV lengths: KV "
            "tiles past a batch element's length are neither loaded nor computed in the forward, "
            "dK/dV programs past it store zeros without running their pipeline, rows without a "
            "valid key return zeros. With KV lengths, fwd+bwd ms:\n"
        )
        body = []
        for s in SHAPES_TORCH:
            vals = [dev_time(s, i, TORCH_LABELS) for i in
                    ("flex:mosaic", "flex:cudnn", "flex:triton", "torch:flex")]
            if all(v is None for v in vals):
                continue
            body.append([shape_name(s), *_bold_min([_t(v) for v in vals], vals)])
        A(table(["B × Lq × Lkv, H, d", "Mosaic", "cuDNN", "Triton", "torch `flex_attention`"],
                body) + "\n")
        tri_mo = {s: dev_time(s, "flex:triton", TORCH_LABELS) / dev_time(s, "flex:mosaic", TORCH_LABELS)
                  for s in SHAPES_TORCH if shape_meta(s)[4] >= 64
                  and isnum(dev_time(s, "flex:triton", TORCH_LABELS))
                  and isnum(dev_time(s, "flex:mosaic", TORCH_LABELS))}
        slower = [shape_name(s) for s, v in tri_mo.items() if v < 1]
        A(
            (f"For head_dim ≥ 64 Mosaic is up to {f_x(max(tri_mo.values()))} faster than the Triton "
             "kernels and close to cuDNN"
             + (f" (Triton is faster at {', '.join(slower)}: few keys per query tile)" if slower else "")
             + ". " if tri_mo else "")
            + "cuDNN is faster where it applies, so the sm_90 default tries cuDNN first and lowers "
            "to Mosaic where cuDNN cannot compute the call (head_dim not a multiple of 8, or "
            "`PROBJAX_ATTENTION_CUDNN=0`). head_dim ≤ 32 is padded to 64 and therefore stays on "
            "Triton. In rows with a single valid key the exact dQ and dK are 0 and carry ~1e-1 "
            "rounding noise from the kernel's approximate `exp2` (upstream `approx_math`); "
            "Triton's is ~1e-5. Masks other than `KVLenMask` and equal-length causal (local "
            "windows, segments, functional biases) are not in the Mosaic kernel yet.\n"
        )
    else:
        A(
            "The forward matches cuDNN; the backward is 8–17% slower, so it is not used by "
            "default. For masks and biases cuDNN cannot express, which run on Triton, a masked "
            "Mosaic kernel would be about 1.3–1.5× faster than Triton (follow-up).\n"
        )

    # ---------------------------------------------------------------- suites
    A("## Attention suite\n")
    A(img(figs["time_bfloat16"], "Attention time vs sequence length on H100, bf16",
          "bf16, batch 4, 8 heads, H100. Top row forward, bottom row forward + backward."))
    A(img(figs["time_float32"], "Attention time vs sequence length on H100, fp32",
          "fp32, batch 4, 8 heads, H100. fp32 matmuls run in TF32 (JAX default precision)."))
    A(img(figs["relative"], "Attention time relative to the probjax default on H100",
          "fwd+bwd time of each implementation divided by the probjax default at the same shape; "
          "below the line is faster than probjax."))
    impls = [("probjax_default", "probjax default"), ("jax_cudnn", "`jax.nn` cuDNN"),
             ("jax_xla", "`jax.nn` XLA"), ("torch_flash", "torch SDPA flash"),
             ("torch_efficient", "torch SDPA mem-efficient"), ("torch_math", "torch SDPA math")]
    cases = [("bfloat16", 64, False, 4096), ("bfloat16", 64, False, 8192), ("bfloat16", 64, True, 8192),
             ("bfloat16", 128, False, 4096), ("bfloat16", 128, False, 8192), ("bfloat16", 128, True, 8192),
             ("float32", 64, False, 4096), ("float32", 128, False, 4096)]
    body = []
    for dt, hd, causal, T in cases:
        vals = [get(GAH, impl=i, dtype=dt, head_dim=hd, causal=causal, seq=T, mode="fwd_bwd")
                for i, _ in impls]
        rtx = get(RTX["attention"], impl="probjax_default", dtype=dt, head_dim=hd, causal=causal,
                  seq=T, mode="fwd_bwd")
        body.append([f"{DT[dt]} d{hd}{' causal' if causal else ''} T={T}",
                     *_bold_min([_t(v) for v in vals], vals), _t(rtx)])
    A(table(["fwd+bwd, B = 4, H = 8"] + [n for _, n in impls] + ["RTX 3090 probjax default"],
            body) + "\n")
    flash = [get(GAH, impl="torch_flash", dtype="bfloat16", head_dim=hd, causal=c, seq=T, mode="fwd_bwd")
             / get(GAH, impl="probjax_default", dtype="bfloat16", head_dim=hd, causal=c, seq=T, mode="fwd_bwd")
             for dt, hd, c, T in cases if dt == "bfloat16"
             and isnum(get(GAH, impl="torch_flash", dtype="bfloat16", head_dim=hd, causal=c, seq=T, mode="fwd_bwd"))]
    if flash:
        A(
            f"The probjax default (cuDNN-routed bf16) is {f_x(min(flash))}–{f_x(max(flash))} faster "
            "than torch SDPA flash. fp32 has no flash path in torch; `jax.nn` XLA is the fastest at "
            "fp32 head_dim 128 (see known issues).\n"
        )

    # ---------------------------------------------------------------- vs RTX
    A("## H100 vs RTX 3090\n")
    A(img(figs["vs_rtx"], "Speed-up of the H100 over the RTX 3090 per suite",
          "RTX 3090 time divided by H100 time, median over every row both GPUs ran (same shapes)."))
    body = []
    for name, src_r, src_h, group in (("kernels", RTX["kernels"], KH, "family"),
                                       ("models", RTX["nn"], NH, "suite")):
        for k, v in speedups(src_r, src_h, None, group).items():
            body.append([f"{name} · {k}", len(v), f_x(statistics.median(v)), f"{f_x(v[0])}–{f_x(v[-1])}"])
    A(table(["suite / family", "rows", "median speed-up", "range"], body,
            align=["l", "r", "r", "l"]) + "\n")

    # ---------------------------------------------------------------- MoE
    A("## MoE dispatch\n")
    A(
        "`MoELayer` with dropless routing, bf16, a large decoder MoE (d_model 512, 64 experts, "
        "top-8, expert width 128, shared expert 1024); train step = forward + gradients w.r.t. "
        "parameters and input, ms:\n"
    )
    A(img(figs["moe"], "MoE train step by dispatch on H100", "MoE train step by dispatch, H100."))
    combos = [("triton", "sorted"), ("triton", "gather"), ("triton", "fused"),
              ("triton", "fused_gsum"), ("mosaic", "sorted"), ("ragged_dot", "sorted")]
    body = []
    for n in sorted({r["tokens"] for r in rows(MOEH)}):
        vals = []
        for impl, disp in combos:
            r = next(iter(rows(MOEH, tokens=n, impl=impl, dispatch=disp)), None)
            vals.append(None if r is None else (r["train_ms"] if r.get("status") == "ok" else (
                "OOM" if "RESOURCE_EXHAUSTED" in r.get("error", "") else "fail†")))
        body.append([f"{n:,}", *_bold_min([_t(v) for v in vals], vals)])
    A(table(["tokens"] + [f"`{i}` / `{d_}`" for i, d_ in combos], body) + "\n")
    g = {n: (get(MOEH, "train_ms", tokens=n, impl="mosaic", dispatch="sorted"),
             get(MOEH, "train_ms", tokens=n, impl="triton", dispatch="fused_gsum"))
         for n in sorted({r["tokens"] for r in rows(MOEH)})}
    gains = ", ".join(f"{1 - m / t:.0%} at {n:,}" for n, (m, t) in g.items() if isnum(m) and isnum(t))
    nmax = max(g)
    fm = get(MOEH, "fwd_ms", tokens=nmax, impl="mosaic", dispatch="sorted")
    ft = get(MOEH, "fwd_ms", tokens=nmax, impl="triton", dispatch="fused_gsum")
    fwd = (f"; the forward alone is {f_ms(fm)} vs {f_ms(ft)} ms at {nmax:,} tokens, the gain is in "
           "the backward" if isnum(fm) and isnum(ft) else "; the gain is in the backward")
    A(
        "`\"mosaic\"` runs jax's Hopper ragged-dot kernels (`ragged_dot_mgpu`, "
        "`transposed_ragged_dot_mgpu` for the weight gradient; wgmma + TMA) on the sorted dispatch; "
        "`\"pallas\"` resolves to it on sm_90 for 16-bit compute (the wgmma kernels need 16-bit "
        "operands; float32 runs the Triton grouped matmul). Per train step it is faster than "
        f"`fused_gsum` ({gains} tokens){fwd}. Outputs and gradients agree with `triton` / `sorted` "
        "to bf16 precision (max relative difference ≤ 1.1%). The sorted dispatch keeps top-k "
        "copies of each token (N·K × D buffers), so it needs more activation memory than "
        "`fused_gsum`.\n"
    )

    # ---------------------------------------------------------------- torch
    A("## Against PyTorch\n")
    A(
        "PyTorch 2.14.1 (CUDA 13), same inputs and cotangents as the probjax rows; every torch "
        "result is checked against the probjax reference (gradients agree to bf16 precision, max "
        "|difference| ≤ 0.06). `torch.nn.attention.flex_attention` is compiled (`torch.compile`, "
        "one compiled entry per shape) with a `create_block_mask` built from the KV lengths; SDPA "
        "is pinned to one backend with `sdpa_kernel`. SDPA flash takes no attention mask, so it "
        "cannot run the masked cases; the memory-efficient backend takes them only as a dense "
        "boolean mask. fwd+bwd, ms ([`gpu_attention_h100_device.json`](gpu_attention_h100_device.json), "
        "labels `h100_torch_*`):\n"
    )
    body = []
    for s in SHAPES_TORCH:
        vals = [dev_time(s, i, TORCH_LABELS) for i in
                ("flex:default", "flex:triton", "torch:flex", "torch:flash", "torch:efficient")]
        if all(v is None for v in vals):
            continue
        body.append([shape_name(s), *_bold_min([_t(v) for v in vals], vals)])
    A(table(["B × Lq × Lkv, H, d", "probjax default", "probjax Triton", "torch `flex_attention`",
             "torch SDPA flash", "torch SDPA mem-efficient"], body) + "\n")
    d32 = [dev_time(s, "flex:default", TORCH_LABELS) / dev_time(s, "torch:flex", TORCH_LABELS)
           for s in ("isab_ind_rows", "sq_isab_256x131072", "isab_1k_ind")
           if isnum(dev_time(s, "torch:flex", TORCH_LABELS)) and isnum(dev_time(s, "flex:default", TORCH_LABELS))]
    fl64 = [dev_time(s, "torch:flex", TORCH_LABELS) / dev_time(s, "flex:default", TORCH_LABELS)
            for s in SHAPES_TORCH if shape_meta(s)[4] >= 64
            and isnum(dev_time(s, "torch:flex", TORCH_LABELS))
            and isnum(dev_time(s, "flex:default", TORCH_LABELS))]
    A(
        "With head_dim ≥ 64 the probjax default is the fastest in every case"
        + (f" (`flex_attention` {f_x(min(fl64))}–{f_x(max(fl64))} slower)" if fl64 else "")
        + ". With head_dim 32 "
        "`torch.nn.attention.flex_attention` is "
        f"{f_x(min(d32))}–{f_x(max(d32))} faster than the probjax default (see known issues).\n"
    )

    # ---------------------------------------------------------------- issues
    A("## Known issues on H100\n")
    xla128 = get(GAH, impl="jax_xla", dtype="float32", head_dim=128, causal=False, seq=4096, mode="fwd_bwd")
    pj128 = get(GAH, impl="probjax_default", dtype="float32", head_dim=128, causal=False, seq=4096, mode="fwd_bwd")
    A(
        "- **Shared memory (fixed here).** Large-SMEM GPUs kept the requested tiles unchecked; on "
        "H100 head_dim 256 (forward) and the fp32 head_dim 128 single-pass backward requested "
        "327,680 and 262,144 bytes and failed to compile. Triton buffers streamed tiles once per "
        "pipeline stage on sm_90, which the sm_86 estimates missed; tiles are now shrunk only when "
        "they would not fit.\n"
        "- **Explicit `backward_pass_impl=\"triton_fused\"` on strongly asymmetric lengths (fixed "
        "here).** Matching the tile counts forced tiles below 16 rows, which fell back to Pallas "
        "interpret mode: 25 s instead of ~1 ms at 256 × 16384, and a wrong dQ at 32768 × 256. It "
        "now runs the split backward. `\"auto\"` never selected this configuration.\n"
        "- **Hopper MoE and MLA on unaligned shapes (fixed here).** The Mosaic ragged matmul took "
        "float32 operands and row counts off its 8-row tiling and failed to lower; `mla` tiled a "
        "short non-power-of-two query or latent length (e.g. 48) as one block, which Triton cannot "
        "lower. Float32 now runs the Triton grouped matmul, rows are padded into the last group, "
        "and `mla` pads and masks unaligned lengths like `mha`.\n"
        f"- **fp32 head_dim 128**: `jax.nn` XLA (TF32 matmuls) is about {f_x(pj128 / xla128)} faster "
        f"than the Triton kernel ({f_ms(xla128)} vs {f_ms(pj128)} ms at T=4096); no fp32 rule yet.\n"
        "- **Sequential reference paths** (`implementation=\"scan\"` for Mamba and RLA) are "
        "1.0–1.8× slower than on the RTX 3090 (latency-bound loops); the GPU defaults are 1.3–2.6× "
        "faster. The bf16 DiffusionTransformer train step at 256 tokens is slower too (4.8 vs 3.0 "
        "ms), at the size where launch overhead dominates.\n"
        "- **head_dim 32**: `torch.nn.attention.flex_attention` is faster than every probjax path "
        "on the ISAB shapes above (inducing points over context rows and back); the cuDNN route "
        "and the Triton tiles are not tuned for head_dim 32 yet.\n"
        "- **Tests**: `pytest --gpu` passes on H100. Tests that compare XLA float32 math against "
        "scipy/numpy or an equivalent formulation are marked `float32_matmul` and run in true "
        "float32 on GPU (`tests/conftest.py`; XLA's default is TF32); Pallas kernel-vs-reference "
        "tests, whose kernels use TF32 dots, compare at TF32-level tolerances "
        "(`tests/_tolerance.py`).\n"
    )
    return "\n".join(o)
