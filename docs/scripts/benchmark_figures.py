"""Figures and pages of the GPU / CPU performance reference.

Reads the benchmark JSON in ``docs/benchmarks/`` and writes

* SVG figures to ``docs/benchmarks/figures/``, and
* the pages ``gpu_overview.md``, ``gpu_attention.md``, ``gpu_kernels.md`` and
  ``gpu_nn.md`` in ``docs/benchmarks/``.

Every number, table and figure on those pages comes from a JSON row, so after
re-running a benchmark runner, re-run this script to refresh the pages::

    python -m docs.scripts.benchmark_figures

Rows with an ``error`` key (out of memory, failed autotuning, unsupported
shape) are drawn as a labelled cross at the top of a panel and shown as text in
tables; they are never dropped. Requires matplotlib (not a probjax dependency).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    FuncFormatter,
    LogLocator,
    NullFormatter,
    NullLocator,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "docs" / "benchmarks"
FIG = BENCH / "figures"

# --------------------------------------------------------------------------- data
_CACHE: dict[str, dict] = {}


def load(name: str) -> dict:
    """Load ``docs/benchmarks/<name>.json`` and normalise a few row layouts."""
    if name not in _CACHE:
        data = json.loads((BENCH / f"{name}.json").read_text())
        for r in data.get("results", []):
            if isinstance(r.get("shape"), dict):  # Mamba rows nest their shape
                for k, v in r["shape"].items():
                    r.setdefault(k, v)
            if "routing" in r and "dispatch" in r:  # MoE: one key per code path
                r["path"] = (
                    r["routing"]
                    if r["routing"] in ("capacity", "dense")
                    else (
                        f"{r['impl']}/{r['dispatch']}"
                        if r["impl"] == "pallas"
                        else r["impl"]
                    )
                )
        _CACHE[name] = data
    return _CACHE[name]


def rows(src, **filt) -> list[dict]:
    rs = load(src)["results"] if isinstance(src, str) else src
    out = []
    for r in rs:
        for k, v in filt.items():
            got = r.get(k)
            if callable(v):
                if not v(got):
                    break
            elif got != v:
                break
        else:
            out.append(r)
    return out


def _nums(rs, field):
    out = []
    for r in rs:
        v = r.get(field)
        if isinstance(v, dict):
            v = max(v.values()) if v else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(v)
    return out


def classify(err) -> str:
    e = str(err)
    if "skipped" in e or "unsupported shape" in e:
        return "unsup."
    if "timeout" in e:
        return "timeout"
    if re.search(r"RESOURCE_EXHAUSTED|out of memory|OutOfMemory", e, re.I):
        return "OOM"
    if "NOT_FOUND" in e:
        return "fail†"
    if re.search(
        r"NotImplementedError|does not support|requires an accelerator|unsupported", e
    ):
        return "unsup."
    return "error"


def get(src, field="median_ms", **filt):
    """Median of ``field`` over matching rows, a status string, or None."""
    rs = rows(src, **filt)
    if not rs:
        return None
    vals = _nums(rs, field)
    if vals:
        return statistics.median(vals)
    for key in ("error", "peak_error", "first_failure_error", "skipped"):
        for r in rs:
            if key in r:
                return classify(r[key])
    return None


def num(src, field="median_ms", **filt) -> float:
    v = get(src, field, **filt)
    if not isinstance(v, (int, float)):
        raise KeyError(
            f"no numeric {field} for {filt} in {src if isinstance(src, str) else 'rows'}"
        )
    return v


def isnum(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --------------------------------------------------------------------------- formatting
def f_ms(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def f_mb(v: float) -> str:
    if v == 0:
        return "0"
    if v < 0.0005:
        return "≈0"
    return f_ms(v)


def f_s(v: float) -> str:
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"


def f_err(v: float) -> str:
    return "0" if v == 0 else f"{v:.1e}"


def f_int(v) -> str:
    return f"{int(v):,}"


def f_x(v: float) -> str:
    return f"{v:.2f}×" if v < 10 else f"{v:.0f}×"


def cell(v, fmt=f_ms) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return fmt(v)


def c(src, field="median_ms", fmt=f_ms, **filt) -> str:
    return cell(get(src, field, **filt), fmt)


def q(src, field="median_ms", fmt=f_ms, **filt) -> str:
    """A number for prose; raises when the row is missing."""
    return fmt(num(src, field, **filt))


def kfmt(n) -> str:
    n = int(n)
    if n >= 1048576 and n % 1048576 == 0:
        return f"{n // 1048576}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def table(headers, body, align=None) -> str:
    if align is None:
        align = ["l"] + ["r"] * (len(headers) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in body]
    return "\n".join(lines)


def collapsible(title: str, body: str) -> str:
    pad = "\n".join(("    " + ln) if ln.strip() else ln for ln in body.splitlines())
    return f'??? note "{title}"\n\n{pad}\n'


def mode_label(m: str) -> str:
    return {"fwd": "fwd", "fwd_bwd": "fwd+bwd", "fwdbwd": "fwd+bwd"}.get(m, m)


DT = {"bfloat16": "bf16", "float32": "fp32"}
FOOT_STATUS = (
    "OOM: out of device memory. fail†: XLA found no kernel configuration that "
    "fits during autotuning (`NOT_FOUND`), so the case did not run. unsup.: the "
    "implementation rejects the shape or backend. error: another failure (the JSON "
    "row has the message). —: not measured."
)

# --------------------------------------------------------------------------- styles
INK = "#808080"  # mid-grey: ~4:1 contrast on both a white and a near-black page
GRID = (0.5, 0.5, 0.5, 0.22)
P1, P2, P3 = "#3987e5", "#1c5cab", "#86b6ef"  # probjax: one blue family
ORANGE, AQUA, MAGENTA = "#d95926", "#199e70", "#d55181"
VIOLET, RED, YELLOW, GREEN = "#7f6fe0", "#e34948", "#c98500", "#008300"


def S(label, color, marker="o", ls="-", md=None):
    return {
        "label": label,
        "color": color,
        "marker": marker,
        "ls": ls,
        "md": md or label,
    }


STY = {
    # attention (GPU)
    "probjax_default": S(
        "probjax dot_product_attention (default)",
        P1,
        md="probjax `dot_product_attention` (default)",
    ),
    "probjax_flex": S(
        "probjax flex_attention", P2, "s", "--", md="probjax `flex_attention`"
    ),
    "probjax_mha": S("probjax mha", P3, "^", ":", md="probjax `mha`"),
    "probjax_flax": S(
        "probjax einsum (implementation='xla')",
        P2,
        "D",
        "-.",
        md="probjax einsum (`implementation='xla'`)",
    ),
    "jax_cudnn": S("jax.nn cuDNN", ORANGE, "v", md="`jax.nn` cuDNN"),
    "jax_xla": S("jax.nn XLA", AQUA, "D", md="`jax.nn` XLA"),
    "torch_flash": S("torch SDPA flash", MAGENTA, "P"),
    "torch_efficient": S("torch SDPA mem-efficient", VIOLET, "X"),
    "torch_math": S("torch SDPA math", RED, "*"),
    "torch_sdpa": S("torch SDPA", MAGENTA, "P"),
    # GQA / decoding
    "flex_native": S(
        "probjax flex_attention (native GQA)", P1, md="probjax `flex_attention`"
    ),
    "flex_repeat": S(
        "flex_attention on repeated K/V",
        P3,
        "^",
        ":",
        md="`flex_attention` on repeated K/V",
    ),
    "probjax_splitkv": S(
        "probjax split-KV (default)", P1, md="probjax split-KV (default)"
    ),
    "probjax_nosplit": S(
        "probjax split_kv=False", P2, "s", "--", md="probjax `split_kv=False`"
    ),
    # attention (CPU)
    "cpu:probjax_mha": S(
        "probjax mha (auto strategy)", P1, md="probjax `mha` (auto strategy)"
    ),
    "cpu:xla_tiled": S("probjax tiled XLA strategy", P2, "s", "--"),
    "cpu:xla_dense": S("probjax dense XLA strategy", P3, "^", ":"),
    # SSD
    "ssd:chunked": S("ssd chunked (default)", P1, md="`ssd` chunked (default)"),
    "ssd:pallas": S(
        "ssd Pallas", P2, "s", "--", md="`ssd` Pallas (`implementation='pallas'`)"
    ),
    "ssd:scan": S("ssd_linear_scan", YELLOW, "h", "--", md="`ssd_linear_scan`"),
    # Mamba
    "mamba:pallas": S("Triton scan (GPU default)", P1),
    "mamba:chunked": S("chunked XLA scan (CPU default)", P2, "s", "--"),
    "mamba:assoc": S("associative scan", GREEN, "p", "-."),
    "mamba:scan": S("lax.scan", YELLOW, "h", "--", md="`lax.scan`"),
    # residual linear attention
    "rla:chunked": S("chunked (default)", P1),
    "rla:pallas": S("Pallas fwd + chunked bwd", P2, "s", "--"),
    "rla:pallas_scanbwd": S("Pallas fwd + scan bwd", P3, "^", ":"),
    "rla:scan": S("lax.scan", YELLOW, "h", "--", md="`lax.scan`"),
    # Gated DeltaNet
    "delta:pallas": S("Pallas (GPU default)", P1),
    "delta:chunked": S("chunked (CPU default)", P2, "s", "--"),
    "delta:chunked_fat": S("chunked, more residuals", P3, "^", ":"),
    "delta:pallas_unfused": S("Pallas, unfused", P2, "D", "-."),
    "delta:recurrent": S("recurrence", YELLOW, "h", "--"),
    "delta:ssd": S(
        "ssd on the same shape", VIOLET, "X", "-.", md="`ssd` on the same shape"
    ),
    # kernel_mv / KDE
    "kmv:pallas": S("Pallas (GPU default)", P1),
    "kmv:xla": S("row-blocked XLA (CPU default)", P2, "s", "--"),
    "kmv:pallas_interpret": S("Pallas interpret mode", P3, "^", ":"),
    "kmv:dense": S("dense K @ v", GREEN, "p", "-."),
    "kmv:torch_dense": S("torch dense", MAGENTA, "P"),
    # kernel_logsumexp / OT
    "lse:pallas": S("kernel_logsumexp (Pallas)", P1, md="`kernel_logsumexp` (Pallas)"),
    "lse:kernel": S("kernel_logsumexp (Pallas)", P1, md="`kernel_logsumexp` (Pallas)"),
    "lse:xla": S("kernel_logsumexp (blocked XLA)", P2, "s", "--"),
    "lse:dense": S("dense logsumexp", GREEN, "p", "-."),
    # MoE
    "moe:pallas/sorted": S("Pallas, sorted (GPU default)", P1),
    "moe:pallas/fused_gsum": S("Pallas, fused_gsum", P2, "s", "--"),
    "moe:pallas/fused": S("Pallas, fused", P3, "^", ":"),
    "moe:pallas/gather": S("Pallas, gather", P2, "D", "-."),
    "moe:chunked": S("chunked (CPU default)", P3, "o", "-"),
    "moe:blocked": S("blocked", YELLOW, "h", "--"),
    "moe:ragged_dot": S("jax.lax.ragged_dot", AQUA, "D", md="`jax.lax.ragged_dot`"),
    "moe:capacity": S("capacity routing (drops tokens)", VIOLET, "X"),
    "moe:dense": S("dense routing", GREEN, "p", "-."),
    # nn attention_fn
    "nn:probjax_default": S(
        "default attention_fn", P1, md="default `dot_product_attention`"
    ),
    "nn:probjax_flex": S("flex_attention", P2, "s", "--", md="`flex_attention`"),
    "nn:jax_xla": S("jax.nn XLA", AQUA, "D", md="`jax.nn` XLA"),
    "nn:jax_cudnn": S("jax.nn cuDNN", ORANGE, "v", md="`jax.nn` cuDNN"),
}


def lab(key: str) -> str:
    return STY[key]["md"]


def _setup_mpl():
    plt.rcParams.update({
        "svg.fonttype": "none",
        "svg.hashsalt": "probjax-benchmarks",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "legend.frameon": False,
    })


# --------------------------------------------------------------------------- plotting
def _log_axis(axis, lo, hi):
    """Readable log ticks: 1-2-5 for narrow ranges, decades for wide ones."""
    decades = math.log10(hi / lo) if lo > 0 and hi > 0 else 3
    subs = (
        (1.0, 2.0, 5.0) if decades < 2.2 else ((1.0, 3.0) if decades < 3.5 else (1.0,))
    )
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1e4 else f"{v:g}")
    )
    axis.set_minor_formatter(NullFormatter())
    axis.set_minor_locator(NullLocator())


def _fail_marks(ax, fails, transform):
    """Crosses for cases that did not run, stacked below the top edge of a panel."""
    for x, items in fails.items():
        pos = 0.955
        for k, (key, _) in enumerate(items):
            pos = 0.955 - 0.07 * k
            ax.plot(
                x,
                pos,
                transform=transform,
                marker="x",
                ms=5,
                mew=1.4,
                ls="none",
                color=STY[key]["color"],
                clip_on=False,
                zorder=4,
            )
        text = "/".join(dict.fromkeys(st for _, st in items))
        ax.text(
            x,
            pos - 0.05,
            text,
            transform=transform,
            fontsize=6,
            ha="center",
            va="top",
            color=INK,
        )


def line_panel(
    ax,
    series,
    *,
    title="",
    xlabel="",
    ylabel="",
    xticks=None,
    logx=True,
    logy=True,
    ref=None,
    ref_label=None,
):
    """Lines of ``{x: value | status}`` per implementation key."""
    fails = defaultdict(list)
    xs_all = set()
    for key, data in series:
        st = STY[key]
        pts = sorted((x, v) for x, v in data.items() if isnum(v))
        if logy and any(v <= 0 for _, v in pts):
            raise ValueError(f"non-positive value on a log axis: {key} in {title!r}")
        xs_all |= set(data)
        if pts:
            ax.plot(
                *zip(*pts, strict=False),
                color=st["color"],
                ls=st["ls"],
                marker=st["marker"],
                label=st["label"],
                zorder=3.5 if st["color"] == P1 else 3,
            )
        # Mark where a series stops running; later failures of the same series are implied.
        last_ok = max((x for x, _ in pts), default=None)
        failed = sorted(x for x, v in data.items() if isinstance(v, str))
        shown = [x for x in failed if last_ok is not None and x < last_ok]
        shown += [x for x in failed if last_ok is None or x > last_ok][:1]
        for x in sorted(shown):
            fails[x].append((key, data[x]))
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    if ref is not None:
        ax.axhline(ref, color=INK, lw=0.8, zorder=2, label=ref_label or "_ref")
    ax.relim()
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    if logy:
        _log_axis(ax.yaxis, lo, hi)
    if fails:
        if logy:
            span = math.log10(hi) - math.log10(lo)
            ax.set_ylim(lo, 10 ** (math.log10(lo) + span / 0.7))
        else:
            ax.set_ylim(lo, lo + (hi - lo) / 0.7)
    ticks = xticks or sorted(xs_all)
    pow2 = [t for t in ticks if t >= 1 and int(t) & (int(t) - 1) == 0]
    if not xticks and len(pow2) >= 2 and len(pow2) < len(ticks):
        ticks = pow2
    while len(ticks) > 7:
        ticks = ticks[::-2][::-1]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: kfmt(v) if v >= 1 else f"{v:g}")
    )
    ax.xaxis.set_minor_locator(FixedLocator([]))
    if xs_all:
        ax.set_xlim(
            min(xs_all) / 1.25 if logx else None, max(xs_all) * 1.25 if logx else None
        )
    _fail_marks(ax, fails, ax.get_xaxis_transform())
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major")
    return bool(fails)


def dot_panel(
    ax, cats, series, *, title="", xlabel="", logx=True, ref=None, labels=True
):
    """Dot plot: one row per category, one marker per implementation."""
    n = len(series)
    offs = [0.0] if n == 1 else [(-0.28 + 0.56 * i / (n - 1)) for i in range(n)]
    fails = defaultdict(list)
    for j, cat in enumerate(cats):
        vals = [d.get(cat) for _, d in series]
        vals = [v for v in vals if isnum(v)]
        if len(vals) > 1:
            ax.plot(
                [min(vals), max(vals)],
                [j, j],
                color=GRID,
                lw=2.5,
                zorder=1,
                solid_capstyle="round",
            )
    for i, (key, data) in enumerate(series):
        st = STY[key]
        first = True
        for j, cat in enumerate(cats):
            v = data.get(cat)
            if isnum(v):
                ax.plot(
                    v,
                    j + offs[i],
                    marker=st["marker"],
                    color=st["color"],
                    ms=5.5,
                    ls="none",
                    label=st["label"] if first else None,
                    zorder=3,
                )
                first = False
            elif isinstance(v, str):
                fails[j + offs[i]].append((key, v))
        if first and any(isinstance(data.get(cat), str) for cat in cats):
            ax.plot(
                [],
                [],
                marker=st["marker"],
                color=st["color"],
                ls="none",
                label=st["label"],
            )
    if logx:
        ax.set_xscale("log")
    if ref is not None:
        ax.axvline(ref, color=INK, lw=0.8, zorder=2)
    ax.relim()
    ax.autoscale_view()
    lo, hi = ax.get_xlim()
    if logx:
        if fails:
            span = math.log10(hi) - math.log10(lo)
            ax.set_xlim(lo, 10 ** (math.log10(lo) + span / 0.78))
        _log_axis(ax.xaxis, *ax.get_xlim())
    elif fails:
        ax.set_xlim(lo, lo + (hi - lo) / 0.78)
    for y, items in fails.items():
        for k, (key, _st) in enumerate(items):
            ax.plot(
                0.975 - 0.05 * k,
                y,
                transform=ax.get_yaxis_transform(),
                marker="x",
                ms=5,
                mew=1.4,
                ls="none",
                color=STY[key]["color"],
                clip_on=False,
                zorder=4,
            )
        ax.text(
            0.95 - 0.05 * (len(items) - 1),
            y,
            "/".join(dict.fromkeys(st for _, st in items)),
            transform=ax.get_yaxis_transform(),
            fontsize=6.5,
            ha="right",
            va="center",
            color=INK,
        )
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats if labels else [""] * len(cats))
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    return bool(fails)


def bar_panel(ax, cats, values, *, color=P1, title="", xlabel="", labels=True):
    """Horizontal bars (linear axis) for a single series."""
    for j, v in enumerate(values):
        if isnum(v):
            ax.barh(j, v, height=0.58, color=color, zorder=3)
        elif isinstance(v, str):
            ax.text(0, j, "  " + v, ha="left", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats if labels else [""] * len(cats))
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    return False


FIGURES: list[str] = []


def _round(m: re.Match) -> str:
    v = float(m.group())
    out = f"{v:.2f}" if abs(v) >= 1 else f"{v:.3f}"
    return out.rstrip("0").rstrip(".") if "." in out else out


def _shrink_svg(path: pathlib.Path) -> None:
    """Round coordinates to 1/100 pt; keeps the committed SVGs small."""
    text = path.read_text()
    text = re.sub(r"(?<![#\w])-?\d+\.\d{3,}", _round, text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    path.write_text(text)


def save_figure(
    name,
    panels,
    ncols,
    *,
    width=8.0,
    panel_h=2.15,
    legend=True,
    legend_ncol=None,
    wspace=None,
    left=None,
):
    """Lay out panel callables on a grid, add one shared legend, write the SVG."""
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, panel_h * nrows), squeeze=False
    )
    any_fail = False
    for ax, p in zip(axes.flat, panels, strict=False):
        any_fail |= bool(p(ax))
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    handles, labels = [], []
    for ax in axes.flat:
        for h, lbl in zip(*ax.get_legend_handles_labels(), strict=False):
            if lbl not in labels and not lbl.startswith("_"):
                handles.append(h)
                labels.append(lbl)
    if any_fail:
        handles.append(Line2D([], [], marker="x", ms=5, mew=1.4, color=INK, ls="none"))
        labels.append("did not run (reason next to the cross)")
    top = 1.0
    if legend and handles:
        ncol = legend_ncol or (3 if len(handles) > 4 else len(handles))
        nleg = math.ceil(len(handles) / ncol)
        leg_h = 0.2 * nleg + 0.12
        fig.set_size_inches(width, panel_h * nrows + leg_h)
        top = 1 - leg_h / (panel_h * nrows + leg_h)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=ncol,
            bbox_to_anchor=(0.5, 1.0),
            handlelength=2.6,
            columnspacing=1.4,
            borderaxespad=0.2,
        )
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=1.0, w_pad=1.2)
    if wspace is not None or left is not None:
        fig.subplots_adjust(wspace=wspace, left=left)
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.svg"
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    _shrink_svg(path)
    FIGURES.append(name)
    return f"figures/{name}.svg"


def img(name_path: str, alt: str, caption: str) -> str:
    return f"![{alt}]({name_path})\n\n*{caption}*\n"


# --------------------------------------------------------------------------- shared
GA = "gpu_attention"
ATT_IMPLS = [
    "probjax_default",
    "probjax_flex",
    "probjax_mha",
    "jax_cudnn",
    "jax_xla",
    "torch_flash",
    "torch_efficient",
    "torch_math",
]
ATT_ALTS = [i for i in ATT_IMPLS if not i.startswith("probjax")]


def gpu_name() -> str:
    d = load(GA)["device"]
    return d["gpu"].replace("NVIDIA GeForce ", "")


def att_seqs(src=None, **filt):
    return sorted({r["seq"] for r in rows(src or GA, **filt)})


def att_series(impls, src=None, **filt):
    src = src or GA
    out = []
    for i in impls:
        data = {s: get(src, impl=i, seq=s, **filt) for s in att_seqs(src, **filt)}
        if any(v is not None for v in data.values()):
            out.append((i, data))
    return out


def seq_range(seqs) -> str:
    seqs = sorted(seqs)
    return f"T={seqs[0]}" if len(seqs) == 1 else ", ".join(f"{s}" for s in seqs)


# =========================================================================== attention
def _h100_mosaic() -> bool:
    from docs.scripts.benchmark_figures_h100 import has_mosaic_backend

    return has_mosaic_backend()


def attention_time_figures(src: str = GA, suffix: str = "") -> dict[str, str]:
    """Time (bf16, fp32) and relative-to-default figures of one attention suite.

    ``src`` is the suite's JSON (``gpu_attention`` on the RTX 3090,
    ``gpu_attention_h100`` on the H100); ``suffix`` is appended to the figure
    file names.
    """
    figs = {}
    configs = [(64, False), (64, True), (128, False), (128, True)]
    pow2 = lambda s: s & (s - 1) == 0  # noqa: E731
    for dt in ("bfloat16", "float32"):
        panels = []
        for mode in ("fwd", "fwd_bwd"):
            for d, causal in configs:
                f = dict(dtype=dt, head_dim=d, causal=causal, mode=mode)
                ser = att_series(ATT_IMPLS, src, **f)
                ticks = [s for s in att_seqs(src, **f) if pow2(s)]
                title = f"{mode_label(mode)} · d={d} · {'causal' if causal else 'non-causal'}"
                panels.append(
                    lambda ax, ser=ser, title=title, ticks=ticks, first=(d, causal) == (64, False): (
                        line_panel(
                            ax,
                            ser,
                            title=title,
                            xlabel="sequence length T",
                            ylabel="time (ms)" if first else "",
                            xticks=ticks,
                        )
                    )
                )
        figs[f"time_{dt}"] = save_figure(
            f"attention_time_{DT[dt]}{suffix}", panels, 4, width=9.5, panel_h=2.35
        )
    # relative to the probjax default
    panels = []
    for dt in ("bfloat16", "float32"):
        for d, causal in configs:
            f = dict(dtype=dt, head_dim=d, causal=causal, mode="fwd_bwd")
            base = {
                s: get(src, impl="probjax_default", seq=s, **f) for s in att_seqs(src, **f)
            }
            ser = []
            for i in ATT_IMPLS[1:]:
                data = {}
                for s, b in base.items():
                    v = get(src, impl=i, seq=s, **f)
                    data[s] = v / b if isnum(v) and isnum(b) else v
                if any(v is not None for v in data.values()):
                    ser.append((i, data))
            ticks = [s for s in att_seqs(src, **f) if pow2(s)]
            title = f"{DT[dt]} · d={d} · {'causal' if causal else 'non-causal'}"
            panels.append(
                lambda ax, ser=ser, title=title, ticks=ticks, first=(d, causal) == (64, False): (
                    line_panel(
                        ax,
                        ser,
                        title=title,
                        xlabel="sequence length T",
                        ylabel="time ÷ probjax default" if first else "",
                        xticks=ticks,
                        ref=1.0,
                        ref_label="probjax default (= 1)",
                    )
                )
            )
    figs["relative"] = save_figure(
        f"attention_relative{suffix}", panels, 4, width=9.5, panel_h=2.35
    )
    return figs


def attention_figures() -> dict[str, str]:
    figs = attention_time_figures()

    # device peak
    M = "kernel_attn_gpu_memory"
    impls = [
        "probjax_default",
        "probjax_flex",
        "probjax_mha",
        "probjax_flax",
        "jax_cudnn",
        "jax_xla",
        "torch_flash",
        "torch_efficient",
        "torch_math",
    ]
    seqs = sorted({r["seq"] for r in rows(M, suite="peak")})
    ser = [
        (i, {s: get(M, "peak_mb", suite="peak", impl=i, seq=s) for s in seqs})
        for i in impls
    ]
    ser = [(k, v) for k, v in ser if any(x is not None for x in v.values())]
    # cases above the largest length that ran are out of memory
    for k, data in ser:
        mx = get(M, "max_seq_ok", suite="peak_max_seq", impl=k)
        if isnum(mx):
            for s in seqs:
                if s > mx and data.get(s) is None:
                    data[s] = "OOM"
    figs["memory"] = save_figure(
        "attention_memory",
        [
            lambda ax: line_panel(
                ax,
                ser,
                title="device peak of one fwd+bwd call",
                xlabel="sequence length T",
                ylabel="peak (MB)",
            )
        ],
        1,
        width=8.0,
        panel_h=3.2,
        legend_ncol=3,
    )

    # short sequences
    D = "kernel_attn_gpu_dispatch"
    impls = ["probjax_default", "probjax_mha", "probjax_flax", "jax_xla", "jax_cudnn"]
    panels = []
    for mode in ("fwd", "fwd_bwd"):
        for d in (32, 64):
            ser = []
            for i in impls:
                rs = rows(
                    D, impl=i, dtype="bfloat16", head_dim=d, causal=False, mode=mode
                )
                if any(r.get("tag") == "after" for r in rs):
                    rs = [r for r in rs if r.get("tag") == "after"]
                seqs = sorted({r["seq"] for r in rs})
                ser.append((i, {s: get(rs, seq=s) for s in seqs}))
            panels.append(
                lambda ax, ser=ser, mode=mode, d=d: line_panel(
                    ax,
                    ser,
                    title=f"{mode_label(mode)} · d={d}",
                    xlabel="sequence length T",
                    ylabel="time (ms)" if d == 32 else "",
                )
            )
    figs["short"] = save_figure("attention_short", panels, 2, width=7.5, panel_h=2.3)

    # GQA training shapes
    E = "kernel_attn_ext_gpu"
    impls = ["flex_native", "flex_repeat", "jax_xla", "jax_cudnn", "torch_sdpa"]
    cats, keyed = [], []
    for dt in ("bfloat16", "float32"):
        for t in (1024, 4096):
            for hkv in (1, 8):
                if rows(E, suite="gqa", dtype=dt, lq=t, lkv=t, hkv=hkv):
                    cats.append(
                        f"{DT[dt]} · T={kfmt(t)} · {hkv} KV head{'s' if hkv > 1 else ''}"
                    )
                    keyed.append(dict(dtype=dt, lq=t, lkv=t, hkv=hkv))
    panels = []
    for mode in ("fwd", "fwd_bwd"):
        ser = [
            (
                i,
                {
                    cat: get(
                        E, suite="gqa", impl=i, mode=mode, hq=32, head_dim=128, **f
                    )
                    for cat, f in zip(cats, keyed, strict=False)
                },
            )
            for i in impls
        ]
        ser = [(k, v) for k, v in ser if any(x is not None for x in v.values())]
        panels.append(
            lambda ax, ser=ser, mode=mode: dot_panel(
                ax,
                cats,
                ser,
                title=mode_label(mode),
                xlabel="time (ms)",
                labels=mode == "fwd",
            )
        )
    figs["gqa"] = save_figure("attention_gqa", panels, 2, width=8.5, panel_h=2.9)

    # decoding
    impls = [
        "probjax_splitkv",
        "probjax_nosplit",
        "flex_repeat",
        "jax_xla",
        "jax_cudnn",
        "torch_sdpa",
    ]
    panels = []
    for hkv in (8, 32):
        for b, lq in ((1, 1), (1, 16), (8, 1), (8, 16)):
            f = dict(
                suite="decode",
                dtype="bfloat16",
                head_dim=128,
                hq=32,
                hkv=hkv,
                batch=b,
                lq=lq,
            )
            lkvs = sorted({r["lkv"] for r in rows(E, **f)})
            ser = [(i, {s: get(E, impl=i, lkv=s, **f) for s in lkvs}) for i in impls]
            ser = [(k, v) for k, v in ser if any(x is not None for x in v.values())]
            panels.append(
                lambda ax, ser=ser, hkv=hkv, b=b, lq=lq: line_panel(
                    ax,
                    ser,
                    title=f"{hkv} KV heads · B={b} · {lq} quer{'ies' if lq > 1 else 'y'}",
                    xlabel="KV length",
                    ylabel="time (ms)" if b == 1 and lq == 1 else "",
                )
            )
    figs["decode"] = save_figure("attention_decode", panels, 4, width=9.5, panel_h=2.3)

    # CPU
    C = "kernel_attn-cpu_cpu"
    impls = ["probjax_mha", "xla_tiled", "xla_dense", "jax_xla", "torch_sdpa"]
    panels = []
    for variant in ("plain", "causal"):
        for mode in ("fwd", "fwdbwd"):
            f = dict(variant=variant, mode=mode, head_dim=64, dtype="float32")
            seqs = sorted({r["seq"] for r in rows(C, **f)})
            ser = [
                (
                    "cpu:" + i if i.startswith(("probjax", "xla_")) else i,
                    {s: get(C, impl=i, seq=s, **f) for s in seqs},
                )
                for i in impls
            ]
            panels.append(
                lambda ax, ser=ser, variant=variant, mode=mode: line_panel(
                    ax,
                    ser,
                    title=f"{mode_label(mode)} · {'causal' if variant == 'causal' else 'non-causal'}",
                    xlabel="sequence length T",
                    ylabel="time (ms)" if mode == "fwd" else "",
                )
            )
    figs["cpu"] = save_figure("attention_cpu", panels, 2, width=7.5, panel_h=2.4)

    Mc = load("kernel_attn-cpu_cpu_memory")
    seqs = sorted({
        r["seq"] for r in rows(C, variant="plain", mode="fwdbwd", head_dim=64)
    })
    tser = [
        (
            "cpu:" + i if i.startswith(("probjax", "xla_")) else i,
            {
                s: get(
                    C,
                    "xla_temp_mb",
                    impl=i,
                    variant="plain",
                    mode="fwdbwd",
                    head_dim=64,
                    dtype="float32",
                    seq=s,
                )
                for s in seqs
            },
        )
        for i in ("probjax_mha", "xla_tiled", "xla_dense", "jax_xla")
    ]
    fit = Mc.get("max_fit", [])
    fser = []
    for i in ("xla_tiled", "xla_dense", "jax_xla"):
        data = {p["seq"]: p["total_mb"] for p in fit if p["impl"] == i}
        fser.append(("cpu:" + i if i.startswith("xla_") else i, data))
    ram = fit[0]["host_ram_mb"] if fit else None

    def fit_panel(ax):
        line_panel(
            ax,
            fser,
            title="memory one fwd+bwd needs (compile-time)",
            xlabel="sequence length T",
            ylabel="temp + args + outputs (MB)",
        )
        if ram:
            ax.axhline(ram, color=INK, lw=0.8)
            ax.text(
                ax.get_xlim()[0] * 1.1, ram * 1.25, "host RAM", fontsize=6.5, color=INK
            )
        return False

    figs["cpu_memory"] = save_figure(
        "attention_cpu_memory",
        [
            lambda ax: line_panel(
                ax,
                tser,
                title="XLA temporaries, fwd+bwd",
                xlabel="sequence length T",
                ylabel="MB",
            ),
            fit_panel,
        ],
        2,
        width=7.5,
        panel_h=2.5,
    )
    return figs


def attn_grid(dt, d, causal, mode, seqs):
    body = [
        [lab(i)]
        + [
            c(GA, impl=i, dtype=dt, head_dim=d, causal=causal, mode=mode, seq=s)
            for s in seqs
        ]
        for i in ATT_IMPLS
    ]
    return table(["Implementation"] + [f"T={s}" for s in seqs], body)


def attn_title(dt, d, causal, mode):
    return (
        f"{DT[dt]}, head_dim {d}, {'causal' if causal else 'non-causal'}, "
        f"{'forward + backward' if mode == 'fwd_bwd' else 'forward'}"
    )


def attn_faster_alternatives(margin=0.05):
    """Bullets for configurations where an alternative beats the probjax default."""
    out = []
    pow2 = lambda s: s & (s - 1) == 0  # noqa: E731
    for mode in ("fwd_bwd", "fwd"):
        for dt in ("bfloat16", "float32"):
            for d in (64, 128):
                for causal in (False, True):
                    f = dict(dtype=dt, head_dim=d, causal=causal, mode=mode)
                    seqs = [s for s in att_seqs(**f) if pow2(s) and s >= 512]
                    parts = []
                    worst = None
                    for a in ATT_ALTS:
                        faster = []
                        for s in seqs:
                            b = get(GA, impl="probjax_default", seq=s, **f)
                            v = get(GA, impl=a, seq=s, **f)
                            if isnum(b) and isnum(v) and v < (1 - margin) * b:
                                faster.append(s)
                                if worst is None or b / v > worst[0]:
                                    worst = (b / v, a, s)
                        if faster:
                            parts.append(
                                f"{lab(a)} at T = {', '.join(map(str, faster))}"
                            )
                    if parts:
                        out.append(
                            f"- **{attn_title(dt, d, causal, mode)}**: "
                            + "; ".join(parts)
                            + f". Largest gap: {lab(worst[1])} is {f_x(worst[0])} faster at "
                            f"T={worst[2]}."
                        )
    return out


def first_fail(impl, mode="fwd_bwd"):
    """Smallest T at which ``impl`` did not run in any configuration of the grid."""
    bad = [r["seq"] for r in rows(GA, impl=impl, mode=mode) if "error" in r]
    return min(bad) if bad else "—"


def attn_t4096_table(seq=4096):
    cfgs = [
        (dt, d, cz)
        for dt in ("bfloat16", "float32")
        for d in (64, 128)
        for cz in (False, True)
    ]
    body = [
        [lab(i)]
        + [
            c(GA, impl=i, dtype=dt, head_dim=d, causal=cz, mode="fwd_bwd", seq=seq)
            for dt, d, cz in cfgs
        ]
        for i in ATT_IMPLS
    ]
    return table(
        ["Implementation"]
        + [f"{DT[dt]} d{d}{' causal' if cz else ''}" for dt, d, cz in cfgs],
        body,
    )


def attn_short_t():
    """Smallest key count from which the kernel beats the einsum in every swept case."""
    D = "kernel_attn_gpu_dispatch"
    res = []
    for d in (32, 64):
        rs_m = [
            r
            for r in rows(
                D,
                impl="probjax_mha",
                dtype="bfloat16",
                head_dim=d,
                causal=False,
                mode="fwd_bwd",
            )
        ]
        rs_e = rows(
            D,
            impl="probjax_flax",
            dtype="bfloat16",
            head_dim=d,
            causal=False,
            mode="fwd_bwd",
        )
        seqs = sorted({r["seq"] for r in rs_m})
        win = [
            isnum(get(rs_m, seq=s))
            and isnum(get(rs_e, seq=s))
            and get(rs_m, seq=s) < 0.95 * get(rs_e, seq=s)
            for s in seqs
        ]
        start = None
        for k in range(
            len(seqs) - 1, -1, -1
        ):  # first length from which the kernel always wins
            if not win[k]:
                break
            start = seqs[k]
        res.append(start)
    return res


def page_attention(figs) -> str:
    o = []
    A = o.append
    dev = load(GA)["device"]
    cfg = load(GA)["config"]
    A("# Attention performance\n")
    A(
        "probjax's attention kernels (`mha`, `flex_attention` and the default "
        "`dot_product_attention`, which every probjax transformer uses as `attention_fn`) "
        "compared with `jax.nn.dot_product_attention` (XLA and cuDNN backends) and PyTorch "
        "`scaled_dot_product_attention` pinned to its flash, memory-efficient and math "
        f"backends. GPU: {dev['gpu']} ({dev['gpu_memory_gb']:.0f} GB), JAX {dev['jax']}, "
        f"torch {dev['torch']}. Setup and methodology are on the [overview](gpu_overview.md).\n"
    )
    A(
        f"Inputs are `[batch, seq, heads, head_dim]` with batch {cfg['batch']} and "
        f"{cfg['heads']} heads unless stated otherwise; torch receives the transposed layout it "
        "expects. fwd is one forward call; fwd+bwd is the forward plus the gradient with "
        "respect to queries, keys and values. Times are medians in milliseconds. "
        + FOOT_STATUS
        + " cuDNN and torch flash have no fp32 path.\n"
    )

    A("## When to use which\n")
    from docs.scripts.benchmark_figures_h100 import (
        cudnn_speedup_range,
        has_mosaic_backend,
    )

    sp = cudnn_speedup_range()
    A(
        "- **On an H100 (sm_90)** the defaults are device-specific: large bf16/fp16 calls with "
        "no mask, a `KVLenMask` or equal-length causal masking run cuDNN"
        + (f" ({sp[0]:.1f}–{sp[1]:.1f}× faster fwd+bwd there)" if sp else "")
        + (" or, where cuDNN cannot take them, the Mosaic GPU kernel;" if has_mosaic_backend() else ",")
        + " and `backward_pass_impl=\"auto\"` resolves to the split backward. Rules and measurements: "
        f"[H100 page](gpu_h100.md#device-specific-defaults). The rest of this page is the {gpu_name()}.\n"
    )
    M = "kernel_attn_gpu_memory"
    swept = max(r["seq"] for r in rows(M, suite="peak"))
    pj_max = get(M, "max_seq_ok", suite="peak_max_seq", impl="probjax_default")
    dense_max = get(M, "max_seq_ok", suite="peak_max_seq", impl="jax_xla")
    bullets = attn_faster_alternatives()
    cud = any("cuDNN" in b for b in bullets)
    A(
        "- **Default (`dot_product_attention`, `implementation=None`).** On NVIDIA GPUs it "
        "runs the fused `flex_attention` kernel whenever that is exact (rules "
        "[below](#short-sequences-and-the-default-dispatch)); `flex_attention` and `mha` "
        "are the same kernel. Its memory grows linearly in T: fwd+bwd ran up to "
        f"T = {f_int(pj_max)} (the largest length tried{'' if pj_max == swept else ' was ' + f_int(swept)}) "
        f"at batch 4 and 8 heads, dense attention only up to T = {f_int(dense_max)}.\n"
        + (
            "- **`jax.nn.dot_product_attention(implementation='cudnn')`** is faster in the "
            "bf16 configurations listed under "
            "[where an alternative is faster](#where-an-alternative-is-faster); use it there "
            "if you need neither probjax masks nor fp32.\n"
            if cud
            else ""
        )
        + "- **Dense attention** (`jax.nn` XLA, torch math, probjax `implementation='xla'`) "
        "materializes the T × T scores. In this grid `jax.nn` XLA fails from "
        f"T = {first_fail('jax_xla')} and torch math from T = {first_fail('torch_math')} "
        "(fwd+bwd); it is only competitive for short sequences.\n"
        "- **Backward pass.** `backward_pass_impl='auto'` picks the kernel. On GPUs with "
        "about 100 KB of shared memory per block (RTX 30/40 class) it uses the single-pass "
        "backward (`'triton_onepass'`: dK/dV per KV block, dQ accumulated with atomic adds, "
        "as in FlashAttention-2) for head_dim 32 and 64 and for bf16 head_dim 128 at the "
        "lengths where it is faster, otherwise the two-pass kernels (`'triton_fused'`, "
        "`'triton_split'`). The single-pass dQ is summed in a nondeterministic order, so "
        "gradients are not bitwise reproducible across runs; select a two-pass kernel if "
        "you need that. It also keeps a float32 dQ buffer, which adds about a quarter to "
        "the backward's temporaries for 16-bit inputs.\n"
        "- **Decoding with a KV cache** (few queries, long cache): the default selects the "
        "split-KV kernel automatically; see [decoding](#grouped-query-attention-and-decoding).\n"
        "- **CPU**: probjax runs an XLA formulation whose memory is linear in T; torch's "
        "native CPU kernel is faster ([CPU](#cpu)).\n"
    )

    A("## Time\n")
    A(
        img(
            figs["time_bfloat16"],
            "Attention time vs sequence length, bf16",
            f"bf16, batch {cfg['batch']}, {cfg['heads']} heads, {gpu_name()}. Top row forward, "
            "bottom row forward + backward; T = 600, 1500 and 3000 test lengths that are not a "
            "multiple of the tile size.",
        )
    )
    A(
        img(
            figs["time_float32"],
            "Attention time vs sequence length, fp32",
            f"fp32, batch {cfg['batch']}, {cfg['heads']} heads, {gpu_name()}. fp32 matmuls run "
            "in TF32 (JAX default precision).",
        )
    )
    A(
        img(
            figs["relative"],
            "Attention time relative to the probjax default",
            "fwd+bwd time of each implementation divided by the probjax default at the same "
            "shape; below the line is faster than probjax.",
        )
    )

    A("### Where an alternative is faster\n")
    A(
        "Configurations (T ≥ 512) in which another implementation is more than 5% faster "
        "than the probjax default, computed from "
        "[`gpu_attention.json`](gpu_attention.json):\n"
    )
    A("\n".join(bullets) + "\n" if bullets else "None.\n")
    A("fwd+bwd at T=4096 in every configuration (ms):\n")
    A(attn_t4096_table() + "\n")
    full = []
    for dt in ("bfloat16", "float32"):
        for d in (64, 128):
            for causal in (False, True):
                for m in ("fwd", "fwd_bwd"):
                    seqs = att_seqs(dtype=dt, head_dim=d, causal=causal, mode=m)
                    full.append(
                        f"**{attn_title(dt, d, causal, m)}**\n\n"
                        + attn_grid(dt, d, causal, m, seqs)
                        + "\n"
                    )
    A(collapsible("All 16 configurations as tables", "\n".join(full)))

    A("## Memory\n")
    M = "kernel_attn_gpu_memory"
    Mr = rows(M, suite="peak")[0]
    A(
        img(
            figs["memory"],
            "Attention device peak vs sequence length",
            f"True device peak of one fwd+bwd call, {DT[Mr['dtype']]}, head_dim {Mr['head_dim']}, "
            f"batch {Mr['batch']}, {Mr['heads']} heads, non-causal, read in a fresh process, "
            f"{gpu_name()}.",
        )
    )
    body = []
    for i in (
        "probjax_default",
        "probjax_flax",
        "jax_xla",
        "jax_cudnn",
        "torch_flash",
        "torch_efficient",
        "torch_math",
    ):
        body.append([lab(i), c(M, "max_seq_ok", f_int, suite="peak_max_seq", impl=i)])
    A(
        "Largest power-of-two T whose fwd+bwd ran (the search stopped at "
        f"{f_int(max(r['seq'] for r in rows(M, suite='peak')))}, so that value means "
        "*at least*):\n"
    )
    A(table(["Implementation", "largest T"], body) + "\n")
    R = "kernel_attn_gpu_residuals"
    rr = rows(R)[0]
    cols = [
        ("bfloat16", 1000),
        ("bfloat16", 2048),
        ("float32", 1000),
        ("float32", 2048),
    ]
    body = [
        [lab(i)]
        + [c(R, "residual_mb", f_mb, impl=i, dtype=dt, seq=s) for dt, s in cols]
        for i in (
            "probjax_default",
            "probjax_mha",
            "probjax_flax",
            "jax_xla",
            "jax_cudnn",
        )
    ]
    A(
        f"Bytes kept for the backward pass (`jax.vjp` residuals, MB), batch {rr['batch']}, "
        f"{rr['heads']} heads, head_dim {rr['head_dim']} "
        "([`kernel_attn_gpu_residuals.json`](kernel_attn_gpu_residuals.json)). The kernels "
        "keep the output and the log-sum-exp; dense paths keep the T × T probabilities:\n"
    )
    A(table(["Implementation"] + [f"{DT[dt]} T={s}" for dt, s in cols], body) + "\n")

    A("## Short sequences and the default dispatch\n")
    A(
        "`probjax.nn.layers.attention.dot_product_attention` is the default `attention_fn` of "
        "`Transformer`, `DiffusionTransformer` and the attention conditioners. With "
        "`implementation=None` it runs the fused kernel (`flex_attention`) on NVIDIA GPUs "
        "when that is exact:\n\n"
        "- no mask, or a position-only mask (causal, length or window masks); no dense "
        "boolean mask,\n"
        "- no additive bias, no dropout, default `precision`,\n"
        "- at least 64 keys.\n\n"
        "Otherwise, and on CPU, it uses the Flax einsum. `implementation='xla'` forces the "
        "einsum and `implementation='pallas'` the kernel. GQA inputs (fewer KV heads than "
        "query heads) also go to the kernel.\n"
    )
    D = rows("kernel_attn_gpu_dispatch")[0]
    t32, t64 = attn_short_t()
    A(
        img(
            figs["short"],
            "Attention time for short sequences",
            f"bf16, batch {D['batch']}, {D['heads']} heads, non-causal, {gpu_name()} "
            "([`kernel_attn_gpu_dispatch.json`](kernel_attn_gpu_dispatch.json)).",
        )
    )
    smallest = min(r["seq"] for r in rows("kernel_attn_gpu_dispatch"))

    def where(t):
        if t is None:
            return "at no swept length"
        return "at every swept length" if t == smallest else f"from T = {t} on"

    A(
        "In this sweep the `mha` kernel is more than 5% faster than the einsum in fwd+bwd "
        + (
            f"{where(t32)} for head_dim 32 and 64. "
            if t32 == t64
            else f"{where(t32)} for head_dim 32 and {where(t64)} for head_dim 64. "
        )
        + "The default switches "
        "to the kernel from 64 keys on. cuDNN pays a fixed launch cost at short lengths.\n"
    )

    A("### Tile selection\n")
    T = rows("gpu_attention_tiles")
    body = []
    for dt in ("bfloat16", "float32"):
        for d in (64, 128):
            for s in (2048, 8192):
                for m in ("fwd", "fwd_bwd"):
                    auto = rows(T, dtype=dt, head_dim=d, seq=s, mode=m, config="auto")
                    forced = rows(
                        T, dtype=dt, head_dim=d, seq=s, mode=m, config="forced"
                    )
                    ok = [r for r in forced if "median_ms" in r]
                    if not auto or not ok:
                        continue
                    best = min(ok, key=lambda r: r["median_ms"])
                    if m == "fwd":
                        cfgs = (
                            f"q{best['block_q']}/k{best['block_k']}, "
                            f"{best['num_stages']} st, {best['num_warps']} w"
                        )
                    else:
                        cfgs = (
                            f"fwd q{best['fwd_block_q']}/k{best['fwd_block_k']}, bwd "
                            f"{best['bwd_outer']}/{best['bwd_inner']}, "
                            f"{best['num_stages']} st, {best['num_warps']} w"
                        )
                    body.append([
                        DT[dt],
                        d,
                        s,
                        mode_label(m),
                        cell(get(auto)),
                        f_ms(best["median_ms"]),
                        cfgs,
                        f"{len(ok)} / {len(forced) - len(ok)}",
                    ])
    A(
        "With default block sizes, `mha` picks tiles from a shared-memory model and a tuned "
        "table per (dtype, head_dim). Auto choice against the best of every forced "
        "(block, stages, warps) combination, batch 4, 8 heads "
        "([`gpu_attention_tiles.json`](gpu_attention_tiles.json)); the last column counts "
        "configurations that ran and those Triton rejected for exceeding shared memory:\n"
    )
    A(
        table(
            [
                "dtype",
                "head_dim",
                "T",
                "mode",
                "auto (ms)",
                "best forced (ms)",
                "best forced config",
                "ran / rejected",
            ],
            body,
        )
        + "\n"
    )

    A("## Grouped-query attention and decoding\n")
    E = "kernel_attn_ext_gpu"
    A(
        "The kernels read KV head `h // G` directly instead of repeating K and V in memory "
        "(`flex_attention` and `dot_product_attention` route GQA inputs to them).\n"
    )
    A(
        img(
            figs["gqa"],
            "GQA training-shaped attention",
            f"32 query heads, head_dim 128, batch 1, Lq = Lkv = T, median of three rounds, "
            f"{gpu_name()} ([`kernel_attn_ext_gpu.json`](kernel_attn_ext_gpu.json), suite `gqa`).",
        )
    )
    body = []
    for dt in ("bfloat16", "float32"):
        for t in (1024, 4096):
            for hkv in (1, 8):
                f = dict(
                    suite="gqa", dtype=dt, lq=t, lkv=t, hkv=hkv, hq=32, head_dim=128
                )
                if not rows(E, **f):
                    continue
                body.append(
                    [DT[dt], t, hkv]
                    + [
                        c(E, "xla_temp_mb", f_mb, impl=i, mode="fwd", **f)
                        for i in ("flex_native", "flex_repeat")
                    ]
                    + [
                        c(E, "residual_mb", f_mb, impl=i, mode="fwd_bwd", **f)
                        for i in ("flex_native", "flex_repeat")
                    ]
                )
    A("Memory of native GQA against running the kernel on repeated K/V (MB):\n")
    A(
        table(
            [
                "dtype",
                "T",
                "KV heads",
                "fwd temp, native",
                "fwd temp, repeated K/V",
                "bwd residuals, native",
                "bwd residuals, repeated K/V",
            ],
            body,
        )
        + "\n"
    )
    A(
        "**Split-KV decoding.** For few queries against a long KV cache the regular kernel "
        "parallelizes only over batch × heads × query tiles. The split-KV forward splits "
        "the keys across programs, packs the query heads of a GQA group into one tile and "
        "combines the partial results with their log-sum-exp. It is selected automatically "
        "for unmasked inputs with Lq ≤ 64, Lkv ≥ 1024 and at least two splits; "
        "`split_kv=False` disables it. Under `jax.grad` the regular kernel runs.\n"
    )
    A(
        img(
            figs["decode"],
            "Decoding time vs KV length",
            f"bf16, 32 query heads, head_dim 128, median of three rounds, {gpu_name()}. Top "
            "row GQA with 8 KV heads, bottom row 32 KV heads (plain multi-head; batch 8 at 64k "
            "keys was not run).",
        )
    )
    impls = [
        "probjax_splitkv",
        "probjax_nosplit",
        "flex_repeat",
        "jax_xla",
        "jax_cudnn",
        "torch_sdpa",
    ]

    def dec_table(dtype, hkv, lqs, head_dim):
        body = []
        for b in (1, 8):
            for lq in lqs:
                for lkv in (4096, 16384, 65536):
                    f = dict(
                        suite="decode",
                        dtype=dtype,
                        batch=b,
                        lq=lq,
                        lkv=lkv,
                        hq=32,
                        hkv=hkv,
                        head_dim=head_dim,
                    )
                    if rows(E, **f):
                        body.append(
                            [b, lq, kfmt(lkv)] + [c(E, impl=i, **f) for i in impls]
                        )
        return table(["B", "Lq", "Lkv"] + [lab(i) for i in impls], body)

    A(
        collapsible(
            "Decoding tables (bf16 Lq = 1, 4, 16; fp32 head_dim 64)",
            "bf16, head_dim 128, 8 KV heads:\n\n"
            + dec_table("bfloat16", 8, (1, 4, 16), 128)
            + "\n\nbf16, head_dim 128, 32 KV heads:\n\n"
            + dec_table("bfloat16", 32, (1, 16), 128)
            + "\n\nfp32, head_dim 64, 8 KV heads:\n\n"
            + dec_table("float32", 8, (1, 4, 16), 64),
        )
    )
    body = []
    for lq in (1, 16):
        for i in impls:
            m = rows(E, suite="peak", lq=lq, lkv=65536, batch=8, impl=i)
            if m:
                body.append([
                    lq,
                    lab(i),
                    c(m, "peak_mb", f_mb),
                    c(m, "inputs_mb", f_mb),
                ])
    A("Device peak at batch 8, 64k keys, 8 KV heads, bf16 (fresh process):\n")
    A(
        table(
            ["Lq", "Implementation", "peak (MB)", "inputs incl. KV cache (MB)"],
            body,
            ["l", "l", "r", "r"],
        )
        + "\n"
    )

    A("## CPU\n")
    C = "kernel_attn-cpu_cpu"
    A(
        "On CPU, `mha`, `mla` and `flex_attention` run an XLA formulation: dense for "
        "T ≤ 256, a single-tile flash `custom_vjp` up to 512, and 256 × 256 tiles with "
        "block-mask tile skipping above that; the backward recomputes from the log-sum-exp. "
        "Masks, biases (with gradients), dropout, GQA without repeating K/V and "
        "forward-mode AD are supported. The default `dot_product_attention` uses the Flax "
        "einsum on CPU.\n"
    )
    A(
        img(
            figs["cpu"],
            "Attention time on CPU",
            "fp32, batch 1, 8 heads, head_dim 64, shared 24-thread host (10–30% run-to-run "
            "noise) ([`kernel_attn-cpu_cpu.json`](kernel_attn-cpu_cpu.json)).",
        )
    )
    A(
        img(
            figs["cpu_memory"],
            "Attention memory on CPU",
            "Left: XLA temporaries of the non-causal fwd+bwd. Right: compile-time memory "
            "(temporaries + arguments + outputs) one fwd+bwd needs, against the host's RAM "
            "([`kernel_attn-cpu_cpu_memory.json`](kernel_attn-cpu_cpu_memory.json)).",
        )
    )
    fwd = dict(variant="plain", mode="fwdbwd", head_dim=64, dtype="float32", seq=4096)
    A(
        "At T=4096 (non-causal fwd+bwd) the probjax default takes "
        f"{q(C, impl='probjax_mha', **fwd)} ms, `jax.nn` XLA {q(C, impl='jax_xla', **fwd)} ms "
        f"and torch {q(C, impl='torch_sdpa', **fwd)} ms; torch's native CPU kernel is the "
        "fastest option on CPU.\n"
    )
    Ec = "kernel_attn_ext_cpu"
    body = []
    for lq in (1, 16):
        for lkv in (4096, 16384, 65536):
            f = dict(suite="decode", lq=lq, lkv=lkv, hkv=8, tag="after")
            body.append(
                [lq, kfmt(lkv)]
                + [
                    c(Ec, impl=i, **f)
                    for i in ("flex_native", "flex_repeat", "jax_xla", "torch_sdpa")
                ]
            )
    A(
        "CPU decoding, fp32, 32 query heads, 8 KV heads, head_dim 128, batch 1 "
        "([`kernel_attn_ext_cpu.json`](kernel_attn_ext_cpu.json)), ms:\n"
    )
    A(
        table(
            ["Lq", "Lkv"]
            + [lab(i) for i in ("flex_native", "flex_repeat", "jax_xla", "torch_sdpa")],
            body,
        )
        + "\n"
    )

    A("## How to reproduce\n")
    A(
        "```bash\n"
        "# Cross-library grid and the tile sweep (CUDA GPU; torch optional).\n"
        "python -m docs.scripts.benchmark_gpu_attention --output docs/benchmarks/gpu_attention.json\n"
        "python -m docs.scripts.benchmark_gpu_attention_tiles --output docs/benchmarks/gpu_attention_tiles.json\n"
        "\n"
        "# Short-sequence sweep, device peak and residuals.\n"
        "python -m docs.scripts.benchmark_kernel_attn_gpu --device gpu --suite dispatch \\\n"
        "  --output docs/benchmarks/kernel_attn_gpu_dispatch.json\n"
        "python -m docs.scripts.benchmark_kernel_attn_gpu --device gpu --suite peak \\\n"
        "  --output docs/benchmarks/kernel_attn_gpu_memory.json\n"
        "\n"
        "# GQA and split-KV decoding.\n"
        "python -m docs.scripts.benchmark_kernel_attn_ext --device gpu --suite gqa \\\n"
        "  --output docs/benchmarks/kernel_attn_ext_gpu.json\n"
        "python -m docs.scripts.benchmark_kernel_attn_ext --device gpu --suite decode \\\n"
        "  --output docs/benchmarks/kernel_attn_ext_gpu.json --append\n"
        "\n"
        "# CPU.\n"
        "JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_attn-cpu \\\n"
        "  --device cpu --output docs/benchmarks/kernel_attn-cpu_cpu.json\n"
        "\n"
        "# Rebuild the figures and this page from the JSON.\n"
        "python -m docs.scripts.benchmark_figures\n"
        "```\n"
    )
    return "\n".join(o)


# =========================================================================== kernels
def fam_series(src, fam, impls, xkey, field="median_ms", xs=None, **filt):
    out = []
    for i in impls:
        rs = rows(src, impl=i, **filt)
        keys = xs or sorted({r[xkey] for r in rs if xkey in r})
        data = {x: get(rs, field, **{xkey: x}) for x in keys}
        if any(v is not None for v in data.values()):
            out.append((f"{fam}:{i}", data))
    return out


def lp(ser, title, xlabel, ylabel=""):
    return lambda ax: line_panel(ax, ser, title=title, xlabel=xlabel, ylabel=ylabel)


def kernel_figures() -> dict[str, str]:
    figs = {}
    # SSD
    G, C = "kernel_ssd_gpu", "kernel_ssd_cpu"
    f = dict(config="g2h8_d128", kind=None)
    panels = []
    for src, dev, impls in (
        (G, "GPU", ("chunked", "pallas", "scan")),
        (C, "CPU", ("chunked", "pallas", "scan")),
    ):
        panels += [
            lp(
                fam_series(src, "ssd", impls, "seq", mode="fwd", **f),
                f"{dev} · fwd",
                "sequence length",
                "time (ms)",
            ),
            lp(
                fam_series(src, "ssd", impls, "seq", mode="fwd_bwd", **f),
                f"{dev} · fwd+bwd",
                "sequence length",
            ),
            lp(
                fam_series(
                    src, "ssd", impls, "seq", "xla_temp_mb", mode="fwd_bwd", **f
                ),
                f"{dev} · fwd+bwd XLA temporaries",
                "sequence length",
                "MB",
            ),
        ]
    figs["ssd"] = save_figure("kernel_ssd", panels, 3, width=8.5)

    # Mamba
    G, C = "kernel_mamba_gpu", "kernel_mamba_cpu"
    f = dict(dim=512, kind=None)
    impls = ("pallas", "chunked", "assoc", "scan")
    panels = []
    for src, dev in ((G, "GPU"), (C, "CPU")):
        panels += [
            lp(
                fam_series(src, "mamba", impls, "seq", mode="fwd", **f),
                f"{dev} · fwd",
                "sequence length",
                "time (ms)",
            ),
            lp(
                fam_series(src, "mamba", impls, "seq", mode="fwd_bwd", **f),
                f"{dev} · fwd+bwd",
                "sequence length",
            ),
            lp(
                fam_series(
                    src, "mamba", impls, "seq", "xla_temp_mb", mode="fwd_bwd", **f
                ),
                f"{dev} · fwd+bwd XLA temporaries",
                "sequence length",
                "MB",
            ),
        ]
    figs["mamba"] = save_figure("kernel_mamba", panels, 3, width=8.5)

    # residual linear attention
    G, C = "kernel_rla_gpu", "kernel_rla_cpu"
    impls = ("chunked", "pallas", "pallas_scanbwd", "scan")
    k = dict(suite="kernel", dtype="float32", dim=64)
    panels = [
        lp(
            fam_series(G, "rla", impls, "seq", mode="fwd", **k),
            "GPU · fwd",
            "sequence length",
            "time (ms)",
        ),
        lp(
            fam_series(G, "rla", impls, "seq", mode="fwd_bwd", **k),
            "GPU · fwd+bwd",
            "sequence length",
        ),
        lp(
            fam_series(G, "rla", impls, "seq", "peak_mb", suite="peak", mode="fwd_bwd"),
            "GPU · fwd+bwd device peak (B4 H8)",
            "sequence length",
            "MB",
        ),
        lp(
            fam_series(C, "rla", impls, "seq", mode="fwd", **k),
            "CPU · fwd",
            "sequence length",
            "time (ms)",
        ),
        lp(
            fam_series(C, "rla", impls, "seq", mode="fwd_bwd", **k),
            "CPU · fwd+bwd",
            "sequence length",
        ),
        lp(
            fam_series(C, "rla", impls, "seq", "xla_temp_mb", mode="fwd_bwd", **k),
            "CPU · fwd+bwd XLA temporaries",
            "sequence length",
            "MB",
        ),
    ]
    figs["rla"] = save_figure("kernel_rla", panels, 3, width=8.5)

    # Gated DeltaNet
    G = "kernel_delta_gpu"
    impls = ("pallas", "chunked", "chunked_fat", "recurrent", "ssd")
    panels = []
    for d in (64, 128):
        f = dict(dtype="bfloat16", dk=d)
        panels += [
            lp(
                fam_series(G, "delta", impls, "seq", mode="fwd", **f),
                f"d={d} · fwd",
                "sequence length",
                "time (ms)",
            ),
            lp(
                fam_series(G, "delta", impls, "seq", mode="fwd_bwd", **f),
                f"d={d} · fwd+bwd",
                "sequence length",
            ),
            lp(
                fam_series(
                    G, "delta", impls, "seq", "residual_mb", mode="fwd_bwd", **f
                ),
                f"d={d} · bwd residuals",
                "sequence length",
                "MB",
            ),
        ]
    figs["delta"] = save_figure("kernel_delta", panels, 3, width=8.5)

    # kernel_mv
    G, C = "kernel_kmv_gpu", "kernel_kmv_cpu"
    f = dict(family="kernel_mv", dim=8, out=16)
    gi = ("pallas", "xla", "dense", "torch_dense")
    ci = ("xla", "pallas_interpret", "dense", "torch_dense")
    panels = [
        lp(
            fam_series(G, "kmv", gi, "n", mode="fwd", batch=1, **f),
            "GPU · fwd",
            "N = M",
            "time (ms)",
        ),
        lp(
            fam_series(G, "kmv", gi, "n", mode="fwd_bwd", batch=1, **f),
            "GPU · fwd+bwd",
            "N = M",
        ),
        lp(
            fam_series(
                G,
                "kmv",
                ("pallas", "xla", "dense"),
                "n",
                "peak_mb",
                family="kernel_mv_peak",
                mode="fwd_bwd",
            ),
            "GPU · fwd+bwd device peak",
            "N = M",
            "MB",
        ),
        lp(
            fam_series(
                G,
                "kmv",
                ("pallas", "xla", "dense"),
                "n",
                "run_s",
                family="kernel_mv_max_n",
                mode="fwd_bwd",
            ),
            "GPU · one fwd+bwd, large N",
            "N = M",
            "wall time (s)",
        ),
        lp(
            fam_series(C, "kmv", ci, "n", mode="fwd", **f),
            "CPU · fwd",
            "N = M",
            "time (ms)",
        ),
        lp(
            fam_series(C, "kmv", ci, "n", mode="fwd_bwd", **f), "CPU · fwd+bwd", "N = M"
        ),
    ]
    figs["kmv"] = save_figure("kernel_kmv", panels, 3, width=8.5)

    # kernel_logsumexp
    G, C = "kernel_ot_gpu", "kernel_ot_cpu"

    def lse_scale():
        out = []
        for i, key in (("pallas", "kernel"), ("dense", "dense")):
            data = {
                r["n"]: get(G, suite="lse", impl=i, d=2, n=r["n"], mode="fwd_bwd")
                for r in rows(G, suite="lse", impl=i, d=2, mode="fwd_bwd")
            }
            for r in rows(G, suite="scale", impl=key, d=2, mode="fwd_bwd"):
                data[r["n"]] = get(
                    G, suite="scale", impl=key, d=2, n=r["n"], mode="fwd_bwd"
                )
            out.append((f"lse:{i}", data))
        return out

    panels = [
        lp(
            fam_series(
                G, "lse", ("pallas", "dense"), "n", suite="lse", d=8, mode="fwd"
            ),
            "GPU · fwd · D=8",
            "N = M",
            "time (ms)",
        ),
        lp(
            fam_series(
                G, "lse", ("pallas", "dense"), "n", suite="lse", d=8, mode="fwd_bwd"
            ),
            "GPU · fwd+bwd · D=8",
            "N = M",
        ),
        lp(lse_scale(), "GPU · fwd+bwd · D=2, large N", "N = M"),
        lp(
            fam_series(C, "lse", ("xla", "dense"), "n", suite="lse", d=8, mode="fwd"),
            "CPU · fwd · D=8",
            "N = M",
            "time (ms)",
        ),
        lp(
            fam_series(
                C, "lse", ("xla", "dense"), "n", suite="lse", d=8, mode="fwd_bwd"
            ),
            "CPU · fwd+bwd · D=8",
            "N = M",
        ),
        lp(
            fam_series(
                G,
                "lse",
                ("pallas", "dense"),
                "n",
                "xla_temp_mb",
                suite="lse",
                d=8,
                mode="fwd_bwd",
            ),
            "GPU · fwd+bwd XLA temporaries · D=8",
            "N = M",
            "MB",
        ),
    ]
    figs["lse"] = save_figure("kernel_logsumexp", panels, 3, width=8.5)

    # MoE
    G, C = "kernel_gmm_gpu", "kernel_gmm_cpu"
    paths = [
        "pallas/sorted",
        "pallas/fused_gsum",
        "pallas/fused",
        "pallas/gather",
        "blocked",
        "ragged_dot",
        "capacity",
        "dense",
    ]
    cats = []
    keyed = []
    for run, dt, lbl in (
        ("f_gpu_a", "float32", "8 experts top-2 · fp32"),
        ("f_gpu_a", "bfloat16", "8 experts top-2 · bf16"),
        ("f_gpu_c", "bfloat16", "64 experts top-4 · bf16"),
    ):
        if rows(G, run=run, dtype=dt, tokens=32768):
            cats.append(lbl)
            keyed.append(dict(run=run, dtype=dt, tokens=32768, suite="layer"))
    panels = []
    for op in ("forward", "train_step"):
        ser = [
            (
                f"moe:{p}",
                {
                    cat: get(G, path=p, op=op, **kf)
                    for cat, kf in zip(cats, keyed, strict=False)
                },
            )
            for p in paths
        ]
        ser = [(k, v) for k, v in ser if any(x is not None for x in v.values())]
        panels.append(
            lambda ax, ser=ser, op=op: dot_panel(
                ax,
                cats,
                ser,
                title=op.replace("_", " "),
                xlabel="time (ms)",
                labels=op == "forward",
            )
        )
    figs["moe"] = save_figure("kernel_moe", panels, 2, width=8.5, panel_h=2.2)

    mser = []
    for p in ("pallas/sorted", "pallas/fused_gsum", "blocked", "ragged_dot", "dense"):
        rs = rows(G, suite="maxtok", code="final", path=p)
        data = {r["tokens"]: get(rs, "peak_mb", tokens=r["tokens"]) for r in rs}
        if data:
            mser.append((f"moe:{p}", data))
    cser = []
    for p in ("chunked", "blocked", "ragged_dot", "capacity", "dense"):
        rs = rows(
            C,
            suite="layer",
            path=p,
            op="train_step",
            dtype="float32",
            experts=8,
            top_k=2,
            skew=lambda s: s in (None, 0.0),
        )
        data = {r["tokens"]: get(rs, tokens=r["tokens"]) for r in rs}
        if data:
            cser.append((f"moe:{p}", data))
    figs["moe_scale"] = save_figure(
        "kernel_moe_scale",
        [
            lp(mser, "GPU · bf16 train step device peak", "tokens", "peak (MB)"),
            lp(cser, "CPU · fp32 train step", "tokens", "time (ms)"),
        ],
        2,
        width=8.0,
        panel_h=2.5,
    )
    return figs


def page_kernels(figs) -> str:
    o = []
    A = o.append
    gpu = gpu_name()
    A("# Sequence-mixer, kernel-sum and MoE performance\n")
    A(
        "probjax's non-softmax kernels against the alternatives in the library and outside "
        "it, on GPU and CPU: SSD, the Mamba selective scan, residual linear attention, "
        "Gated DeltaNet, kernel matrix-vector products and KDE, log-domain kernel "
        f"reductions (Sinkhorn, OT), and the MoE grouped matmul. GPU: {gpu}; CPU: shared "
        "24-thread host, whose timings carry 10–30% run-to-run noise. Attention is on its "
        "[own page](gpu_attention.md), setup and methodology on the "
        "[overview](gpu_overview.md). fwd+bwd is the forward plus the gradient with respect "
        "to all array inputs. Times are medians in milliseconds. " + FOOT_STATUS + "\n"
    )

    # ---------------------------------------------------------------- SSD
    G, C = "kernel_ssd_gpu", "kernel_ssd_cpu"
    A("## SSD (Mamba-2 state-space duality)\n")
    A(
        "`ssd(q, k, v, log_alpha)` computes the Mamba-2 / SSD recurrence. "
        "`implementation=None` runs a chunked pure-JAX form (the Mamba-2 chunked algorithm "
        "with `remat='full'`) on CPU and GPU; `implementation='pallas'` selects the Triton "
        "kernel, and `ssd_linear_scan` is the per-step reference. `SSDCell` and "
        "`Mamba2Cell` are built on `ssd` ([models page](gpu_nn.md#ssmmodel-cells)).\n"
    )
    A(
        img(
            figs["ssd"],
            "SSD time and memory",
            "Batch 4, 2 groups, 8 heads, d_k = d_v = 128, fp32 "
            "([`kernel_ssd_gpu.json`](kernel_ssd_gpu.json), "
            "[`kernel_ssd_cpu.json`](kernel_ssd_cpu.json)).",
        )
    )
    body = []
    for cfg in ("g2h8_d128", "g1h1_d128"):
        for impl in ("chunked", "pallas", "scan"):
            body.append(
                [cfg, lab(f"ssd:{impl}")]
                + [
                    c(
                        G,
                        "peak_mb",
                        f_mb,
                        kind="peak",
                        config=cfg,
                        impl=impl,
                        mode=m,
                        seq=8192,
                    )
                    for m in ("fwd", "fwd_bwd")
                ]
            )
    A("Device peak at sequence length 8192 (fresh process, MB):\n")
    A(
        table(
            ["config", "implementation", "fwd", "fwd+bwd"], body, ["l", "l", "r", "r"]
        )
        + "\n"
    )
    body = [
        [
            lab(f"ssd:{r['impl']}"),
            f_int(r["largest_ok"]["seq"]),
            f_mb(r["largest_ok"]["peak_mb"]),
        ]
        for r in rows(G, kind="max_len")
    ]
    A(
        "Longest power-of-two sequence whose fwd+bwd ran on the 24 GB card (g2h8_d128):\n"
    )
    A(table(["implementation", "longest sequence", "peak there (MB)"], body) + "\n")
    other = []
    for src, dev in ((G, "GPU"), (C, "CPU")):
        body = []
        for cfg in ("g1h24_dk128_dv64", "g1h1_d128", "g2h4_dk48_dv40"):
            for s in (512, 2048, 8192):
                for m in ("fwd", "fwd_bwd"):
                    if rows(src, config=cfg, seq=s, mode=m, kind=None):
                        body.append(
                            [cfg, s, mode_label(m)]
                            + [
                                c(src, impl=i, config=cfg, seq=s, mode=m, kind=None)
                                for i in ("chunked", "pallas", "scan")
                            ]
                        )
        other.append(
            f"**{dev}** ([`{src}.json`]({src}.json))\n\n"
            + table(
                ["config", "seq", "mode"]
                + [lab(f"ssd:{i}") for i in ("chunked", "pallas", "scan")],
                body,
                ["l", "r", "l", "r", "r", "r"],
            )
        )
    A(
        collapsible(
            "Other head configurations (24 heads with d_v 64, one head, d_k 48 / d_v 40)",
            "\n\n".join(other),
        )
    )
    g = dict(config="g2h8_d128", mode="fwd_bwd", seq=8192)
    A(
        "**Accuracy**, relative (`max|a − b| / max|b|`) against an fp32 `HIGHEST`-precision "
        "run of the chunked algorithm: the chunked default's gradients differ by at most "
        f"{q(G, 'rel_err_grads', f_err, impl='chunked', **g)} at sequence length 8192; the "
        "Pallas kernel runs its matmuls in TF32, with output error "
        f"{q(G, 'rel_err', f_err, impl='pallas', config='g2h8_d128', mode='fwd', seq=8192)} "
        f"and gradient error up to {q(G, 'rel_err_grads', f_err, impl='pallas', **g)}.\n"
    )
    A(
        "**When to use which.** Keep the default. On GPU the chunked form is at least as "
        "fast as the Triton kernel at the shapes above, keeps no residuals and runs any "
        "shape; it is the only fast path on CPU. `ssd_linear_scan` is a reference: its "
        "backward stores every state and runs out of memory at long sequences. `ssd()` "
        "forwards `chunk_size`, `remat`, `precision` and `return_final_state` to the "
        "chunked path.\n"
    )

    # ---------------------------------------------------------------- Mamba
    G, C = "kernel_mamba_gpu", "kernel_mamba_cpu"
    A("## Mamba selective scan\n")
    A(
        "`compute_mamba_scan` is the selective scan of `MambaCell`. On GPU it runs Triton "
        "sequence-loop kernels that keep the state in registers and recompute instead of "
        "storing states in the backward pass; on CPU, and on GPU for unsupported shapes, a "
        "chunked XLA scan with a custom VJP. The associative scan and `lax.scan` are the "
        "generic JAX alternatives.\n"
    )
    A(
        img(
            figs["mamba"],
            "Mamba scan time and memory",
            "Batch 4, D 512, state 16, fp32; on CPU the associative scan was not run at 8192 "
            "([`kernel_mamba_gpu.json`](kernel_mamba_gpu.json), "
            "[`kernel_mamba_cpu.json`](kernel_mamba_cpu.json)).",
        )
    )
    impls = ("pallas", "chunked", "assoc", "scan")
    body = [
        [
            lab(f"mamba:{i}"),
            c(G, "max_seq", f_int, kind="max_len", impl=i),
            c(G, "peak_mb_at_max", f_mb, kind="max_len", impl=i),
        ]
        for i in impls
    ]
    lim = rows(G, kind="max_len")[0].get("limit")
    A(
        f"Longest power-of-two sequence whose fwd+bwd ran on GPU (batch 4, D 1024, state 16; "
        f"search limit {f_int(lim)}):\n"
    )
    A(table(["implementation", "longest sequence", "peak there (MB)"], body) + "\n")
    body = []
    for seq in (2048, 8192):
        for m in ("fwd", "fwd_bwd"):
            body.append(
                [seq, mode_label(m)]
                + [
                    c(
                        G,
                        "peak_mb",
                        f_mb,
                        kind="device_peak",
                        impl=i,
                        mode=m,
                        seq=seq,
                        dim=512,
                    )
                    for i in impls
                ]
            )
    A("GPU device peak, batch 4, D 512 (fresh process, MB):\n")
    A(table(["seq", "mode"] + [lab(f"mamba:{i}") for i in impls], body) + "\n")
    S = rows("kernel_mamba_gpu_state")
    body = []
    for key in sorted({(r["batch"], r["seq"], r["dim"], r["state"]) for r in S}):
        b, s, d, st = key
        body.append(
            [b, s, d, st]
            + [
                cell(
                    next(
                        (
                            r["after_ms"]
                            for r in S
                            if (r["batch"], r["seq"], r["dim"], r["state"]) == key
                            and r["mode"] == m
                        ),
                        None,
                    )
                )
                for m in ("fwd", "bwd")
            ]
        )
    A(
        "The figure uses state 16; `MambaCell` defaults to `state_dim = model_dim`. GPU "
        "time of the Triton scan at larger states, median of three rounds "
        "([`kernel_mamba_gpu_state.json`](kernel_mamba_gpu_state.json)), ms:\n"
    )
    A(table(["B", "L", "D", "state", "fwd", "bwd"], body) + "\n")
    more = []
    for src, dev, imps in ((G, "GPU", impls), (C, "CPU", ("chunked", "assoc", "scan"))):
        body = []
        for d in (256, 1024):
            for s in (512, 2048, 8192):
                for m in ("fwd", "fwd_bwd"):
                    if rows(src, dim=d, seq=s, mode=m, kind=None):
                        body.append(
                            [d, s, mode_label(m)]
                            + [
                                c(src, impl=i, dim=d, seq=s, mode=m, kind=None)
                                for i in imps
                            ]
                        )
        more.append(
            f"**{dev}**\n\n"
            + table(["D", "seq", "mode"] + [lab(f"mamba:{i}") for i in imps], body)
        )
    A(collapsible("D 256 and D 1024", "\n\n".join(more)))
    A(
        "**Accuracy.** The fp32 implementations agree with `lax.scan` to "
        f"{q(G, 'rel_err_vs_scan', f_err, impl='pallas', mode='fwd_bwd', seq=8192, dim=512, kind=None)} "
        "relative error at sequence length 8192 (fwd+bwd, D 512).\n"
    )
    A(
        "**When to use which.** Keep the default dispatch. On GPU the Triton scan is the "
        "fastest measured implementation at every shape. On CPU the chunked scan is the "
        "fastest for fwd+bwd; for a forward-only pass `lax.scan` is within host noise of it "
        "and sometimes faster at D 512. The associative scan materializes "
        "`[B, L, D, S]` and `lax.scan` stores every state for the backward pass; both run "
        "out of memory well before the defaults.\n"
    )

    # ---------------------------------------------------------------- RLA
    G, C = "kernel_rla_gpu", "kernel_rla_cpu"
    A("## Residual linear attention (RAttention)\n")
    A(
        "`residual_linear_attention`, the linear-attention branch of `RAttention`, defaults "
        "to a chunk-parallel pure-JAX form (`implementation='auto'`) on every backend. "
        "`implementation='pallas'` runs a Pallas forward with the chunked backward, "
        "`implementation='scan'` the token-by-token recurrence.\n"
    )
    A(
        img(
            figs["rla"],
            "Residual linear attention time and memory",
            "Batch 2, 4 heads, D 64, chunk 128, fp32; device peak at batch 4, 8 heads "
            "([`kernel_rla_gpu.json`](kernel_rla_gpu.json), "
            "[`kernel_rla_cpu.json`](kernel_rla_cpu.json)).",
        )
    )
    body = []
    for s in (512, 2048, 8192):
        for m in ("fwd", "fwd_bwd"):
            f = dict(suite="layer", seq=s, mode=m)
            body.append(
                [s, mode_label(m)]
                + [c(G, impl=i, **f) for i in ("chunked", "pallas", "scan")]
                + [c(C, impl=i, **f) for i in ("chunked", "scan")]
            )
    lr = rows(G, suite="layer")[0]
    A(
        f"**`RAttention` layer** end to end (forward, and gradient of a scalar loss), batch "
        f"{lr['batch']}, {lr['heads']} heads, head_dim {lr['dim']}, window {lr['window']}. "
        "The layer runs out of memory at 8192 tokens on the 24 GB GPU because its "
        "sliding-window branch is dense:\n"
    )
    A(
        table(
            [
                "seq",
                "mode",
                "GPU chunked",
                "GPU Pallas",
                "GPU scan",
                "CPU chunked",
                "CPU scan",
            ],
            body,
        )
        + "\n"
    )
    body = []
    for d in (32, 64, 128):
        for s in (512, 2048, 8192):
            for m in ("fwd", "fwd_bwd"):
                f = dict(suite="kernel", dtype="bfloat16", dim=d, seq=s, mode=m)
                if rows(G, **f):
                    body.append(
                        [d, s, mode_label(m)]
                        + [
                            c(G, impl=i, **f)
                            for i in ("chunked", "pallas", "pallas_scanbwd", "scan")
                        ]
                    )
    A(
        collapsible(
            "GPU bf16 kernel timings",
            table(
                ["D", "seq", "mode"]
                + [
                    lab(f"rla:{i}")
                    for i in ("chunked", "pallas", "pallas_scanbwd", "scan")
                ],
                body,
            ),
        )
    )
    g = dict(suite="kernel", dtype="float32", dim=64, seq=8192)
    A(
        "**Accuracy**, relative to the fp32 `lax.scan`: chunked "
        f"{q(G, 'rel_err', f_err, impl='chunked', mode='fwd', **g)} (forward) and "
        f"{q(G, 'rel_err', f_err, impl='chunked', mode='fwd_bwd', **g)} (fwd+bwd, max over "
        "outputs and gradients) at sequence length 8192, D 64; the Pallas forward "
        f"{q(G, 'rel_err', f_err, impl='pallas', mode='fwd', **g)} (TF32).\n"
    )
    A(
        "**When to use which.** Keep `implementation='auto'`: the chunked form is the "
        "fastest measured option on both backends and its backward does not store "
        "per-token state the way the scan backward does.\n"
    )

    # ---------------------------------------------------------------- Delta
    G, C = "kernel_delta_gpu", "kernel_delta_cpu"
    A("## Gated DeltaNet\n")
    A(
        "`gated_delta_rule(implementation='auto')` runs `gated_delta_rule_pallas` on GPU for "
        "supported chunk and head sizes, and the chunked pure-JAX form "
        "(`gated_delta_rule_chunked`, WY/UT representation with a hand-written VJP that "
        "keeps only chunk-boundary states) otherwise and on CPU. "
        "`gated_delta_rule_recurrent` is the token-by-token reference. `ssd` on the same "
        "shape is plotted as a cost reference; it has no delta-rule correction term. "
        "`GatedDeltaNetCell` is built on it.\n"
    )
    A(
        img(
            figs["delta"],
            "Gated DeltaNet time and residual memory",
            f"bf16, batch 4, 8 heads, chunk 64, d_k = d_v = d, {gpu} "
            "([`kernel_delta_gpu.json`](kernel_delta_gpu.json)).",
        )
    )
    body = []
    for d in (64, 128):
        for s in (512, 2048, 8192):
            for m in ("fwd", "fwd_bwd"):
                f = dict(dtype="float32", dk=d, seq=s, mode=m)
                body.append(
                    [d, s, mode_label(m)]
                    + [
                        c(G, impl=i, **f)
                        for i in (
                            "pallas",
                            "pallas_unfused",
                            "chunked",
                            "chunked_fat",
                            "recurrent",
                            "ssd",
                        )
                    ]
                )
    A(
        collapsible(
            "GPU fp32 timings",
            table(
                ["d", "seq", "mode"]
                + [
                    lab(f"delta:{i}")
                    for i in (
                        "pallas",
                        "pallas_unfused",
                        "chunked",
                        "chunked_fat",
                        "recurrent",
                        "ssd",
                    )
                ],
                body,
            ),
        )
    )
    body = []
    for d in (64, 128):
        for s in (512, 2048, 8192):
            for m in ("fwd", "fwd_bwd"):
                f = dict(dk=d, seq=s, mode=m)
                if any(isnum(get(C, impl=i, **f)) for i in ("chunked", "recurrent")):
                    body.append(
                        [d, s, mode_label(m)]
                        + [
                            c(C, impl=i, **f)
                            for i in ("chunked", "chunked_fat", "recurrent", "ssd")
                        ]
                    )
    A(
        "**CPU**, fp32, batch 2, 8 heads ([`kernel_delta_cpu.json`](kernel_delta_cpu.json); "
        "the recurrence was not run at 8192), ms:\n"
    )
    A(
        table(
            ["d", "seq", "mode"]
            + [
                lab(f"delta:{i}")
                for i in ("chunked", "chunked_fat", "recurrent", "ssd")
            ],
            body,
        )
        + "\n"
    )
    g = dict(dtype="bfloat16", dk=128, seq=8192, mode="fwd_bwd")
    A(
        "**Accuracy**, relative to an fp32 `HIGHEST`-precision chunked reference: bf16 Pallas "
        f"{q(G, 'rel_err', f_err, impl='pallas', **g)} (fwd+bwd, d 128, sequence length "
        f"8192), fp32 Pallas "
        f"{q(G, 'rel_err', f_err, impl='pallas', dtype='float32', dk=128, seq=8192, mode='fwd_bwd')}.\n"
    )
    A(
        "**When to use which.** Use the default. The recurrence stores every state in its "
        "backward pass and fails at long sequences. Compared with `ssd` on the same shape, "
        "Gated DeltaNet is cheaper at d 128 and more expensive at d 64 on this GPU; choose "
        "between them for the model, not the kernel.\n"
    )

    # ---------------------------------------------------------------- kmv
    G, C = "kernel_kmv_gpu", "kernel_kmv_cpu"
    A("## RBF kernel matrix-vector products and KDE\n")
    A(
        "`rbf_kernel_mv` computes `K(q, k) @ v` without materializing K; `kde_density` and "
        "`rbf_kde_density` build on it. The default is the Pallas kernel on GPU and a "
        "row-blocked XLA implementation on CPU (`implementation='xla'` on either).\n"
    )
    A(
        img(
            figs["kmv"],
            "Kernel matrix-vector product time and memory",
            "fp32, D 8, 16 output columns, batch 1; GPU device peak in a fresh process "
            "([`kernel_kmv_gpu.json`](kernel_kmv_gpu.json), "
            "[`kernel_kmv_cpu.json`](kernel_kmv_cpu.json)).",
        )
    )
    body = []
    for d in (2, 8):
        for n in sorted({r["n"] for r in rows(G, family="kde_density", dim=d)}):
            for m in ("fwd", "fwd_bwd"):
                f = dict(family="kde_density", dim=d, n=n, mode=m)
                body.append(
                    [d, f_int(n), mode_label(m)]
                    + [c(G, impl=i, **f) for i in ("pallas", "xla", "dense")]
                )
    A("GPU `kde_density`, ms:\n")
    A(
        table(
            ["D", "N = M", "mode", lab("kmv:pallas"), lab("kmv:xla"), lab("kmv:dense")],
            body,
        )
        + "\n"
    )
    more = []
    for src, dev, imps in (
        (G, "GPU", ("pallas", "xla", "dense", "torch_dense")),
        (C, "CPU", ("xla", "pallas_interpret", "dense", "torch_dense")),
    ):
        body = []
        for d, out in ((2, 1), (8, 1), (8, 16), (32, 1), (32, 16)):
            for n in sorted({
                r["n"] for r in rows(src, family="kernel_mv", dim=d, out=out)
            }):
                for m in ("fwd", "fwd_bwd"):
                    f = dict(family="kernel_mv", dim=d, out=out, n=n, mode=m)
                    body.append(
                        [d, out, f_int(n), mode_label(m)]
                        + [c(src, impl=i, **f) for i in imps]
                    )
        more.append(
            f"**{dev}**\n\n"
            + table(
                ["D", "outputs", "N = M", "mode"] + [lab(f"kmv:{i}") for i in imps],
                body,
            )
        )
    A(collapsible("`rbf_kernel_mv` at every measured shape", "\n\n".join(more)))
    A(
        "**Accuracy.** Each JSON row carries `rel_err`, the norm-relative error against the "
        "row-blocked XLA implementation at `HIGHEST` matmul precision; the Pallas kernel is "
        "within "
        f"{q(G, 'rel_err', f_err, family='kernel_mv', impl='pallas', dim=8, out=16, n=16384, mode='fwd_bwd', batch=1)} "
        "(fwd+bwd, D 8, N = 16384, largest gradient error).\n"
    )
    A(
        "**When to use which.** Keep the defaults. At N = 1024 the dense product and the XLA "
        "path are faster than the Pallas kernel in fwd+bwd; from a few thousand points on "
        "the kernel is the fastest, and the dense product runs out of memory around "
        "N = 32k on the 24 GB GPU. The Pallas kernel ran fwd+bwd at N = "
        f"{f_int(max(r['n'] for r in rows(G, family='kernel_mv_max_n', impl='pallas') if 'run_s' in r))} "
        "with a "
        f"{q(G, 'peak_mb', f_mb, family='kernel_mv_max_n', impl='pallas', n=1048576)} MB "
        "device peak.\n"
    )

    # ---------------------------------------------------------------- OT
    G, C = "kernel_ot_gpu", "kernel_ot_cpu"
    A("## Log-domain kernel reductions, Sinkhorn and OT\n")
    A(
        "`kernel_logsumexp` computes `logsumexp_j(-c(x_i, y_j) / ε + g_j)` without "
        "materializing the N × M cost: a Pallas kernel on GPU (cross term in 3×TF32), "
        "blocked XLA on CPU, dense for small problems. `probjax.nn.ot` builds on it "
        "(Sinkhorn with implicit or unrolled gradients, Sinkhorn divergence, entropic map, "
        "plan sampler, MMD, KDE log-density, `ot_pairing`); `FlowMatcher` / `LinearFlow` "
        "accept `ot_coupling=None | 'exact' | 'sinkhorn'`. The comparison is a dense "
        "`jax.nn.logsumexp` on the materialized cost (ott-jax is not installed).\n"
    )
    A(
        img(
            figs["lse"],
            "kernel_logsumexp time and memory",
            "fp32, N = M ([`kernel_ot_gpu.json`](kernel_ot_gpu.json), "
            "[`kernel_ot_cpu.json`](kernel_ot_cpu.json)).",
        )
    )
    body = []
    for src, dev, ki in ((G, "GPU", "pallas"), (C, "CPU", "xla")):
        for d in (2, 32):
            for n in sorted({r["n"] for r in rows(src, suite="lse", d=d)}):
                f = dict(suite="lse", d=d, n=n)
                body.append(
                    [dev, d, f_int(n)]
                    + [
                        c(src, impl=i, mode=m, **f)
                        for m in ("fwd", "fwd_bwd")
                        for i in (ki, "dense")
                    ]
                )
    A(
        collapsible(
            "D 2 and D 32",
            table(
                [
                    "device",
                    "D",
                    "N = M",
                    "fwd kernel",
                    "fwd dense",
                    "fwd+bwd kernel",
                    "fwd+bwd dense",
                ],
                body,
            ),
        )
    )
    body = [
        [
            lab("lse:" + r["impl"]),
            f_int(r["n"]),
            r["d"],
            mode_label(r["mode"]),
            cell(r.get("peak_mb"), f_mb)
            if "peak_mb" in r
            else classify(r.get("error")),
        ]
        for r in rows(G, suite="peak")
    ]
    A("GPU device peak (fresh process):\n")
    A(
        table(
            ["implementation", "N = M", "D", "mode", "peak (MB)"],
            body,
            ["l", "r", "r", "l", "r"],
        )
        + "\n"
    )
    body = []
    for src, dev in ((G, "GPU"), (C, "CPU")):
        for n in sorted({r["n"] for r in rows(src, suite="sinkhorn")}):
            f = dict(suite="sinkhorn", d=8, n=n)
            body.append([
                dev,
                f_int(n),
                c(src, impl="kernel", mode="fwd", **f),
                c(src, impl="dense", mode="fwd", **f),
                c(src, impl="kernel", mode="fwd_bwd_implicit", **f),
                c(src, "xla_temp_mb", f_mb, impl="kernel", mode="fwd", **f),
                c(src, "xla_temp_mb", f_mb, impl="dense", mode="fwd", **f),
                c(src, "abs_diff", f_err, impl="agreement", **f),
            ])
    A(
        "**Sinkhorn**, 50 log-domain iterations, D 8, against a dense log-domain solver on "
        "the materialized cost; the last column is the absolute difference of the two "
        "transport costs:\n"
    )
    A(
        table(
            [
                "device",
                "N = M",
                "kernel (ms)",
                "dense (ms)",
                "kernel + implicit grad (ms)",
                "kernel temp (MB)",
                "dense temp (MB)",
                "cost difference",
            ],
            body,
        )
        + "\n"
    )
    body = []
    for src, dev in ((G, "GPU"), (C, "CPU")):
        for r in rows(src, suite="accuracy"):
            body.append([
                dev,
                r["impl"],
                r["d"],
                f"{r['eps']:.3g}",
                f_err(r["lse_max_abs_err"]),
                f_err(r["grad_rel_err"]) if "grad_rel_err" in r else "—",
            ])
    A(
        "**Accuracy** against an fp64 reference, N = M = 2048. `dense_default_precision` is "
        "the dense fp32 computation at JAX's default (TF32) matmul precision, which is why "
        "the kernel computes the cross term in 3×TF32:\n"
    )
    A(
        table(
            [
                "device",
                "implementation",
                "D",
                "ε",
                "max abs err (logsumexp)",
                "rel err (gradient)",
            ],
            body,
            ["l", "l", "r", "r", "r", "r"],
        )
        + "\n"
    )
    cf = rows(G, suite="cfm")
    body = [
        [
            r["coupling"],
            r["seed"],
            f_ms(r["ms_per_step"]),
            f"{r['final_loss']:.3g}",
            f"{r['straightness']:.3g}",
            f"{r['sdiv_nfe1']:.3g}",
            f"{r['sdiv_nfe2']:.3g}",
            f"{r['sdiv_nfe4']:.3g}",
            f"{r['sdiv_nfe32']:.3g}",
            f"{r['sdiv_floor']:.3g}",
        ]
        for r in cf
    ]
    A(
        f"**OT-CFM** (`LinearFlow` from N(0, I) to 8 Gaussians, {cf[0]['steps']} steps, batch "
        f"{cf[0]['batch']}, GPU). Straightness is lower for straighter paths; the Sinkhorn "
        "divergence to held-out target samples after 1, 2, 4 and 32 Euler steps measures "
        "sample quality (the floor is the divergence between two target samples):\n"
    )
    A(
        table(
            [
                "coupling",
                "seed",
                "ms / step",
                "final loss",
                "straightness",
                "S-div 1 step",
                "2 steps",
                "4 steps",
                "32 steps",
                "floor",
            ],
            body,
            ["l"] + ["r"] * 9,
        )
        + "\n"
    )
    A(
        "**When to use which.** On GPU the kernel matches or beats the dense forward from a "
        "few thousand points on and is the only option once the cost matrix no longer fits "
        "(dense fails from N = 32k–49k in the sweeps above). Its backward costs several "
        "times its forward, so for fwd+bwd at large D the dense form is faster while it fits "
        "(D 32, N = 16384: dense "
        f"{q(G, suite='lse', impl='dense', d=32, n=16384, mode='fwd_bwd')} ms vs kernel "
        f"{q(G, suite='lse', impl='pallas', d=32, n=16384, mode='fwd_bwd')} ms, with "
        f"{q(G, 'xla_temp_mb', f_mb, suite='lse', impl='dense', d=32, n=16384, mode='fwd_bwd')} "
        "vs "
        f"{q(G, 'xla_temp_mb', f_mb, suite='lse', impl='pallas', d=32, n=16384, mode='fwd_bwd')} "
        "MB of temporaries). For OT-CFM, Sinkhorn pairing costs little more per step than "
        "independent pairing and approaches the straightness of exact OT.\n"
    )

    # ---------------------------------------------------------------- MoE
    G, C = "kernel_gmm_gpu", "kernel_gmm_cpu"
    A("## MoE grouped matmul\n")
    A(
        "`MoELayer(routing='dropless')` sorts token-expert assignments by expert and runs "
        "one grouped matmul. On GPU the default is the Pallas grouped GEMM "
        "(`dropless_impl='auto'`, `dropless_dispatch='sorted'`), which processes the grouped "
        "SwiGLU in row slabs so the train step holds no full-size intermediates; on CPU it "
        "is `dropless_impl='chunked'`, an XLA loop over expert-sorted row chunks. "
        "`routing='capacity'` drops tokens beyond the expert capacity and is not exact; "
        "`routing='dense'` runs every expert on every token.\n"
    )
    A(
        img(
            figs["moe"],
            "MoE layer time by code path",
            "32768 tokens, d_model 512; 8 experts with d_hidden 1024, 64 experts with d_hidden "
            f"256; {gpu} ([`kernel_gmm_gpu.json`](kernel_gmm_gpu.json)).",
        )
    )
    A(
        img(
            figs["moe_scale"],
            "MoE train-step memory and CPU time",
            "Left: GPU device peak of the bf16 train step (8 experts top-2, fresh process). "
            "Right: CPU fp32 train step, 8 experts top-2 "
            "([`kernel_gmm_cpu.json`](kernel_gmm_cpu.json)).",
        )
    )
    body = []
    for r in rows(G, suite="maxtok_summary", code="final"):
        body.append([lab(f"moe:{r['path']}"), f_int(r["max_tokens"])])
    A("Largest bf16 train step that ran on the 24 GB GPU:\n")
    A(table(["code path", "max tokens"], body) + "\n")
    A(
        "**Accuracy**, relative output error against an fp32 dense-routing reference: "
        f"{q(G, 'rel_err_out', f_err, run='f_gpu_a', path='pallas/sorted', dtype='float32', tokens=32768, op='forward')} "
        "for fp32 Pallas on GPU (TF32 matmuls), "
        f"{q(G, 'rel_err_out', f_err, run='f_gpu_a', path='pallas/sorted', dtype='bfloat16', tokens=32768, op='forward')} "
        "in bf16, "
        f"{q(C, 'rel_err_out', f_err, run='cpu_A', path='chunked', tokens=4096, op='forward', dtype='float32')} "
        "for the CPU chunked path.\n"
    )
    A(
        "**When to use which.** Keep the defaults. On GPU with 8 experts the sorted Pallas "
        "dispatch has the fastest exact train step and fits the most tokens, "
        "`dropless_dispatch='fused_gsum'` the fastest forward; with many small experts "
        "(64, top-4) `fused_gsum` also wins the train step. `jax.lax.ragged_dot` is far "
        "slower on this GPU. On CPU `chunked` is the fastest exact path.\n"
    )

    A("## How to reproduce\n")
    A(
        "```bash\n"
        "# Per-family runners (CUDA GPU); add JAX_PLATFORMS=cpu and --device cpu for the CPU files.\n"
        "python -m docs.scripts.benchmark_kernel_ssd --device gpu --output docs/benchmarks/kernel_ssd_gpu.json\n"
        "python -m docs.scripts.benchmark_kernel_mamba --device gpu --output docs/benchmarks/kernel_mamba_gpu.json\n"
        "python -m docs.scripts.benchmark_kernel_rla --device gpu --suites kernel layer \\\n"
        "  --output docs/benchmarks/kernel_rla_gpu.json\n"
        "python -m docs.scripts.benchmark_kernel_delta --device gpu --output docs/benchmarks/kernel_delta_gpu.json\n"
        "python -m docs.scripts.benchmark_kernel_kmv --device gpu --output docs/benchmarks/kernel_kmv_gpu.json\n"
        "python -m docs.scripts.benchmark_kernel_ot --device gpu --suite lse\n"
        "python -m docs.scripts.benchmark_kernel_gmm --device gpu --output docs/benchmarks/kernel_gmm_gpu.json\n"
        "JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_ssd --device cpu \\\n"
        "  --output docs/benchmarks/kernel_ssd_cpu.json\n"
        "\n"
        "# Rebuild the figures and this page from the JSON.\n"
        "python -m docs.scripts.benchmark_figures\n"
        "```\n"
    )
    A(
        "The Mamba runner's `--peak` and `--max-len` flags, and the `peak`, `scale` and "
        "`cfm` suites of the OT runner, produce the memory and largest-size rows.\n"
    )
    return "\n".join(o)


# =========================================================================== nn
NN, NG, NC = "gpu_nn", "kernel_nn_gpu", "kernel_nn_cpu"
NN_ATT = ["probjax_default", "probjax_flex", "jax_xla", "jax_cudnn"]


def nn_figures() -> dict[str, str]:
    figs = {}
    panels = []
    for suite, name in (
        ("transformer", "Transformer"),
        ("diffusion", "DiffusionTransformer"),
    ):
        cats, keyed = [], []
        for dt in ("bfloat16", "float32"):
            for t in sorted({r["tokens"] for r in rows(NN, suite=suite)}):
                cats.append(f"{DT[dt]} · {t} tokens")
                keyed.append(dict(dtype=dt, tokens=t))
        for op in ("forward", "train_step"):
            ser = [
                (
                    f"nn:{a}",
                    {
                        cat: get(NN, suite=suite, attention=a, op=op, **kf)
                        for cat, kf in zip(cats, keyed, strict=False)
                    },
                )
                for a in NN_ATT
            ]
            panels.append(
                lambda ax, ser=ser, cats=cats, op=op, name=name: dot_panel(
                    ax,
                    cats,
                    ser,
                    title=f"{name} · {op.replace('_', ' ')}",
                    xlabel="time (ms)",
                    labels=op == "forward",
                )
            )
    figs["transformer"] = save_figure(
        "nn_transformer", panels, 2, width=8.0, panel_h=2.3
    )

    combos = [
        (False, "none"),
        (False, "dots_saveable"),
        (False, "full"),
        (True, "none"),
        (True, "full"),
    ]
    cats = [f"{'scan' if s else 'no scan'} · {r}" for s, r in combos]
    panels = []
    variants = [
        ("transformer", "probjax_flex", "Transformer, flex_attention"),
        ("diffusion", "probjax_flex", "DiffusionTransformer, flex_attention"),
        ("transformer", "probjax_default", "Transformer, einsum attention"),
    ]
    for model, att, title in variants:
        f = dict(
            suite="scan_remat",
            model=model,
            attention=att,
            layers=24,
            dtype="bfloat16",
            tokens=4096,
        )
        step = [get(NG, scan=s, remat=r, measure=None, **f) for s, r in combos]
        comp = [
            get(NG, "compile_s", scan=s, remat=r, measure=None, **f) for s, r in combos
        ]
        peak = [
            get(NG, "peak_mb", scan=s, remat=r, measure="device_peak", **f)
            for s, r in combos
        ]
        comp = [
            v if isnum(v) else (st if isinstance(st, str) else v)
            for v, st in zip(comp, step, strict=False)
        ]
        first = model == "transformer" and att == "probjax_flex"
        panels += [
            lambda ax, v=step, t=title, first=first: bar_panel(
                ax, cats, v, title=t + "\n train step (ms)", labels=True
            ),
            lambda ax, v=comp: bar_panel(
                ax, cats, v, title="\ncompile (s)", labels=False
            ),
            lambda ax, v=peak: bar_panel(
                ax, cats, v, title="\ndevice peak (MB)", labels=False
            ),
        ]
    figs["scan_remat"] = save_figure(
        "nn_scan_remat", panels, 3, width=8.0, panel_h=1.9, legend=False
    )

    cells = ["lru", "ssd", "mamba2", "gdn", "mamba"]
    names = {
        "lru": "LRUCell",
        "ssd": "SSDCell",
        "mamba2": "Mamba2Cell",
        "gdn": "GatedDeltaNetCell",
        "mamba": "MambaCell",
    }
    toks = sorted({r["tokens"] for r in rows(NN, suite="ssm")})
    STY.update({
        f"tok:{t}": S(f"{t} tokens", col, "o" if i == 0 else "s")
        for i, (t, col) in enumerate(zip(toks, (P3, P2), strict=False))
    })
    panels = []
    for op in ("forward", "train_step"):
        ser = [
            (
                f"tok:{t}",
                {
                    names[cl]: get(NN, suite="ssm", cell=cl, op=op, tokens=t)
                    for cl in cells
                },
            )
            for t in toks
        ]
        panels.append(
            lambda ax, ser=ser, op=op: dot_panel(
                ax,
                [names[cl] for cl in cells],
                ser,
                title=op.replace("_", " "),
                xlabel="time (ms)",
                labels=op == "forward",
            )
        )
    figs["ssm"] = save_figure("nn_ssm", panels, 2, width=7.5, panel_h=2.1)
    return figs


def page_nn(figs) -> str:
    o = []
    A = o.append
    gpu = gpu_name()
    cfg = load(NN)["config"]
    A("# Model performance (`probjax.nn`)\n")
    A(
        "End-to-end forward and train-step cost of `probjax.nn` models and of the "
        "`scan_layers` / `remat_policy` options, measured on the "
        f"{gpu} ([`gpu_nn.json`](gpu_nn.json), [`kernel_nn_gpu.json`](kernel_nn_gpu.json)) "
        "and on CPU ([`kernel_nn_cpu.json`](kernel_nn_cpu.json)). Kernel-level numbers are "
        "on the [attention](gpu_attention.md) and [kernel](gpu_kernels.md) pages, setup on "
        "the [overview](gpu_overview.md). A train step is `value_and_grad` of a scalar loss "
        "plus an optimizer update. Times are medians in milliseconds. "
        + FOOT_STATUS
        + "\n"
    )

    A("## Transformer and DiffusionTransformer\n")
    p = rows(NN, suite="transformer")[0]
    A(
        "The standard profile (model width "
        f"{p['model_dim']}, 8 heads of 32, 4 layers), batch {cfg['batch']}, with each "
        "`attention_fn`. The default `dot_product_attention` dispatches to `flex_attention` "
        "on NVIDIA GPUs ([rules](gpu_attention.md#short-sequences-and-the-default-dispatch)).\n"
    )
    A(
        img(
            figs["transformer"],
            "Transformer forward and train-step time by attention_fn",
            f"Batch {cfg['batch']}, width {p['model_dim']}, 4 layers, {gpu}. cuDNN has no fp32 "
            "path.",
        )
    )
    t = dict(suite="transformer", dtype="bfloat16", tokens=4096)
    A(
        "At 4096 tokens in bf16 the default train step takes "
        f"{q(NN, attention='probjax_default', op='train_step', **t)} ms against "
        f"{q(NN, attention='jax_cudnn', op='train_step', **t)} ms with cuDNN attention; "
        f"`jax.nn` XLA attention is {c(NN, attention='jax_xla', op='train_step', **t)} at "
        "that size."
        + (
            " At this head size (32) the probjax kernel is faster than cuDNN in both the "
            "forward and the train step."
            if all(
                num(NN, attention="probjax_default", op=op, **t)
                < num(NN, attention="jax_cudnn", op=op, **t)
                for op in ("forward", "train_step")
            )
            else ""
        )
        + "\n"
    )
    body = []
    for suite in ("transformer", "diffusion"):
        for dt in ("bfloat16", "float32"):
            for tok in sorted({r["tokens"] for r in rows(NN, suite=suite)}):
                for op in ("forward", "train_step"):
                    body.append(
                        [suite, DT[dt], tok, op.replace("_", " ")]
                        + [
                            c(NN, suite=suite, attention=a, dtype=dt, tokens=tok, op=op)
                            for a in NN_ATT
                        ]
                    )
    A(
        collapsible(
            "Table",
            table(
                ["model", "dtype", "tokens", "op"] + [lab(f"nn:{a}") for a in NN_ATT],
                body,
            ),
        )
    )

    A("## Scan over layers and rematerialization\n")
    A(
        "`Transformer` and `DiffusionTransformer` take `remat_policy=None | 'full' | "
        "'nothing_saveable' | <jax.checkpoint_policies name> | callable` and "
        "`scan_layers=False`. `scan_layers=True` stacks the layer parameters and runs the "
        "blocks under `jax.lax.scan`; it changes the parameter tree, so checkpoints of an "
        "unscanned model do not load into a scanned one. Both are opt-in.\n"
    )
    A(
        img(
            figs["scan_remat"],
            "Train step, compile time and device peak by scan_layers and remat_policy",
            f"24 layers, width 512, 4096 tokens, batch 2, bf16, {gpu}; device peak in a fresh "
            "process. The last row uses einsum attention (`implementation='xla'`), which "
            "materializes the scores.",
        )
    )
    f = dict(
        suite="scan_remat",
        model="transformer",
        attention="probjax_flex",
        layers=24,
        dtype="bfloat16",
        tokens=4096,
    )
    A(
        "`remat_policy='full'` cuts the device peak of the 24-layer bf16 model from "
        f"{q(NG, 'peak_mb', f_mb, scan=False, remat='none', measure='device_peak', **f)} to "
        f"{q(NG, 'peak_mb', f_mb, scan=False, remat='full', measure='device_peak', **f)} MB "
        f"for {q(NG, scan=False, remat='full', measure=None, **f)} instead of "
        f"{q(NG, scan=False, remat='none', measure=None, **f)} ms per step (the forward runs "
        "twice), and is what lets the einsum-attention model train at all at this length. "
        "`scan_layers=True` compiles in "
        f"{q(NG, 'compile_s', f_s, scan=True, remat='full', measure=None, **f)} s instead of "
        f"{q(NG, 'compile_s', f_s, scan=False, remat='full', measure=None, **f)} s, but "
        "without remat it keeps every layer's activations as scan residuals and runs out of "
        "memory. For deep stacks use `scan_layers=True, remat_policy='full'`; "
        "`dots_saveable` sits between the two.\n"
    )

    def sr_table(src, model, att, layers, dtype, tokens):
        body = []
        for scan in (False, True):
            for remat in ("none", "dots_saveable", "full"):
                ff = dict(
                    suite="scan_remat",
                    model=model,
                    attention=att,
                    layers=layers,
                    scan=scan,
                    remat=remat,
                    dtype=dtype,
                    tokens=tokens,
                )
                if not rows(src, **ff):
                    continue
                step = get(src, measure=None, **ff)
                body.append([
                    "yes" if scan else "no",
                    remat,
                    cell(step),
                    c(src, "compile_s", f_s, measure=None, **ff)
                    if isnum(step)
                    else "—",
                    c(src, "xla_temp_mb", f_mb, measure=None, **ff)
                    if isnum(step)
                    else "—",
                    c(src, "peak_mb", f_mb, measure="device_peak", **ff),
                ])
        return table(
            [
                "`scan_layers`",
                "`remat_policy`",
                "train step (ms)",
                "compile (s)",
                "XLA temp (MB)",
                "device peak (MB)",
            ],
            body,
            ["l", "l", "r", "r", "r", "r"],
        )

    A(
        collapsible(
            "Tables (GPU 24 and 12 layers; CPU)",
            "**GPU, Transformer, 24 layers, `flex_attention`**\n\n"
            + sr_table(NG, "transformer", "probjax_flex", 24, "bfloat16", 4096)
            + "\n\n**GPU, DiffusionTransformer, 24 layers, `flex_attention`**\n\n"
            + sr_table(NG, "diffusion", "probjax_flex", 24, "bfloat16", 4096)
            + "\n\n**GPU, Transformer, 24 layers, einsum attention**\n\n"
            + sr_table(NG, "transformer", "probjax_default", 24, "bfloat16", 4096)
            + "\n\n**GPU, Transformer, 12 layers, einsum attention**\n\n"
            + sr_table(NG, "transformer", "probjax_default", 12, "bfloat16", 4096)
            + "\n\n**CPU, width 512, 512 tokens, batch 2, fp32, 24 layers**\n\n"
            + sr_table(NC, "transformer", "probjax_default", 24, "float32", 512)
            + "\n\n**CPU, 12 layers**\n\n"
            + sr_table(NC, "transformer", "probjax_default", 12, "float32", 512),
        )
    )
    fc = dict(
        suite="scan_remat",
        model="transformer",
        attention="probjax_default",
        layers=24,
        dtype="float32",
        tokens=512,
        measure=None,
    )
    A(
        "On CPU (24 layers, 512 tokens), `remat_policy='full'` is also faster, not only "
        f"smaller: {q(NC, scan=False, remat='full', **fc)} vs "
        f"{q(NC, scan=False, remat='none', **fc)} ms per step, because the smaller working "
        "set stays in cache.\n"
    )

    A("## SSMModel cells\n")
    sm = rows(NN, suite="ssm")[0]
    A(
        f"`SSMModel(16, {sm['model_dim']}, 16, 4)` with each recurrent cell, batch "
        f"{cfg['batch']}, fp32. `Mamba2Cell` is the full Mamba-2 block on `ssd` "
        "(`expand=2`, short causal convolution, gated RMSNorm, `head_dim=64`, "
        "`state_dim=64`); `SSDCell` is a multi-head SSD layer with the Mamba-2 decay "
        "parameterization; `MambaCell` uses `state_dim = model_dim` by default; "
        "`GatedDeltaNetCell` and `LRUCell` use their defaults.\n"
    )
    A(
        img(
            figs["ssm"],
            "SSMModel forward and train-step time by cell",
            f"`SSMModel(16, {sm['model_dim']}, 16, 4)`, batch {cfg['batch']}, fp32, {gpu}.",
        )
    )
    names = {
        "lru": "`LRUCell`",
        "ssd": "`SSDCell`",
        "mamba2": "`Mamba2Cell`",
        "gdn": "`GatedDeltaNetCell`",
        "mamba": "`MambaCell`",
    }
    body = []
    for cl in ("lru", "ssd", "mamba2", "gdn", "mamba"):
        for tok in sorted({r["tokens"] for r in rows(NN, suite="ssm")}):
            ff = dict(suite="ssm", cell=cl, tokens=tok)
            body.append([
                names[cl],
                f_int(get(NN, "params", **ff)),
                tok,
                c(NN, op="forward", **ff),
                c(NN, op="train_step", **ff),
                c(NN, "compile_s", f_s, op="train_step", **ff),
            ])
    A(
        table(
            [
                "cell",
                "parameters",
                "tokens",
                "forward (ms)",
                "train step (ms)",
                "train-step compile (s)",
            ],
            body,
            ["l", "r", "r", "r", "r", "r"],
        )
        + "\n"
    )
    A(
        "For the kernels underneath see [SSD](gpu_kernels.md#ssd-mamba-2-state-space-duality), "
        "[Mamba](gpu_kernels.md#mamba-selective-scan) and "
        "[Gated DeltaNet](gpu_kernels.md#gated-deltanet).\n"
    )

    A("## UNet\n")
    body = []
    for dt in ("bfloat16", "float32"):
        for res in (32, 64):
            for op in ("forward", "train_step"):
                ff = dict(suite="unet", dtype=dt, resolution=res, op=op)
                body.append([
                    DT[dt],
                    res,
                    op.replace("_", " "),
                    c(NN, **ff),
                    c(NG, "xla_temp_mb", f_mb, measure=None, variant=None, **ff),
                    c(NG, "peak_mb", f_mb, measure="device_peak", **ff),
                ])
    A(
        f"`UNet(3, (64, 128, 256))`, batch {cfg['batch']}, {gpu}; device peak in a fresh "
        "process:\n"
    )
    A(
        table(
            [
                "dtype",
                "resolution",
                "op",
                "time (ms)",
                "XLA temp (MB)",
                "device peak (MB)",
            ],
            body,
        )
        + "\n"
    )
    body = []
    for v in ("plain", "remat", "donate"):
        ff = dict(
            suite="unet", variant=v, dtype="bfloat16", resolution=64, op="train_step"
        )
        body.append([
            v,
            c(NG, **ff),
            c(NG, "compile_s", f_s, **ff),
            c(NG, "xla_temp_mb", f_mb, **ff),
        ])
    A(
        "bf16 train step at resolution 64 with remat or donated buffers. Remat costs time "
        "without reducing temporaries at this size:\n"
    )
    A(table(["variant", "time (ms)", "compile (s)", "XLA temp (MB)"], body) + "\n")

    A("## MoE layer\n")
    mo = rows(NN, suite="moe")[0]
    body = []
    for dt in ("bfloat16", "float32"):
        for tok in sorted({r["tokens"] for r in rows(NN, suite="moe")}):
            for op in ("forward", "train_step"):
                body.append(
                    [DT[dt], f_int(tok), op.replace("_", " ")]
                    + [
                        c(NN, suite="moe", impl=i, dtype=dt, tokens=tok, op=op)
                        for i in ("pallas", "blocked", "ragged_dot")
                    ]
                )
    A(
        f"Dropless `MoELayer`, {mo['experts']} experts, top-{mo['top_k']}, d_model "
        f"{mo['d_model']}, d_hidden {mo['d_hidden']}, one column per `dropless_impl`. More "
        "code paths, memory and the largest token counts are on the "
        "[kernel page](gpu_kernels.md#moe-grouped-matmul):\n"
    )
    A(
        table(
            ["dtype", "tokens", "op", "Pallas (default)", "blocked", "`ragged_dot`"],
            body,
        )
        + "\n"
    )

    A("## Training-loop overhead\n")
    body = []
    for src, dev in ((NG, "GPU"), (NC, "CPU")):
        for w in (64, 512):
            ff = dict(suite="train_loop", width=w)
            body.append([
                dev,
                w,
                c(src, "per_step_ms", op="bare_jit", **ff),
                c(src, "per_step_ms", op="fit_array", **ff),
                c(src, "per_step_ms", op="fit_stream", **ff),
                c(src, "first_call_s", f_s, op="fit_array", **ff),
                c(src, "second_call_s", f_s, op="fit_array", **ff),
            ])
    A(
        "Per-step time of an MLP trained with a bare jitted step and with `fit()` on an "
        "in-memory array and on a stream. `fit()` recompiles on every call: a second call "
        "with the same model and data costs nearly as much wall time as the first (last "
        "two columns).\n"
    )
    A(
        table(
            [
                "device",
                "width",
                "bare jit (ms/step)",
                "`fit` array (ms/step)",
                "`fit` stream (ms/step)",
                "`fit` first call (s)",
                "`fit` second call (s)",
            ],
            body,
        )
        + "\n"
    )

    A("## How to reproduce\n")
    A(
        "```bash\n"
        "# Model grid (Transformer, DiffusionTransformer, SSM cells, UNet, MoE).\n"
        "python -m docs.scripts.benchmark_gpu_nn --output docs/benchmarks/gpu_nn.json\n"
        "\n"
        "# scan_layers / remat_policy, UNet variants, training loop; --peak adds device peaks.\n"
        "python -m docs.scripts.benchmark_kernel_nn --device gpu --output docs/benchmarks/kernel_nn_gpu.json\n"
        "JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_nn --device cpu \\\n"
        "  --output docs/benchmarks/kernel_nn_cpu.json\n"
        "\n"
        "# Rebuild the figures and this page from the JSON.\n"
        "python -m docs.scripts.benchmark_figures\n"
        "```\n"
    )
    return "\n".join(o)


# =========================================================================== overview
def glance_rows():
    """(label, probjax label, probjax value, {alternative key: value}) per workload."""
    W = []
    att = lambda dt, d, causal, mode, s: dict(
        dtype=dt,
        head_dim=d,
        causal=causal,
        mode=mode,  # noqa: E731
        seq=s,
    )
    for lbl, f in (
        (
            "Attention · bf16 d64 · fwd+bwd · T=4096",
            att("bfloat16", 64, False, "fwd_bwd", 4096),
        ),
        (
            "Attention · bf16 d128 · fwd+bwd · T=4096",
            att("bfloat16", 128, False, "fwd_bwd", 4096),
        ),
        (
            "Attention · bf16 d128 causal · fwd · T=4096",
            att("bfloat16", 128, True, "fwd", 4096),
        ),
        (
            "Attention · fp32 d64 · fwd+bwd · T=4096",
            att("float32", 64, False, "fwd_bwd", 4096),
        ),
        (
            "Attention · bf16 d64 · fwd+bwd · T=8192",
            att("bfloat16", 64, False, "fwd_bwd", 8192),
        ),
    ):
        W.append((lbl, GA, f, "impl", "probjax_default", ATT_ALTS, ""))
    W.append((
        "GQA decoding · bf16 · B=8, 16 queries, 64k keys",
        "kernel_attn_ext_gpu",
        dict(
            suite="decode",
            dtype="bfloat16",
            batch=8,
            lq=16,
            lkv=65536,
            hkv=8,
            head_dim=128,
        ),
        "impl",
        "probjax_splitkv",
        ["jax_xla", "jax_cudnn", "torch_sdpa"],
        "",
    ))
    W.append((
        "Attention on CPU · causal fwd+bwd · T=4096",
        "kernel_attn-cpu_cpu",
        dict(variant="causal", mode="fwdbwd", head_dim=64, dtype="float32", seq=4096),
        "impl",
        "probjax_mha",
        ["jax_xla", "torch_sdpa"],
        "cpu:",
    ))
    W.append((
        "SSD · fwd+bwd · L=8192",
        "kernel_ssd_gpu",
        dict(config="g2h8_d128", seq=8192, mode="fwd_bwd", kind=None),
        "impl",
        "chunked",
        ["pallas", "scan"],
        "ssd:",
    ))
    W.append((
        "Mamba scan · fwd+bwd · L=8192",
        "kernel_mamba_gpu",
        dict(mode="fwd_bwd", seq=8192, dim=512, kind=None),
        "impl",
        "pallas",
        ["chunked", "assoc", "scan"],
        "mamba:",
    ))
    W.append((
        "Residual linear attention · fwd+bwd · L=8192",
        "kernel_rla_gpu",
        dict(suite="kernel", dtype="float32", dim=64, seq=8192, mode="fwd_bwd"),
        "impl",
        "chunked",
        ["pallas", "pallas_scanbwd", "scan"],
        "rla:",
    ))
    W.append((
        "Gated DeltaNet · bf16 d128 · fwd+bwd · L=8192",
        "kernel_delta_gpu",
        dict(dtype="bfloat16", dk=128, seq=8192, mode="fwd_bwd"),
        "impl",
        "pallas",
        ["chunked", "chunked_fat", "recurrent"],
        "delta:",
    ))
    W.append((
        "rbf_kernel_mv · fwd+bwd · N=M=65536",
        "kernel_kmv_gpu",
        dict(family="kernel_mv", dim=8, out=16, n=65536, mode="fwd_bwd", batch=1),
        "impl",
        "pallas",
        ["xla", "dense", "torch_dense"],
        "kmv:",
    ))
    W.append((
        "kernel_logsumexp · fwd · N=M=16384, D=8",
        "kernel_ot_gpu",
        dict(suite="lse", d=8, n=16384, mode="fwd"),
        "impl",
        "pallas",
        ["dense"],
        "lse:",
    ))
    W.append((
        "Sinkhorn · 50 iterations · N=M=16384",
        "kernel_ot_gpu",
        dict(suite="sinkhorn", d=8, n=16384, mode="fwd"),
        "impl",
        "kernel",
        ["dense"],
        "lse:",
    ))
    W.append((
        "MoE · bf16 train step · 32k tokens",
        "kernel_gmm_gpu",
        dict(
            run="f_gpu_a",
            dtype="bfloat16",
            tokens=32768,
            op="train_step",
            suite="layer",
        ),
        "path",
        "pallas/sorted",
        ["blocked", "ragged_dot"],
        "moe:",
    ))
    W.append((
        "Transformer · bf16 train step · 4096 tokens",
        NN,
        dict(suite="transformer", dtype="bfloat16", tokens=4096, op="train_step"),
        "attention",
        "probjax_default",
        ["jax_xla", "jax_cudnn"],
        "nn:",
    ))
    out = []
    for lbl, src, f, key, ours, alts, pre in W:
        mine = get(src, **{key: ours}, **f)
        others = {
            pre + a if pre + a in STY else a: get(src, **{key: a}, **f) for a in alts
        }
        out.append((lbl, pre + ours if pre + ours in STY else ours, mine, others))
    return out


def overview_figure(glance) -> str:
    cats, data = [], {}
    for lbl, _, mine, others in glance:
        ok = {k: v for k, v in others.items() if isnum(v)}
        cats.append(lbl)
        if not isnum(mine):
            data[lbl] = None
        elif ok:
            data[lbl] = min(ok.values()) / mine
        else:
            data[lbl] = next(iter(others.values()), None)
    STY["glance"] = S("fastest alternative ÷ probjax", P1)

    def panel(ax):
        r = dot_panel(
            ax,
            cats,
            [("glance", data)],
            xlabel="speed-up of the probjax default "
            "over the fastest other implementation (×, log scale)",
            ref=1.0,
        )
        lo, hi = ax.get_xlim()
        ax.text(
            1.03, len(cats) - 0.35, "probjax faster →", fontsize=7, color=INK, ha="left"
        )
        ax.text(
            0.97,
            len(cats) - 0.35,
            "← alternative faster",
            fontsize=7,
            color=INK,
            ha="right",
        )
        ax.set_ylim(len(cats) - 0.1, -0.6)
        if hi < 1.3:
            ax.set_xlim(lo, 1.3)
        return r

    return save_figure(
        "overview_speedup", [panel], 1, width=8.0, panel_h=4.6, legend=False
    )


def page_overview(glance, fig) -> str:
    o = []
    A = o.append
    dev = load(GA)["device"]
    A("# GPU and CPU performance\n")
    A(
        "How probjax's kernels and models compare with the alternatives in JAX and PyTorch, "
        "in time and memory, on a consumer NVIDIA GPU and on CPU:\n\n"
        "- [Attention](gpu_attention.md): `dot_product_attention`, `flex_attention` and "
        "`mha` against `jax.nn` (XLA, cuDNN) and PyTorch SDPA; memory, short sequences, "
        "GQA and split-KV decoding, CPU.\n"
        "- [Sequence mixers, kernel sums and MoE](gpu_kernels.md): SSD, Mamba scan, residual "
        "linear attention, Gated DeltaNet, `rbf_kernel_mv` / KDE, `kernel_logsumexp` / "
        "Sinkhorn / OT, MoE grouped matmul.\n"
        "- [Models](gpu_nn.md): `Transformer` / `DiffusionTransformer` per `attention_fn`, "
        "`scan_layers` and `remat_policy`, `SSMModel` cells, `UNet`, `MoELayer`.\n"
        "- [H100 (Hopper)](gpu_h100.md): the same suites on an H100, H100 vs RTX 3090, the "
        "device-specific attention defaults for sm_90"
        + (" (cuDNN routing, the Mosaic GPU backend, backward choice)" if _h100_mosaic() else "")
        + ", MoE dispatch on H100, and PyTorch (`flex_attention`, SDPA) on H100.\n\n"
        f"**Devices.** These pages measure the {gpu_name()} unless they say otherwise; the H100 "
        "results are on [their own page](gpu_h100.md). Defaults can differ per device: on sm_90 "
        "`flex_attention` / `dot_product_attention` route large bf16 calls to cuDNN and resolve "
        "`backward_pass_impl=\"auto\"` differently ([rules](gpu_h100.md#device-specific-defaults)).\n"
    )
    A("## At a glance\n")
    A(
        img(
            fig,
            "Speed-up of probjax over the fastest alternative per workload",
            f"Time of the fastest other implementation that ran (another probjax "
            "implementation or an external library) divided by the time of the probjax default, "
            f"{gpu_name()}; the CPU row ran on the shared host. Right of the line: the default is "
            "faster.",
        )
    )
    body = []
    for lbl, ours, mine, others in glance:
        ok = {k: v for k, v in others.items() if isnum(v)}
        if ok:
            best = min(ok, key=ok.get)
            alt, alt_v = lab(best), f_ms(ok[best])
            ratio = f_x(ok[best] / mine) if isnum(mine) else "—"
        else:
            alt, alt_v, ratio = (
                "all " + "/".join(dict.fromkeys(map(str, others.values()))),
                "—",
                "—",
            )
        body.append([lbl, lab(ours), cell(mine), alt, alt_v, ratio])
    A(
        table(
            [
                "workload",
                "probjax default",
                "ms",
                "fastest other implementation",
                "ms",
                "speed-up",
            ],
            body,
            ["l", "l", "r", "l", "r", "r"],
        )
        + "\n"
    )
    A(
        "Exact shapes are on the linked pages; ratios at other shapes differ, sometimes a "
        "lot.\n"
    )

    A("## Setup\n")
    A(
        table(
            ["", ""],
            [
                [
                    "GPU",
                    f"{dev['gpu']}, {dev['gpu_memory_gb']:.0f} GB (sm_86, about 99 KB opt-in "
                    "shared memory per block)",
                ],
                ["Driver", dev["driver"]],
                ["JAX", f"{dev['jax']} (Pallas on Triton for GPU kernels)"],
                [
                    "PyTorch (comparisons only)",
                    f"{dev['torch']} (CUDA {dev['torch_cuda']})",
                ],
                ["Python", dev["python"]],
                ["Host", "24 CPU threads, shared with other jobs"],
            ],
            ["l", "l"],
        )
        + "\n"
    )
    A(
        "Every JSON file records the device, the library versions and the runner configuration "
        "under `device` and `config`.\n"
    )
    A("## Methodology\n")
    A(
        "- **Timing.** Each case is compiled once, then timed as `inner` back-to-back calls "
        "bracketed by a device synchronization, repeated several times. Pages show the median "
        "per call; every JSON row also has the inter-quartile range (`q1_ms`, `q3_ms`). "
        "Compilation is timed separately (`compile_s`) and never included in the step time. "
        "No two benchmarks shared the GPU.\n"
        "- **Memory**, two kinds. `xla_temp_mb` / `xla_arg_mb` / `xla_out_mb` are XLA's "
        "compile-time `memory_analysis()` of the jitted function: temporaries, arguments, "
        "outputs. `peak_mb` is the true device peak (`peak_bytes_in_use`) read in a fresh "
        "subprocess that runs only that case. Temporaries say what a kernel allocates; the "
        "peak says whether the whole call fits. `residual_mb` is the size of the `jax.vjp` "
        "residuals kept for the backward pass.\n"
        "- **Accuracy** is reported as a *relative* error (`max|a − b| / max|b|` or a norm "
        "ratio; the JSON says which) against an fp32 reference at `HIGHEST` matmul precision "
        "or an fp64 host reference.\n"
        "- **TF32.** On Ampere GPUs JAX runs fp32 matmuls in TF32 by default, which costs "
        "about three decimal digits; fp32 kernels not pinned to `HIGHEST` show relative "
        "errors around 1e-3.\n"
        "- **Shared host.** CPU timings ran on a shared 24-thread machine and carry 10–30% "
        "run-to-run noise; read CPU ratios as approximate.\n"
        "- **Cases that did not run.** "
        + FOOT_STATUS
        + " Figures mark them with a cross "
        "and the reason; the JSON keeps the `error` text.\n"
        "- **Figures** are generated by `docs/scripts/benchmark_figures.py` from the JSON, "
        "together with these four pages.\n"
    )

    A("## What runs by default\n")
    A(
        "With no `implementation` argument, the public functions dispatch as follows. "
        "\"Kernel\" means a Pallas kernel compiled through Triton.\n"
    )
    A(
        "| Operation | CPU | NVIDIA GPU | Override |\n"
        "| --- | --- | --- | --- |\n"
        "| `dot_product_attention` (default `attention_fn`) | Flax einsum | `flex_attention` "
        "kernel when exact (no dense mask or bias, no dropout, default precision, ≥ 64 keys); "
        "einsum otherwise | `implementation='xla' / 'pallas'` |\n"
        "| `flex_attention`, `mha`, `mla` | XLA: dense ≤ 256, single tile ≤ 512, else "
        "256×256 tiles | kernel; split-KV forward for unmasked Lq ≤ 64, Lkv ≥ 1024 | "
        "`split_kv=False` |\n"
        "| `ssd` | chunked pure JAX | chunked pure JAX (Pallas only on TPU) | "
        "`implementation='pallas' / 'xla'` |\n"
        "| `compute_mamba_scan` | chunked XLA scan | Triton sequence-loop kernel (chunked XLA "
        "for unsupported shapes) | — |\n"
        "| `residual_linear_attention` | chunked pure JAX | chunked pure JAX | "
        "`implementation='pallas' / 'scan'` |\n"
        "| `gated_delta_rule` | chunked pure JAX | kernel for supported chunk / head sizes, "
        "else chunked | `implementation='recurrent' / 'chunked' / 'pallas'` |\n"
        "| `rbf_kernel_mv`, `kde_density` | row-blocked XLA | kernel | "
        "`implementation='xla' / 'pallas'` |\n"
        "| `kernel_logsumexp` | blocked XLA | kernel (3×TF32 cross term); dense for "
        "N·M ≤ 2²¹ | `impl='pallas' / 'xla' / 'dense'` |\n"
        "| `MoELayer(routing='dropless')` | `chunked` | grouped-matmul kernel, `sorted` "
        "dispatch | `dropless_impl`, `dropless_dispatch` |\n"
        "| `Transformer`, `DiffusionTransformer` | no remat, no scan | no remat, no scan | "
        "`remat_policy`, `scan_layers` (changes the parameter tree) |\n"
    )

    A("## Known limitations\n")
    bullets = attn_faster_alternatives()
    g = dict(dtype="bfloat16", head_dim=128, causal=False, mode="fwd_bwd")
    lims = []
    if bullets:
        lims.append(
            "- **Attention:** in some bf16 configurations cuDNN or torch flash is faster than "
            "the probjax kernel (bf16 head_dim 128 fwd+bwd at T=8192: probjax "
            f"{q(GA, impl='probjax_default', seq=8192, **g)} ms, cuDNN "
            f"{q(GA, impl='jax_cudnn', seq=8192, **g)} ms); the full list is "
            "[on the attention page](gpu_attention.md#where-an-alternative-is-faster)."
        )
    C = "kernel_attn-cpu_cpu"
    f = dict(variant="causal", mode="fwdbwd", head_dim=64, dtype="float32", seq=4096)
    lims += [
        "- **Attention on CPU:** torch's native SDPA kernel is faster than any XLA "
        f"formulation (causal fwd+bwd T=4096: probjax {q(C, impl='probjax_mha', **f)} ms, "
        f"torch {q(C, impl='torch_sdpa', **f)} ms).",
        "- `KeyPaddingMask` gives wrong results in the attention kernel path; use "
        "`KVLenMask`.",
        "- The `RAttention` layer runs out of memory at 8192 tokens on a 24 GB GPU because "
        "its sliding-window attention branch is dense "
        "([kernel page](gpu_kernels.md#residual-linear-attention-rattention)).",
        "- `fit()` recompiles on every call ([models page](gpu_nn.md#training-loop-overhead)).",
        "- Dense attention (`jax.nn` XLA, torch math, probjax `implementation='xla'`) does not "
        f"run fwd+bwd beyond a few thousand tokens at batch 4 and 8 heads (`jax.nn` XLA fails "
        f"from T={first_fail('jax_xla')}, torch math from T={first_fail('torch_math')}).",
    ]
    A("\n".join(lims) + "\n")

    A("## How to reproduce\n")
    A(
        "All runners live in `docs/scripts/` and write JSON to `docs/benchmarks/`. GPU runs "
        "need a CUDA GPU; torch is optional and only adds comparison rows. Run one benchmark "
        "at a time on an otherwise idle GPU, then rebuild the figures and pages.\n"
    )
    A(
        "```bash\n"
        "# Attention grid against JAX and PyTorch, and the tile sweep.\n"
        "python -m docs.scripts.benchmark_gpu_attention --output docs/benchmarks/gpu_attention.json\n"
        "python -m docs.scripts.benchmark_gpu_attention_tiles --output docs/benchmarks/gpu_attention_tiles.json\n"
        "\n"
        "# Model grid.\n"
        "python -m docs.scripts.benchmark_gpu_nn --output docs/benchmarks/gpu_nn.json\n"
        "\n"
        "# One kernel family, CPU then GPU (other families: see each page).\n"
        "JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_kernel_ssd --device cpu \\\n"
        "  --output docs/benchmarks/kernel_ssd_cpu.json\n"
        "python -m docs.scripts.benchmark_kernel_ssd --device gpu --output docs/benchmarks/kernel_ssd_gpu.json\n"
        "\n"
        "# Figures (docs/benchmarks/figures/*.svg) and the four GPU pages; needs matplotlib.\n"
        "python -m docs.scripts.benchmark_figures\n"
        "```\n"
    )
    A(
        "Shared timing helpers are in `docs/scripts/_gpu_bench.py`. Each page ends with the "
        "commands for its own files.\n"
    )
    return "\n".join(o)


# =========================================================================== main
def _write(name: str, text: str) -> None:
    text = text.rstrip() + "\n"
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    (BENCH / f"{name}.md").write_text(text)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--no-pages", action="store_true", help="only write the figures"
    )
    args = parser.parse_args(argv)
    _setup_mpl()
    for old in FIG.glob("*.svg") if FIG.exists() else ():
        old.unlink()
    att = attention_figures()
    ker = kernel_figures()
    nnf = nn_figures()
    glance = glance_rows()
    ov = overview_figure(glance)
    from docs.scripts import benchmark_figures_h100 as h100

    h100_figs = h100.h100_figures()
    if not args.no_pages:
        _write("gpu_attention", page_attention(att))
        _write("gpu_kernels", page_kernels(ker))
        _write("gpu_nn", page_nn(nnf))
        _write("gpu_overview", page_overview(glance, ov))
        _write("gpu_h100", h100.page_h100(h100_figs))
    for name in FIGURES:
        size = (FIG / f"{name}.svg").stat().st_size
        print(f"figures/{name}.svg  {size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
