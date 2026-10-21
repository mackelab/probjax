"""Small controlled benchmark: masked discrete diffusion vs quantization+continuous.

Question
--------
For tabular-style values normalized to [-1, 1], compare, holding the backbone
size and the number of training steps fixed:

  * ``masked``      -- absorbing-mask discrete diffusion over N bins (one MASK
                       token as an extra class), trained with the boxed
                       masked-only objective
                       ``E[-sum_{i: z_t^i = MASK} log p(z0^i | z_t, t)]``.
  * ``uniform``     -- uniform-corruption discrete diffusion over the same bins
                       (standard CE at every position).
  * ``continuous``  -- continuous EDM/flow matching on the dequantized values.
  * ``dequant``     -- continuous EDM/flow matching with data dequantized by
                       sampling uniformly inside each bin first.

Swept over bin counts N and over uniform-width vs quantile binning. Continuous
arms produce densities so they are scored by held-out NLL; discrete arms are
scored by held-out NLL *minus* the ``d * log(2 / N)`` piecewise-constant offset
so all arms are on the same continuous-density scale.

This is a quick sanity experiment, not a publication-grade harness: one seed, a
small MLP backbone, a few thousand steps, 2D targets.

Run:
    JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_discretization --steps 2000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from probjax.nn import EDM, MLP, FitConfig, GaussianFourierEmbedding
from probjax.nn.generative.discrete.model import (
    MultinomialDiffusion,
    MultinomialDiffusionSchedule,
)
from probjax.nn.nets.time import TimeMLP

BIN_COUNTS = (32, 64, 128, 256)
BINNING = ("uniform", "quantile")
ARMS = ("masked", "uniform", "continuous", "dequant")

LO, HI = -1.0, 1.0  # normalized value range


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def crescent(num_points, key=None):
    """Two-moons-like crescent: curved, bimodal, mildly skewed marginals."""
    key = jax.random.key(0) if key is None else key
    n = int(num_points)
    batch = max(4 * n, n + 128)

    def body(state):
        points, count, rng = state
        rng, k1, k2 = jax.random.split(rng, 3)
        theta = 2 * jnp.pi * jax.random.uniform(k1, (batch,))
        radius = jnp.sqrt(jax.random.uniform(k2, (batch,)))
        x = radius * jnp.cos(theta)
        y = radius * jnp.sin(theta)
        accept = (x - 0.5) ** 2 + y**2 > 0.36
        samples = jnp.stack((x, y), axis=1)
        pos = count + jnp.cumsum(accept.astype(jnp.int32)) - 1
        write = accept & (pos >= 0) & (pos < n)
        idx = jnp.where(write, pos, 0)
        points = points.at[idx].add(jnp.where(write[:, None], samples, 0.0))
        return points, count + write.sum(dtype=jnp.int32), rng

    out, _, _ = jax.lax.while_loop(
        lambda s: s[1] < n, body, (jnp.zeros((n, 2)), jnp.int32(0), key)
    )
    return out


def skewed_copula(num_points, key=None):
    """Strongly skewed, positively dependent margins: gamma + Gaussian copula.

    Complexity comes from the dependence (correlation 0.8 on the latent scale)
    plus heavy marginal skew; this is the case uniform binning may mishandle.
    """
    key = jax.random.key(0) if key is None else key
    n = int(num_points)
    k1, k2 = jax.random.split(key)
    cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    z = jax.random.multivariate_normal(k1, jnp.zeros(2), cov, shape=(n,))
    u = jax.nn.sigmoid(z)  # Gaussian copula uniforms
    # Gamma marginals (shape 1.5) via inverse-CDF would be ideal; use the
    # repo's gamma ppf applied elementwise for a genuine skewed density.
    from probjax.stats import gamma

    a, scale = 1.5, 1.0
    x = gamma.ppf(jnp.clip(u[:, 0], 1e-6, 1 - 1e-6), a, scale=scale)
    y = gamma.ppf(jnp.clip(u[:, 1], 1e-6, 1 - 1e-6), a, scale=scale)
    samples = jnp.stack((x, y), axis=1)
    # Robust normalize into [-1, 1] using the data quantiles (99.5%).
    lo = jnp.quantile(samples, 0.005, axis=0)
    hi = jnp.quantile(samples, 0.995, axis=0)
    return 2.0 * (samples - lo) / (hi - lo) - 1.0


TARGETS = {"crescent": crescent, "skewed_copula": skewed_copula}


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #
@dataclass
class Bins:
    edges: np.ndarray  # shape (d, N+1) in value space
    lo: np.ndarray
    hi: np.ndarray

    @property
    def num_bins(self):
        return self.edges.shape[1] - 1

    def encode(self, x):
        """(n, d) values -> (n, d) integer bin indices via searchsorted."""
        idx = np.stack(
            [
                np.clip(
                    np.searchsorted(self.edges[i, 1:-1], x[:, i]), 0, self.num_bins - 1
                )
                for i in range(x.shape[1])
            ],
            axis=1,
        )
        return idx.astype(np.int32)

    def centers(self):
        return 0.5 * (self.edges[:, :-1] + self.edges[:, 1:])  # (d, N)

    def sample_inside(self, rng, idx):
        """Uniform-in-bin dequantization -> (n, d) continuous values."""
        rngs = jax.random.split(rng, idx.shape[1])
        out = []
        for i in range(idx.shape[1]):
            e0 = jnp.asarray(self.edges[i])[idx[:, i]]
            e1 = jnp.asarray(self.edges[i])[idx[:, i] + 1]
            out.append(e0 + (e1 - e0) * jax.random.uniform(rngs[i], idx[:, i].shape))
        return jnp.stack(out, axis=1)

    def logpdf_from_bin_probs(self, probs):
        """Piecewise-constant density logpdf: log p_k - log(width_k)."""
        widths = np.diff(self.edges, axis=1)  # (d, N)
        logw = np.log(widths)
        return lambda idx, p=probs, lw=logw: (
            np.asarray(p)[
                np.arange(idx.shape[0])[:, None], np.arange(idx.shape[1])[None, :], idx
            ]
            - lw[np.arange(idx.shape[1])[None, :], idx]
        )


def build_bins(x, num_bins, scheme):
    d = x.shape[1]
    edges = np.empty((d, num_bins + 1))
    for i in range(d):
        if scheme == "uniform":
            grid = np.linspace(LO, HI, num_bins + 1)
        else:
            qs = np.linspace(0.0, 1.0, num_bins + 1)
            grid = np.quantile(x[:, i], qs)
            grid[0], grid[-1] = LO, HI
            grid = np.maximum.accumulate(grid)  # enforce monotone edges
        edges[i] = grid
    return Bins(edges, x.min(axis=0), x.max(axis=0))


def clamp_to_bins(x, bins):
    return np.clip(x, LO + 1e-6, HI - 1e-6)


# --------------------------------------------------------------------------- #
# Nets
# --------------------------------------------------------------------------- #
class CategoricalMLP(nnx.Module):
    """Backbone for discrete arms: consumes one-hot-minus-prior features."""

    def __init__(self, dim, num_classes, hidden=96, rngs=None):
        self.dim = dim
        self.num_classes = num_classes
        self.temb = GaussianFourierEmbedding(1, 32, rngs=rngs)
        self.in_layer = nnx.Linear(dim * num_classes, hidden, rngs=rngs)
        self.mlp = MLP([hidden, hidden, dim * num_classes], rngs=rngs, context_dim=32)

    def __call__(self, t, x, *args, **kwargs):
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        t = jnp.asarray(t, dtype=jnp.float32)
        if t.ndim == 1:
            t = t[:, None]
        elif t.ndim >= 2 and t.shape[1] != 1:
            t = jnp.mean(t, axis=tuple(range(1, t.ndim)), keepdims=True).reshape(b, 1)
        h = self.in_layer(x_flat)
        out = self.mlp(h, context=self.temb(t))
        return out.reshape(b, self.dim, self.num_classes)


def discrete_backbone(dim, num_classes, seed=0):
    return CategoricalMLP(dim, num_classes, rngs=nnx.Rngs(seed))


class MaskedDiffusion(MultinomialDiffusion):
    """Absorbing-mask diffusion trained with the boxed masked-only CE objective.

    The base ``MultinomialDiffusion.loss`` computes cross-entropy at every
    position; the boxed objective is
    ``E[-sum_{i: z_t^i = MASK} log p(z0^i | z_t, t)]``, i.e. only the absorbed
    positions contribute. This subclass samples ``z_t`` from the schedule and
    averages the CE over masked positions only. Sampling still uses the base
    reverse process, which is correct for the absorbing kernel.
    """

    def loss(self, rng, x0, *args, **kwargs):
        x0 = jnp.asarray(x0, dtype=jnp.int32)
        rng_t, rng_q = jax.random.split(rng)
        time_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        t = self.train_cfg.sample_times(rng_t, time_shape)
        x_t = jnp.asarray(self.schedule.q_sample(rng_q, x0, t), dtype=jnp.int32)
        logits = self.predict_x0_logits(t, x_t, *args, **kwargs)
        logp = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.take_along_axis(logp, x0[..., None], axis=-1).squeeze(-1)
        mask = (x_t == self.num_classes - 1).astype(jnp.float32)
        return jnp.sum(ce * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def continuous_backbone(dim, seed=0):
    return TimeMLP(dim, hidden_dim=128, depth=3, rngs=nnx.Rngs(seed))


# --------------------------------------------------------------------------- #
# Arm construction
# --------------------------------------------------------------------------- #
def build_model(arm, target_name, bins, N, steps, seed=0):
    d = bins.edges.shape[0]
    if arm in ("masked", "uniform"):
        if arm == "masked":
            base = jnp.zeros(N + 1).at[-1].set(1.0)  # pure MASK token prior
        else:
            base = jnp.ones(N + 1) / (N + 1)  # uniform-corruption prior
        schedule = MultinomialDiffusionSchedule.from_cosine(
            num_steps=64, num_classes=N + 1, base_probs=base
        )
        model = (MaskedDiffusion if arm == "masked" else MultinomialDiffusion)(
            discrete_backbone(d, N + 1, seed),
            schedule,
            event_spec=d,
            rao_blackwellize_xt=(arm != "masked"),
            rao_blackwellize_xt_num_features=1,
        )
        return model
    net = continuous_backbone(d, seed)
    if arm in ("continuous", "dequant"):
        # Same backbone, same objective (EDM denoising, x0-prediction); the only
        # difference is whether the training data is the raw values or the
        # uniform-in-bin dequantized values.
        return EDM(net, event_spec=d, num_steps=64, loss_type="x0")
    raise ValueError(arm)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def sample_arm(arm, model, bins, N, n, key, steps=64):
    if arm in ("masked", "uniform"):
        idx = jnp.asarray(
            model.as_dist((bins.edges.shape[0],), num_sample_steps=steps).sample(
                key, shape=(n,)
            )
        )
        if arm == "masked":
            # Absorbing diffusion's reverse process can leave tokens on the
            # MASK class; decode any stragglers with the model's x0 argmax over
            # the real bins (a standard absorbing-diffusion decode step).
            mask_token = N
            t0 = jnp.zeros_like(idx, dtype=jnp.float32)
            x0_arg = jnp.asarray(model.predict_x0_probs(t0, idx))[..., :N].argmax(-1)
            idx = jnp.where(idx == mask_token, x0_arg, idx)
        idx_np = np.asarray(idx)
        return bins.sample_inside(jax.random.key(0), idx), idx_np
    dist = model.as_dist((bins.edges.shape[0],), num_steps=steps)
    samples = np.asarray(dist.sample(key, shape=(n,)))
    return samples, None


def energy_distance(a, b):
    def mean_dist(x, y):
        d = jnp.sqrt(jnp.maximum(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1), 0.0))
        return d.mean()

    k = min(len(a), 400)
    a = jnp.asarray(a[:k])
    b = jnp.asarray(b[:k])
    return float(2 * mean_dist(a, b) - mean_dist(a, a) - mean_dist(b, b))


class ReferenceDensity:
    """Gaussian-KDE reference density, fitted once per target.

    Every arm is scored by the mean log-density of its own samples under this
    fixed reference, so all arms share one yardstick regardless of family
    (discrete densities have no exact logpdf here). Scott's rule bandwidth;
    higher is better.
    """

    def __init__(self, reference, scale=0.5):
        self.ref = np.asarray(reference, dtype=np.float64)  # (m, d)
        d = self.ref.shape[1]
        std = self.ref.std(axis=0)
        self.h = float(scale) * np.mean(std) * len(self.ref) ** (-1.0 / (d + 4))

    def logpdf(self, x):
        x = np.asarray(x, dtype=np.float64)[:, None, :]  # (n, 1, d)
        diff = (x - self.ref[None, :, :]) / self.h
        sq = (diff**2).sum(-1)
        logk = -0.5 * sq
        m = logk.max(axis=1, keepdims=True)
        logsum = m[:, 0] + np.log(np.exp(logk - m).mean(axis=1))
        d = self.ref.shape[1]
        return logsum - d * np.log(self.h) - 0.5 * d * np.log(2 * np.pi)

    def mean_logpdf(self, x):
        return float(np.mean(self.logpdf(x)))


def discrete_ce_minus_offset(model, bins, holdout):
    """Held-out CE of the discrete arm minus the d*log(2/N) bin-width offset.

    Diagnostic: shows that the raw cross-entropy over N bins grows with N for
    pure resolution reasons, and that the offset puts it on a continuous scale.
    Uses a mid-mask x0 posterior (the training objective as a density probe).
    """
    d = bins.edges.shape[0]
    N = bins.num_bins
    idx = jnp.asarray(bins.encode(holdout))
    t = jnp.full(idx.shape, 0.5, dtype=jnp.float32)
    probs = np.clip(
        np.asarray(model.predict_x0_probs(t, idx), dtype=np.float64)[..., :N],
        1e-9,
        None,
    )
    ce = -np.log(
        probs[np.arange(idx.shape[0])[:, None], np.arange(d)[None, :], idx]
    ).mean()
    widths = np.diff(bins.edges, axis=1)
    off = -np.mean(np.log(widths)[np.arange(d)[None, :], idx]) * d
    return float(ce), float(off)


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    target: str
    scheme: str
    N: int
    arm: str
    train_s: float
    sample_s: float
    energy: float
    nll: float | None = None
    nll_offset: float | None = None
    note: str = ""


def run_cell(target_name, scheme, N, arm, steps, batch, seed, holdout_size=2000):
    gen = TARGETS[target_name]
    data = np.asarray(gen(4000, jax.random.key(seed)))
    holdout = np.asarray(gen(holdout_size, jax.random.key(seed + 999)))
    data = clamp_to_bins(data, None)
    holdout = clamp_to_bins(holdout, None)
    bins = build_bins(data, N, scheme)
    idx = bins.encode(data)

    model = build_model(arm, target_name, bins, N, steps, seed)

    t0 = time.time()
    if arm in ("masked", "uniform"):
        model.fit(
            jax.random.key(seed + 1),
            jnp.asarray(idx),
            config=FitConfig(max_steps=steps, batch_size=batch),
        )
    elif arm == "continuous":
        model.fit(
            jax.random.key(seed + 1),
            jnp.asarray(data),
            config=FitConfig(max_steps=steps, batch_size=batch),
        )
    else:  # dequant: sample uniformly inside each bin, then fit continuously
        rng = jax.random.key(seed + 12345)
        deq = np.asarray(bins.sample_inside(rng, jnp.asarray(idx)))
        model.fit(
            jax.random.key(seed + 1),
            jnp.asarray(deq),
            config=FitConfig(max_steps=steps, batch_size=batch),
        )
    train_s = time.time() - t0

    sample_steps = 128 if arm in ("masked", "uniform") else 64
    t0 = time.time()
    samples, _ = sample_arm(
        arm, model, bins, N, holdout_size, jax.random.key(seed + 2), steps=sample_steps
    )
    sample_s = time.time() - t0

    ref = ReferenceDensity(
        np.asarray(TARGETS[target_name](20000, jax.random.key(seed + 7)))
    )
    row = Row(
        target_name,
        scheme,
        N,
        arm,
        train_s,
        sample_s,
        energy=energy_distance(samples, holdout),
        nll=ref.mean_logpdf(samples),
    )
    if arm in ("masked", "uniform"):
        _, row.nll_offset = discrete_ce_minus_offset(model, bins, holdout)
    return row


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(targets, schemes, arms, bin_counts, steps, batch, seed):
    rows = []
    for target in targets:
        for scheme in schemes:
            for N in bin_counts:
                for arm in arms:
                    row = run_cell(target, scheme, N, arm, steps, batch, seed)
                    rows.append(row)
                    print(
                        f"{target:14s} {scheme:8s} N={N:4d} {arm:10s} "
                        f"E={row.energy: .4f} ref_ll={row.nll: .3f} "
                        f"| off={row.nll_offset or 0: .2f} "
                        f"train={row.train_s:5.1f}s",
                        flush=True,
                    )
    return rows


def summarize(rows):
    print(
        "\n=== energy distance (lower better) / reference log-density (higher better) ==="
    )
    targets = sorted({r.target for r in rows})
    for target in targets:
        for scheme in sorted({r.scheme for r in rows}):
            sub = [r for r in rows if r.target == target and r.scheme == scheme]
            if not sub:
                continue
            print(f"\n[{target} / {scheme}]  cells: energy | ref_ll")
            print("arm      " + "".join(f"{('N=%d' % n):>18s}" for n in BIN_COUNTS))
            for arm in ARMS:
                cells = []
                for N in BIN_COUNTS:
                    m = [r for r in sub if r.N == N and r.arm == arm]
                    if m:
                        cells.append(f"{m[0].energy:8.4f} |{m[0].nll:7.2f}")
                    else:
                        cells.append(f"{'-':>18s}")
                print(f"{arm:9s}" + "".join(cells))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=["crescent", "skewed_copula"])
    p.add_argument("--schemes", nargs="+", default=list(BINNING))
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--bin-counts", nargs="+", type=int, default=list(BIN_COUNTS))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args(argv)

    print(f"jax devices: {jax.devices()}")
    rows = run(
        args.targets,
        args.schemes,
        args.arms,
        args.bin_counts,
        args.steps,
        args.batch,
        args.seed,
    )
    summarize(rows)
    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.__dict__ for r in rows], f, indent=2)
        print(f"\nwrote {args.output}")
    return rows


if __name__ == "__main__":
    main()
