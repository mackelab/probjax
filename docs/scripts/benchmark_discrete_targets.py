"""Do discrete-target models work, and can continuous-then-quantize match them?

The companion ``benchmark_discretization.py`` studies quantization of
*continuous* targets. This script uses *intrinsically discrete* targets and asks
the practical question: if we model a discrete distribution with a continuous
model and then quantize (round) the samples, how close do we get to the true PMF,
versus modelling the tokens directly with discrete diffusion?

Targets (all simple, fixed finite support):
  * ``poisson``        -- univariate Poisson(3), truncated to k=0..K-1.
  * ``corr_poisson``   -- bivariate Poisson(3) margins coupled by a Gaussian
                          copula (correlation 0.7): simple but genuinely
                          dependent, not a product.
  * ``grid``           -- 2D categorical on a 4x4 grid supported only on
                          cells where (i + j) is even: non-product dependence.

Arms:
  * ``masked``         -- absorbing-mask discrete diffusion over the K token
                          classes (the boxed masked-only objective).
  * ``uniform``        -- uniform-corruption discrete diffusion over K classes.
  * ``continuous``     -- EDM on the integer values treated as real numbers.
  * ``dequant``        -- EDM on the values plus U(-0.5, 0.5) jitter; samples
                          are rounded back to integers.

Metric: total variation between the empirical sample PMF and a reference PMF.
For 2D targets we also report the marginal TV and the correlation error.

Run:
    JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_discrete_targets --steps 2000
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from docs.scripts.benchmark_discretization import (
    CategoricalMLP,
    MaskedDiffusion,
)
from probjax.nn import EDM, FitConfig
from probjax.nn.generative.discrete.model import (
    MultinomialDiffusion,
    MultinomialDiffusionSchedule,
)
from probjax.nn.nets.time import TimeMLP
from probjax.stats import norm

ARMS = ("masked", "uniform", "continuous", "dequant")


# --------------------------------------------------------------------------- #
# Targets: return integer arrays (n, d) over a fixed support 0..K-1
# --------------------------------------------------------------------------- #
def poisson_target(num_points, key, K=20, lam=3.0):
    key = jax.random.key(0) if key is None else key
    x = jax.random.poisson(key, lam, (int(num_points), 1))
    return jnp.clip(x, 0, K - 1)


def corr_poisson_target(num_points, key, K=20, lam=3.0, rho=0.7):
    """Bivariate Poisson via Gaussian copula: dependence without a product form."""
    key = jax.random.key(0) if key is None else key
    k1, k2 = jax.random.split(key)
    n = int(num_points)
    cov = jnp.array([[1.0, rho], [rho, 1.0]])
    z = jax.random.multivariate_normal(k1, jnp.zeros(2), cov, shape=(n,))
    u = jnp.clip(norm.cdf(z), 1e-6, 1 - 1e-6)  # copula uniforms
    # Poisson.ppf(u, lam) elementwise -> inverse-CDF transform of the uniforms.
    from probjax.stats import poisson

    x = poisson.ppf(u, lam)
    return jnp.clip(x, 0, K - 1)


def grid_target(num_points, key, side=4):
    """2D categorical supported only on (i + j) even cells: non-product support."""
    key = jax.random.key(0) if key is None else key
    cells = jnp.array([
        [i, j] for i, j in itertools.product(range(side), repeat=2) if (i + j) % 2 == 0
    ])
    ids = jax.random.randint(key, (int(num_points),), 0, cells.shape[0])
    return cells[ids]


TARGETS = {
    "poisson": (poisson_target, 1),
    "corr_poisson": (corr_poisson_target, 2),
    "grid": (grid_target, 2),
}


def support_size(target, side=4, K=20):
    if target == "grid":
        return side  # per-dimension cardinality
    return K


# --------------------------------------------------------------------------- #
# Backbones
# --------------------------------------------------------------------------- #
class CatNet(nnx.Module):
    """Discrete-arm net; same shape contract as CategoricalMLP but standalone."""

    def __init__(self, dim, num_classes, hidden=96, rngs=None):
        self.inner = CategoricalMLP(dim, num_classes, hidden=hidden, rngs=rngs)
        self.dim = dim
        self.num_classes = num_classes

    def __call__(self, t, x, *a, **k):
        return self.inner(t, x, *a, **k)


def continuous_dim(target):
    return TARGETS[target][1]


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(arm, target, steps, seed=0):
    dim = continuous_dim(target)
    K = support_size(target)
    if arm in ("masked", "uniform"):
        base = (
            jnp.zeros(K + 1).at[-1].set(1.0)
            if arm == "masked"
            else jnp.ones(K + 1) / (K + 1)
        )
        schedule = MultinomialDiffusionSchedule.from_cosine(
            num_steps=64, num_classes=K + 1, base_probs=base
        )
        cls = MaskedDiffusion if arm == "masked" else MultinomialDiffusion
        return cls(
            CatNet(dim, K + 1, rngs=nnx.Rngs(seed)),
            schedule,
            event_spec=dim,
            rao_blackwellize_xt=(arm != "masked"),
            rao_blackwellize_xt_num_features=1,
        )
    return EDM(
        TimeMLP(dim, hidden_dim=128, depth=3, rngs=nnx.Rngs(seed)),
        event_spec=dim,
        num_steps=64,
        loss_type="x0",
    )


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_arm(arm, model, target, n, key, steps=128):
    K = support_size(target)
    dim = continuous_dim(target)
    if arm in ("masked", "uniform"):
        idx = jnp.asarray(
            model.as_dist((dim,), num_sample_steps=steps).sample(key, shape=(n,))
        )
        if arm == "masked":
            t0 = jnp.zeros_like(idx, dtype=jnp.float32)
            arg = jnp.asarray(model.predict_x0_probs(t0, idx))[..., :K].argmax(-1)
            idx = jnp.where(idx == K, arg, idx)
        return np.asarray(idx)
    x = np.asarray(model.as_dist((dim,), num_steps=steps).sample(key, shape=(n,)))
    # Quantize continuous output back to the integer support.
    return np.rint(x).astype(np.int64)  # round to nearest integer


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def empirical_pmf(samples, target, K=20):
    dim = continuous_dim(target)
    if dim == 1:
        counts = np.bincount(np.clip(samples[:, 0], 0, K - 1), minlength=K)
        return counts / counts.sum()
    side = support_size(target) if target == "grid" else K
    # joint on the side x side grid
    i = np.clip(samples[:, 0], 0, side - 1)
    j = np.clip(samples[:, 1], 0, side - 1)
    flat = i * side + j
    counts = np.bincount(flat, minlength=side * side)
    return counts / counts.sum(), (i, j)


def reference_pmf(target, key, n=400_000):
    gen = TARGETS[target][0]
    s = np.asarray(gen(n, key))
    return empirical_pmf(s, target)


def total_variation(p, q):
    if isinstance(p, tuple):
        p = p[0]
    if isinstance(q, tuple):
        q = q[0]
    return 0.5 * float(np.abs(p - q).sum())


def corr_error(samples, ref_i, ref_j, K):
    d = TARGETS["corr_poisson"][1]
    del d

    def corr(i, j):
        i = i.astype(np.float64)
        j = j.astype(np.float64)
        i = i - i.mean()
        j = j - j.mean()
        denom = np.sqrt((i**2).mean() * (j**2).mean())
        return float((i * j).mean() / max(denom, 1e-9))

    return abs(corr(samples[:, 0], samples[:, 1]) - corr(ref_i, ref_j))


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    target: str
    arm: str
    steps: int
    train_s: float
    tv: float
    tv_marg: float | None = None
    corr_err: float | None = None
    frc_off_support: float | None = None


def run_cell(target, arm, steps, batch, seed, holdout=4000):
    gen = TARGETS[target][0]
    data = np.asarray(gen(4000, jax.random.key(seed)))
    model = build_model(arm, target, steps, seed)
    K = support_size(target)
    dim = continuous_dim(target)

    train_x = jnp.asarray(data, dtype=jnp.float32)
    if arm in ("masked", "uniform"):
        train_x = jnp.asarray(data, dtype=jnp.int32)
    elif arm == "dequant":
        rng = jax.random.key(seed + 55)
        jitter = jax.random.uniform(rng, data.shape) - 0.5
        train_x = jnp.asarray(data, dtype=jnp.float32) + jitter
    # ``continuous`` trains on raw integers as real values.

    t0 = time.time()
    model.fit(
        jax.random.key(seed + 1),
        train_x,
        config=FitConfig(max_steps=steps, batch_size=batch),
    )
    train_s = time.time() - t0

    samples = sample_arm(arm, model, target, holdout, jax.random.key(seed + 2))

    ref = reference_pmf(target, jax.random.key(seed + 7))
    emp = empirical_pmf(samples, target)
    tv = total_variation(emp, ref)

    row = Row(target=target, arm=arm, steps=steps, train_s=train_s, tv=tv)
    if dim == 2:
        pi, (ri, rj) = emp
        qi, (si, sj) = ref
        side = support_size(target) if target == "grid" else K
        # marginal TV (dim 0)
        m0 = np.bincount(ri, minlength=side).astype(float)
        m0 /= m0.sum()
        q0 = np.bincount(si, minlength=side).astype(float)
        q0 /= q0.sum()
        row.tv_marg = 0.5 * float(np.abs(m0 - q0).sum())
        row.corr_err = corr_error(samples, si, sj, side)
        # fraction of samples off the true support (grid: odd cells)
        if target == "grid":
            row.frc_off_support = float(((ri + rj) % 2 != 0).mean())
    return row


def summarize(rows):
    for target in sorted({r.target for r in rows}):
        print(f"\n[{target}] total variation vs reference PMF (lower better)")
        print(
            f"{'arm':10s}{'train_s':>9s}{'TV':>10s}{'TV_marg':>10s}"
            f"{'corr_err':>10s}{'off_supp':>10s}"
        )
        for arm in ARMS:
            m = [r for r in rows if r.target == target and r.arm == arm]
            if not m:
                continue
            r = m[0]
            fmt = lambda v: f"{v:10.4f}" if v is not None else f"{'-':>10s}"
            print(
                f"{arm:10s}{r.train_s:9.1f}{r.tv:10.4f}"
                f"{fmt(r.tv_marg)}{fmt(r.corr_err)}{fmt(r.frc_off_support)}"
            )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=list(TARGETS))
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args(argv)

    print(f"jax devices: {jax.devices()}")
    rows = []
    for target in args.targets:
        for arm in args.arms:
            row = run_cell(target, arm, args.steps, args.batch, args.seed)
            rows.append(row)
            print(
                f"{target:14s} {arm:10s} TV={row.tv:.4f} "
                f"corr_err={row.corr_err if row.corr_err is None else round(row.corr_err, 4)} "
                f"off_supp={row.frc_off_support} train={row.train_s:.1f}s",
                flush=True,
            )
    summarize(rows)
    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.__dict__ for r in rows], f, indent=2)
        print(f"\nwrote {args.output}")
    return rows


if __name__ == "__main__":
    main()
