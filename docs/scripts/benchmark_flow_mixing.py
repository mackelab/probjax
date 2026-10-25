"""Matched NSF mixer comparison on synthetic density-estimation problems.

Run from the repository root:
    JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_flow_mixing

Reports held-out NLL in nats/event, fit wall time (including compilation), and
warmed train-step/logpdf/sample timings. Conditioner initialization, datasets,
and minibatch keys are matched across mixers within each seed. Extra updates
used for train-step timing happen after the likelihood/sample measurements.
"""

import argparse
import json
import platform
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from probjax.nn import (
    CirculantDiagonalMixingConfig,
    CompositeMixingConfig,
    FlipMixingConfig,
    MLPConditionerConfig,
    PermuteMixingConfig,
    nsf,
)


def crescent(key, count):
    x = jax.random.normal(key, (count, 2))
    return x.at[:, 1].set(0.5 * x[:, 1] + 0.5 * x[:, 0] ** 2 - 1)


def spiral(key, count):
    angle_key, sign_key, noise_key = jax.random.split(key, 3)
    angle = jnp.sqrt(jax.random.uniform(angle_key, (count,))) * 3 * jnp.pi
    sign = jnp.where(jax.random.bernoulli(sign_key, 0.5, (count,)), 1, -1)
    points = jnp.stack(
        (sign * angle * jnp.cos(angle), sign * angle * jnp.sin(angle)), -1
    )
    return points / 3 + 0.08 * jax.random.normal(noise_key, (count, 2))


def rotated_banana(key, count):
    x = jax.random.normal(key, (count, 8))
    x = x.at[:, 4:].add(0.8 * (x[:, :4] ** 2 - 1))
    rotation, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(42), (8, 8)))
    return (x * jnp.linspace(0.5, 2.0, 8)) @ rotation


def rotation8():
    return jnp.linalg.qr(jax.random.normal(jax.random.key(73), (8, 8)))[0]


def correlated_gaussian(key, count):
    scales = jnp.geomspace(0.2, 2.0, 8)
    return (jax.random.normal(key, (count, 8)) * scales) @ rotation8()


def correlated_gaussian_logpdf(data):
    scales = jnp.geomspace(0.2, 2.0, 8)
    white = (data @ rotation8().T) / scales
    return jax.scipy.stats.norm.logpdf(white).sum(-1) - jnp.log(scales).sum()


def mixture_parameters():
    weights = jnp.array([0.15, 0.25, 0.35, 0.25])
    means = 1.8 * jax.random.normal(jax.random.key(74), (4, 8))
    scales = jnp.array([0.6, 1.0, 0.8, 1.2])[:, None] * jnp.linspace(0.4, 1.0, 8)
    return weights, means, scales


def gaussian_mixture(key, count):
    component_key, noise_key = jax.random.split(key)
    weights, means, scales = mixture_parameters()
    component = jax.random.categorical(component_key, jnp.log(weights), shape=(count,))
    noise = jax.random.normal(noise_key, (count, 8))
    return (means[component] + scales[component] * noise) @ rotation8()


def gaussian_mixture_logpdf(data):
    weights, means, scales = mixture_parameters()
    white = ((data @ rotation8().T)[:, None, :] - means) / scales
    component_logpdf = jax.scipy.stats.norm.logpdf(white).sum(-1) - jnp.log(scales).sum(
        -1
    )
    return jax.scipy.special.logsumexp(component_logpdf + jnp.log(weights), axis=-1)


def student_t(key, count):
    normal_key, gamma_key = jax.random.split(key)
    chi_square = 2 * jax.random.gamma(gamma_key, 2.0, shape=(count, 1))
    white = jax.random.normal(normal_key, (count, 8)) * jnp.sqrt(4.0 / chi_square)
    return (white * jnp.linspace(0.5, 2.0, 8)) @ rotation8()


def student_t_logpdf(data):
    scales = jnp.linspace(0.5, 2.0, 8)
    white = (data @ rotation8().T) / scales
    return (
        jax.scipy.special.gammaln(6.0)
        - jax.scipy.special.gammaln(2.0)
        - 4 * jnp.log(4 * jnp.pi)
        - jnp.log(scales).sum()
        - 6 * jnp.log1p(jnp.sum(white**2, axis=-1) / 4)
    )


def correlated_lognormal(key, count):
    log_values = 0.55 * correlated_gaussian(key, count) + jnp.linspace(-0.4, 0.4, 8)
    return jnp.exp(log_values)


def correlated_lognormal_logpdf(data):
    log_values = jnp.log(data)
    latent = (log_values - jnp.linspace(-0.4, 0.4, 8)) / 0.55
    return correlated_gaussian_logpdf(latent) - 8 * jnp.log(0.55) - log_values.sum(-1)


def beta_parameters():
    return (
        jnp.array([1.2, 2, 3, 5, 1.5, 4, 2.5, 8]),
        jnp.array([5, 3, 2, 1.3, 4, 2, 6, 2]),
    )


def bounded_beta(key, count):
    alpha, beta = beta_parameters()
    return jax.random.beta(key, alpha, beta, shape=(count, 8))


def bounded_beta_logpdf(data):
    alpha, beta = beta_parameters()
    return jax.scipy.stats.beta.logpdf(data, alpha, beta).sum(-1)


def funnel(key, count):
    scale_key, noise_key = jax.random.split(key)
    log_variance = 1.5 * jax.random.normal(scale_key, (count, 1))
    features = jax.random.normal(noise_key, (count, 7)) * jnp.exp(log_variance / 2)
    return jnp.concatenate((log_variance, features), axis=-1)


def funnel_logpdf(data):
    log_variance = data[:, 0]
    white = data[:, 1:] * jnp.exp(-log_variance[:, None] / 2)
    return (
        jax.scipy.stats.norm.logpdf(log_variance, scale=1.5)
        + jax.scipy.stats.norm.logpdf(white).sum(-1)
        - 7 * log_variance / 2
    )


TARGETS = {
    "crescent2": crescent,
    "spiral2": spiral,
    "rotated_banana8": rotated_banana,
    "correlated_gaussian8": correlated_gaussian,
    "gaussian_mixture8": gaussian_mixture,
    "student_t8": student_t,
    "correlated_lognormal8": correlated_lognormal,
    "bounded_beta8": bounded_beta,
    "funnel8": funnel,
}
REALISTIC_TARGETS = [
    "correlated_gaussian8",
    "gaussian_mixture8",
    "student_t8",
    "correlated_lognormal8",
    "bounded_beta8",
    "funnel8",
]
REFERENCE_LOGPDF = {
    "correlated_gaussian8": correlated_gaussian_logpdf,
    "gaussian_mixture8": gaussian_mixture_logpdf,
    "student_t8": student_t_logpdf,
    "correlated_lognormal8": correlated_lognormal_logpdf,
    "bounded_beta8": bounded_beta_logpdf,
    "funnel8": funnel_logpdf,
}
MIXERS = {
    "flip": FlipMixingConfig,
    "permute": PermuteMixingConfig,
    "circulant_diagonal": CirculantDiagonalMixingConfig,
    "permute_cd": lambda: CompositeMixingConfig((
        PermuteMixingConfig(),
        CirculantDiagonalMixingConfig(),
    )),
    "flip_cd": lambda: CompositeMixingConfig((
        FlipMixingConfig(),
        CirculantDiagonalMixingConfig(),
    )),
}
MIXERS["flip_wide_damped"] = MIXERS["flip"]
MIXERS["flip_cd_wide_damped"] = MIXERS["flip_cd"]
WIDE_DAMPED_SPLINE = {
    "x_min": -6.0,
    "x_max": 6.0,
    "y_min": -6.0,
    "y_max": 6.0,
    "tail_logit_scale": 0.1,
}
SPLINE_OPTIONS = {
    "flip_wide_damped": WIDE_DAMPED_SPLINE,
    "flip_cd_wide_damped": WIDE_DAMPED_SPLINE,
}


def build_flow(features, seed, mixing, args, spline_options=None):
    return nsf(
        features,
        args.transforms,
        nnx.Rngs(seed),
        num_bins=args.bins,
        conditioner=MLPConditionerConfig(hidden_dims=(64, 64)),
        mixing=mixing,
        # Keep the original benchmark baselines explicit after default changes.
        **{"tail_logit_scale": 1.0, **(spline_options or {})},
    )


def timed(call, repeats=5):
    jax.block_until_ready(call())
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(call())
        durations.append(time.perf_counter() - start)
    return 1000 * float(np.median(durations))


@nnx.jit
def train_step(flow, optimizer, batch):
    loss, grads = nnx.value_and_grad(lambda model: model.loss(None, batch))(flow)
    optimizer.update(flow, grads)
    return loss


def summarize(rows, targets, mixers):
    print("\nSummary: mean held-out NLL ± sample standard deviation", flush=True)
    for target in targets:
        for mixer in mixers:
            selected = [
                row for row in rows if row["target"] == target and row["mixer"] == mixer
            ]
            if not selected:
                print(f"{target:24} {mixer:22} no completed measurements", flush=True)
                continue
            values = [row["nll"] for row in selected]
            print(
                f"{target:24} {mixer:22} {np.mean(values):.4f} ± "
                f"{np.std(values, ddof=1) if len(values) > 1 else 0:.4f}; "
                f"train {np.mean([r['train_step_ms'] for r in selected]):.2f} ms; "
                f"logpdf {np.mean([r['logpdf_ms'] for r in selected]):.2f} ms; "
                f"sample {np.mean([r['sample_ms'] for r in selected]):.2f} ms",
                flush=True,
            )
            if selected and selected[0].get("reference_nll") is not None:
                print(
                    f"  reference {np.mean([r['reference_nll'] for r in selected]):.4f}; "
                    f"excess {np.mean([r['excess_nll'] for r in selected]):.4f}; "
                    f"worst per seed {[round(r['worst_nll'], 2) for r in selected]}; "
                    f"support violations {[r['sample_support_violation_fraction'] for r in selected]}",
                    flush=True,
                )


def run(args):
    rows = []
    output = Path(args.output)
    if args.resume:
        saved = json.loads(output.read_text())
        for name in (
            "suite",
            "steps",
            "learning_rate",
            "seeds",
            "transforms",
            "bins",
            "batch_size",
            "train_size",
            "test_size",
            "targets",
            "mixers",
        ):
            if saved["settings"][name] != getattr(args, name):
                raise ValueError(f"Cannot resume with a different {name}")
        rows = saved["rows"]
    completed = {(row["target"], row["seed"], row["mixer"]) for row in rows}
    metadata = {
        "settings": vars(args),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }
    for target_name in args.targets:
        sampler = TARGETS[target_name]
        for seed in range(args.seeds):
            if all((target_name, seed, mixer) in completed for mixer in args.mixers):
                continue
            train_key, test_key, fit_key = jax.random.split(
                jax.random.key(seed + 100), 3
            )
            train = sampler(train_key, args.train_size)
            test = sampler(test_key, args.test_size)
            reference = REFERENCE_LOGPDF.get(target_name)
            reference_nll = (
                None if reference is None else float(-jnp.mean(reference(test)))
            )
            features = train.shape[-1]
            # Always use flip-baseline initialization, even for follow-up runs
            # that select only one mixer. NNX states contain live Variables.
            baseline = build_flow(features, seed, FlipMixingConfig(), args)
            initial_states = [
                jax.tree.map(lambda value: value.copy(), nnx.state(layer))
                for layer in list(baseline.transformation.layers)[::2]
            ]
            for mixer_name in args.mixers:
                if (target_name, seed, mixer_name) in completed:
                    continue
                print(f"Starting {target_name}, seed {seed}, {mixer_name}", flush=True)
                flow = build_flow(
                    features,
                    seed,
                    MIXERS[mixer_name](),
                    args,
                    SPLINE_OPTIONS.get(mixer_name),
                )
                conditioners = list(flow.transformation.layers)[::2]
                for layer, state in zip(conditioners, initial_states, strict=True):
                    nnx.update(layer, jax.tree.map(lambda value: value.copy(), state))
                for layer, state in zip(conditioners, initial_states, strict=True):
                    assert all(
                        jnp.array_equal(actual, expected)
                        for actual, expected in zip(
                            jax.tree.leaves(nnx.state(layer)),
                            jax.tree.leaves(state),
                            strict=True,
                        )
                    ), "Conditioner initialization differs across mixers"
                parameter_count = sum(
                    value.size for value in jax.tree.leaves(nnx.state(flow, nnx.Param))
                )
                start = time.perf_counter()
                losses = flow.fit(
                    fit_key,
                    train,
                    num_steps=args.steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    clip_norm=10.0,
                ).losses
                jax.block_until_ready(losses)
                fit_seconds = time.perf_counter() - start
                logpdf = jax.jit(jax.vmap(flow._logpdf))
                values = logpdf(test)
                nll = float(-jnp.mean(values))
                individual_nll = -np.asarray(values)
                training_nll = float(-jnp.mean(logpdf(train)))
                logpdf_ms = timed(partial(logpdf, test))
                distribution = flow.as_dist()
                distribution.compile()
                sample = jax.jit(partial(distribution.sample, shape=(args.test_size,)))
                samples = sample(test_key)
                sample_ms = timed(partial(sample, test_key))
                support_violations = None
                if target_name == "correlated_lognormal8":
                    support_violations = float(jnp.mean(jnp.any(samples <= 0, axis=-1)))
                elif target_name == "bounded_beta8":
                    support_violations = float(
                        jnp.mean(jnp.any((samples <= 0) | (samples >= 1), axis=-1))
                    )

                optimizer = nnx.Optimizer(
                    flow,
                    optax.chain(
                        optax.clip_by_global_norm(10), optax.adam(args.learning_rate)
                    ),
                    wrt=nnx.Param,
                )
                batch = train[: args.batch_size]
                train_step_ms = timed(
                    partial(train_step, flow, optimizer, batch), repeats=10
                )
                row = {
                    "target": target_name,
                    "seed": seed,
                    "mixer": mixer_name,
                    "parameters": parameter_count,
                    "nll": nll,
                    "reference_nll": reference_nll,
                    "excess_nll": None
                    if reference_nll is None
                    else nll - reference_nll,
                    "train_nll": training_nll,
                    "median_nll": float(np.median(individual_nll)),
                    "p99_nll": float(np.quantile(individual_nll, 0.99)),
                    "worst_nll": float(np.max(individual_nll)),
                    "mean_without_worst": float(np.sort(individual_nll)[:-1].mean()),
                    "finite_logpdf": bool(jnp.all(jnp.isfinite(values))),
                    "sample_support_violation_fraction": support_violations,
                    "spline_options": {
                        "tail_logit_scale": 1.0,
                        **SPLINE_OPTIONS.get(mixer_name, {}),
                    },
                    "finite_losses": bool(jnp.all(jnp.isfinite(losses))),
                    "finite_samples": bool(jnp.all(jnp.isfinite(samples))),
                    "fit_seconds": fit_seconds,
                    "train_step_ms": train_step_ms,
                    "logpdf_ms": logpdf_ms,
                    "sample_ms": sample_ms,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                output.write_text(json.dumps(metadata, indent=2) + "\n")
                jax.clear_caches()
    summarize(rows, args.targets, args.mixers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["toy", "realistic"], default="toy")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--transforms", type=int, default=5)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--targets", nargs="+", choices=TARGETS)
    parser.add_argument("--mixers", nargs="+", choices=MIXERS)
    parser.add_argument("--output", default="docs/benchmarks/flow_mixing.json")
    args = parser.parse_args()
    if args.summary_only:
        saved = json.loads(Path(args.output).read_text())
        summarize(
            saved["rows"], saved["settings"]["targets"], saved["settings"]["mixers"]
        )
    else:
        if args.targets is None:
            args.targets = (
                REALISTIC_TARGETS
                if args.suite == "realistic"
                else [
                    "crescent2",
                    "spiral2",
                    "rotated_banana8",
                ]
            )
        if args.mixers is None:
            args.mixers = (
                ["flip", "flip_cd", "flip_wide_damped", "flip_cd_wide_damped"]
                if args.suite == "realistic"
                else [
                    "flip",
                    "permute",
                    "circulant_diagonal",
                    "permute_cd",
                    "flip_cd",
                ]
            )
        run(args)
