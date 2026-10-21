"""Benchmark all concrete diffusion config types in controlled panels.

Run from the repository root with JAX_PLATFORMS=cpu:
    python -m docs.scripts.benchmark_diffusion_configs --phase oracle
    python -m docs.scripts.benchmark_diffusion_configs --phase learned
    python -m docs.scripts.benchmark_diffusion_configs --phase categorical

Each case is checkpointed; --resume skips completed cases with identical settings.
Oracle and learned-model sample metrics are kept separate. No config defaults
are changed. DDIM's SDE interface is evaluated but remains deterministic (eta=0).
"""

import argparse
import itertools
import json
import platform
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from probjax.nn import MLP, TimeMLP, FitConfig
from probjax.nn.generative.diffusion import EDM, VE, VP, CosineDM, DiffusionDenoiser
from probjax.nn.generative.diffusion import config as C
from probjax.nn.generative.discrete import config as D
from probjax.nn.generative.discrete.model import MultinomialDiffusion

PRESETS = {"EDM": EDM, "VE": VE, "VP": VP, "CosineDM": CosineDM}
SOLVERS = {
    "BaseSolverConfig": C.BaseSolverConfig,
    "EDMSolverConfig": C.EDMSolverConfig,
    "DDIMSolverConfig": C.DDIMSolverConfig,
    "VSolverConfig": C.VSolverConfig,
}
TRAININGS = {
    "EDMTrainingConfig": C.EDMTrainingConfig,
    "SigmaEffEDMTrainingConfig": C.SigmaEffEDMTrainingConfig,
    "LogSNRTrainingConfig": C.LogSNRTrainingConfig,
    "UniformTTrainingConfig": C.UniformTTrainingConfig,
}
CATEGORICAL_KINDS = ("linear", "cosine", "sigmoid", "logsnr")
CLASS_PROBS = jnp.array([0.05, 0.15, 0.3, 0.5])
ROTATION = jnp.array([[0.8, -0.6], [0.6, 0.8]])
GAUSSIAN_VARIANCES = jnp.array([0.25, 1.75])
MIXTURE_CENTERS = jnp.array([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])


def target_sample(target, key, count):
    if target == "gaussian":
        return (
            jax.random.normal(key, (count, 2)) * jnp.sqrt(GAUSSIAN_VARIANCES)
        ) @ ROTATION.T
    component_key, noise_key = jax.random.split(key)
    component = jax.random.randint(component_key, (count,), 0, 4)
    return MIXTURE_CENTERS[component] + 0.5 * jax.random.normal(noise_key, (count, 2))


def target_std(target):
    return 1.0 if target == "gaussian" else float(np.sqrt(4.25))


def posterior_mean(target, alpha, sigma, value):
    """Exact clean-data posterior mean under physical Gaussian corruption."""
    if target == "gaussian":
        gain = alpha * GAUSSIAN_VARIANCES / (alpha**2 * GAUSSIAN_VARIANCES + sigma**2)
        return ((value @ ROTATION) * gain) @ ROTATION.T
    variance = alpha**2 * 0.25 + sigma**2
    residual = value[..., None, :] - alpha[..., None] * MIXTURE_CENTERS
    logits = -jnp.sum(residual**2, axis=-1) / (2 * variance)
    probabilities = jax.nn.softmax(logits, axis=-1)
    mean = probabilities @ MIXTURE_CENTERS
    gain = alpha * 0.25 / variance
    return gain * value + (1 - gain * alpha) * mean


class Zero(nnx.Module):
    def __call__(self, t, x, **kwargs):
        return jnp.zeros_like(x)


class OracleDenoiser(DiffusionDenoiser):
    def __init__(self, target, **kwargs):
        self.target = target
        super().__init__(**kwargs)

    def denoise(self, t, x, *args, **kwargs):
        return posterior_mean(self.target, self.scale_fn(t), self.std_fn(t), x)


def make_model(preset, target, seed, steps, oracle=False, training=None, head="x0"):
    net = (
        Zero()
        if oracle
        else TimeMLP(
            2,
            hidden_dim=32,
            depth=1,
            time_embed_dim=32,
            fourier_dim=16,
            rngs=nnx.Rngs(seed),
        )
    )
    model = PRESETS[preset](
        net, event_spec=2, std0=target_std(target), num_steps=steps, loss_type=head
    )
    if training is not None:
        kwargs = dict(
            t_min=model.train_cfg.t_min, t_max=model.train_cfg.t_max, loss_type=head
        )
        if training in {"SigmaEffEDMTrainingConfig", "LogSNRTrainingConfig"}:
            kwargs["schedule"] = model.schedule
        model.train_cfg = TRAININGS[training](**kwargs)
    if oracle:
        model = OracleDenoiser(
            target,
            net=net,
            schedule=model.schedule,
            precond=model.precond,
            train_cfg=model.train_cfg,
            solver_cfg=model.solver_cfg,
            std0=target_std(target),
            event_spec=2,
        )
    return model


def sample_metrics(samples, reference, scale):
    samples, reference = (
        np.asarray(samples, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )
    finite = np.isfinite(samples).all(axis=-1)
    metrics = {"finite_fraction": float(finite.mean())}
    if not finite.all():
        return {
            **metrics,
            "swd2": None,
            "covariance_relative_error": None,
            "mean_error": None,
        }
    directions = np.random.default_rng(10).normal(size=(2, 32))
    directions /= np.linalg.norm(directions, axis=0)
    delta = np.sort(samples @ directions, axis=0) - np.sort(
        reference @ directions, axis=0
    )
    covariance = np.cov(reference, rowvar=False)
    metrics.update({
        "swd2": float(np.sqrt(np.mean(delta**2)) / scale),
        "covariance_relative_error": float(
            np.linalg.norm(np.cov(samples, rowvar=False) - covariance)
            / np.linalg.norm(covariance)
        ),
        "mean_error": float(
            np.linalg.norm(samples.mean(0) - reference.mean(0)) / scale
        ),
        "max_abs_sample": float(np.abs(samples).max()),
    })
    return metrics


def measure_sampler(model, mode, key, count, repeats):
    distribution = model.as_dist(mode=mode)
    call = jax.jit(partial(distribution.sample, shape=(count,)))
    start = time.perf_counter()
    distribution.compile()
    values = jax.block_until_ready(call(key))
    cold = time.perf_counter() - start
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(call(key))
        durations.append(time.perf_counter() - start)
    return values, {"cold_seconds": cold, "warm_ms": 1000 * float(np.median(durations))}


@nnx.jit
def denoise(model, times, values):
    return model.denoise(times, values)


def denoising_metrics(model, target, seed):
    clean = target_sample(target, jax.random.key(seed + 200), 256)
    noise = jax.random.normal(jax.random.key(seed + 201), clean.shape)
    excess, clean_error = [], []
    for sigma_eff in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        times = jnp.full((256, 1), model.inv_sigma_eff(sigma_eff))
        alpha, sigma = model.scale_fn(times), model.std_fn(times)
        noisy = alpha * clean + sigma * noise
        predicted = denoise(model, times, noisy)
        exact = posterior_mean(target, alpha, sigma, noisy)
        excess.append(
            float(jnp.mean((predicted - exact) ** 2) / target_std(target) ** 2)
        )
        clean_error.append(
            float(jnp.mean((predicted - clean) ** 2) / target_std(target) ** 2)
        )
    return {
        "denoising_excess_mse": float(np.mean(excess)),
        "excess_mse_by_sigma": excess,
        "clean_mse_by_sigma": clean_error,
    }


class CategoricalNet(nnx.Module):
    def __init__(self, seed):
        self.net = MLP([5, 32, 32, 4], activation=jax.nn.silu, rngs=nnx.Rngs(seed))

    def __call__(self, t, x, **kwargs):
        t = jnp.asarray(t)
        t = t.reshape(t.shape + (1,) * (x.ndim - t.ndim))
        t = jnp.broadcast_to(t, x.shape[:-1] + (1,))
        return self.net(jnp.concatenate((x, t), axis=-1))


def categorical_posterior(schedule, t, tokens):
    alpha = schedule.alpha_bar(t)[..., None]
    likelihood = (
        alpha * jax.nn.one_hot(tokens, 4)
        + (1 - alpha) * schedule.base_probs[tokens][..., None]
    )
    probabilities = likelihood * CLASS_PROBS
    return probabilities / probabilities.sum(-1, keepdims=True)


class OracleCategorical(MultinomialDiffusion):
    def predict_x0_probs(self, t, x_t, *args, **kwargs):
        return categorical_posterior(self.schedule, t, x_t)


class MatchedTerminalSchedule(D.MultinomialDiffusionSchedule):
    def prior_sample(self, rng, shape):
        alpha = self.alpha_bar(jnp.asarray(self.t_max, jnp.float32))
        probabilities = alpha * CLASS_PROBS + (1 - alpha) * self.base_probs
        return jax.random.categorical(rng, jnp.log(probabilities), shape=shape)


def make_categorical_schedule(kind, steps, matched=False, legacy=False):
    cls = MatchedTerminalSchedule if matched else D.MultinomialDiffusionSchedule
    kwargs = {}
    if legacy and kind in {"linear", "sigmoid"}:
        kwargs = {"beta_start": 1e-4, "beta_end": 0.1}
    elif legacy and kind == "cosine":
        kwargs = {"max_beta": 0.999}
    return getattr(cls, "from_" + kind)(steps, 4, **kwargs)


def categorical_metrics(samples):
    samples = np.asarray(samples).reshape(-1)
    frequencies = np.bincount(samples, minlength=4) / len(samples)
    return {
        "frequencies": frequencies.tolist(),
        "total_variation": float(
            0.5 * np.abs(frequencies - np.asarray(CLASS_PROBS)).sum()
        ),
    }


def categorical_validation(model, seed):
    clean = jax.random.categorical(
        jax.random.key(seed + 210), jnp.log(CLASS_PROBS), shape=(1024, 1)
    )
    times = jax.random.uniform(
        jax.random.key(seed + 211), (1024, 1), minval=0.001, maxval=0.999
    )
    noisy = model.schedule.q_sample(jax.random.key(seed + 212), clean, times)
    probabilities = model.predict_x0_probs(times, noisy)
    exact = categorical_posterior(model.schedule, times, noisy)
    selected = jnp.take_along_axis(probabilities, clean[..., None], axis=-1)
    selected_exact = jnp.take_along_axis(exact, clean[..., None], axis=-1)
    ce = float(-jnp.log(jnp.maximum(selected, 1e-12)).mean())
    reference_ce = float(-jnp.log(jnp.maximum(selected_exact, 1e-12)).mean())
    return {
        "validation_ce": ce,
        "bayes_ce": reference_ce,
        "excess_ce": ce - reference_ce,
    }


def cases(args):
    if args.phase == "oracle":
        for target, preset, solver, mode, steps in itertools.product(
            ("gaussian", "mixture"), PRESETS, SOLVERS, ("ode", "sde"), args.sample_steps
        ):
            yield dict(
                target=target, preset=preset, solver=solver, mode=mode, steps=steps
            )
    elif args.phase == "learned":
        for preset, training, head, seed in itertools.product(
            PRESETS, TRAININGS, ("x0", "eps", "v"), range(args.seeds)
        ):
            # Direct log-normal physical-time sampling is an EDM sigma-time config.
            if training == "EDMTrainingConfig" and preset != "EDM":
                continue
            yield dict(
                target="mixture", preset=preset, training=training, head=head, seed=seed
            )
    else:
        for kind, steps, matched in itertools.product(
            CATEGORICAL_KINDS, args.sample_steps, (False, True)
        ):
            yield dict(kind=kind, steps=steps, oracle=True, matched_terminal=matched)
        for kind, training, seed in itertools.product(
            CATEGORICAL_KINDS, ("uniform", "importance"), range(args.seeds)
        ):
            yield dict(
                kind=kind,
                steps=args.learned_sample_steps,
                oracle=False,
                training=training,
                seed=seed,
            )


def run_case(case, args):
    if args.phase == "oracle":
        model = make_model(
            case["preset"], case["target"], 0, case["steps"], oracle=True
        )
        model.set_solver_cfg(
            SOLVERS[case["solver"]](schedule=model.schedule, num_steps=case["steps"])
        )
        key = jax.random.key(42)
        samples, timing = measure_sampler(
            model, case["mode"], key, args.samples, args.repeats
        )
        reference = target_sample(case["target"], jax.random.key(43), args.samples)
        independent = target_sample(case["target"], jax.random.key(44), args.samples)
        return {
            **timing,
            **sample_metrics(samples, reference, target_std(case["target"])),
            "reference_sample_floor": sample_metrics(
                independent, reference, target_std(case["target"])
            )["swd2"],
            "sigma_eff_min": float(jnp.squeeze(model.sigma_eff(model.train_cfg.t_min))),
            "sigma_eff_max": float(jnp.squeeze(model.sigma_eff(model.train_cfg.t_max))),
            "ode_method": model.solver_cfg.ode_method,
            "sde_method": model.solver_cfg.sde_method,
            "deterministic_sde_interface": case["mode"] == "sde"
            and case["solver"] == "DDIMSolverConfig",
        }
    if args.phase == "learned":
        model = make_model(
            case["preset"],
            case["target"],
            case["seed"],
            args.learned_sample_steps,
            training=case["training"],
            head=case["head"],
        )
        data = target_sample(case["target"], jax.random.key(case["seed"] + 100), 4096)
        start = time.perf_counter()
        losses = model.fit(
            jax.random.key(case["seed"] + 101),
            data,
            config=FitConfig(
                max_steps=args.train_steps,
                batch_size=128,
                learning_rate=1e-3,
                clip_norm=10.0,
            ),
        ).losses
        jax.block_until_ready(losses)
        training_time = time.perf_counter() - start
        model.set_solver_cfg(
            C.EDMSolverConfig(
                schedule=model.schedule, num_steps=args.learned_sample_steps
            )
        )
        samples, timing = measure_sampler(
            model,
            "ode",
            jax.random.key(case["seed"] + 102),
            args.learned_samples,
            args.repeats,
        )
        reference = target_sample(
            case["target"], jax.random.key(case["seed"] + 103), args.learned_samples
        )
        independent = target_sample(
            case["target"], jax.random.key(case["seed"] + 104), args.learned_samples
        )
        return {
            "training_seconds": training_time,
            "finite_losses": bool(jnp.isfinite(losses).all()),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            **timing,
            **sample_metrics(samples, reference, target_std(case["target"])),
            "reference_sample_floor": sample_metrics(
                independent, reference, target_std(case["target"])
            )["swd2"],
            **denoising_metrics(model, case["target"], case["seed"]),
        }
    schedule = make_categorical_schedule(
        case["kind"],
        case["steps"],
        matched=case.get("matched_terminal", False),
        legacy=args.legacy_categorical,
    )
    model_class = OracleCategorical if case["oracle"] else MultinomialDiffusion
    seed = case.get("seed", 0)
    train_cfg = D.UniformContinuousTimeTrainingConfig(num_steps=case["steps"])
    if case.get("training") == "importance":
        train_cfg = D.ImportanceContinuousTimeTrainingConfig(schedule=schedule)
    model = model_class(
        CategoricalNet(seed), schedule, event_spec=(1,), train_cfg=train_cfg
    )
    metrics = {
        "terminal_alpha_bar": float(
            schedule.alpha_bar(jnp.asarray(schedule.t_max, jnp.float32))
        )
    }
    if not case["oracle"]:
        data = jax.random.categorical(
            jax.random.key(seed + 100), jnp.log(CLASS_PROBS), shape=(4096, 1)
        )
        start = time.perf_counter()
        losses = model.fit(
            jax.random.key(seed + 101),
            data,
            config=FitConfig(
                max_steps=args.train_steps,
                batch_size=128,
                learning_rate=1e-3,
                clip_norm=10.0,
            ),
        ).losses
        jax.block_until_ready(losses)
        metrics.update({
            "training_seconds": time.perf_counter() - start,
            "finite_losses": bool(jnp.isfinite(losses).all()),
            "final_loss": float(losses[-1]),
            **categorical_validation(model, seed),
        })
    distribution = model.as_dist(num_sample_steps=case["steps"])
    call = jax.jit(partial(distribution.sample, shape=(args.samples,)))
    start = time.perf_counter()
    samples = jax.block_until_ready(call(jax.random.key(seed + 102)))
    cold = time.perf_counter() - start
    durations = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        jax.block_until_ready(call(jax.random.key(seed + 102)))
        durations.append(time.perf_counter() - start)
    return {
        **metrics,
        **categorical_metrics(samples),
        "cold_seconds": cold,
        "warm_ms": 1000 * float(np.median(durations)),
    }


def summarize(saved):
    rows = saved["rows"]
    phase = saved["settings"]["phase"]
    expected = len(list(cases(argparse.Namespace(**saved["settings"]))))
    errors = [row for row in rows if row["status"] != "ok"]
    print(f"{phase}: {len(rows)}/{expected} cases, {len(errors)} errors")
    for row in errors:
        print(row["case"], row["error"])
    rows = [row for row in rows if row["status"] == "ok"]
    bad = [
        row
        for row in rows
        if not row.get("finite_losses", True) or row.get("finite_fraction", 1) < 1
    ]
    print(f"Non-finite cases: {len(bad)}")
    if phase == "oracle":
        for row in rows:
            case = row["case"]
            if case["target"] == "mixture" and case["mode"] == "ode":
                print(
                    f"{case['preset']:9} {case['solver']:20} {case['steps']:3}: "
                    f"SWD {row['swd2']:.5f}, covariance {row['covariance_relative_error']:.5f}, "
                    f"warm {row['warm_ms']:.3f} ms"
                )
    elif phase == "learned":
        for preset, training in itertools.product(PRESETS, TRAININGS):
            selected = [
                row
                for row in rows
                if row["case"]["preset"] == preset
                and row["case"]["training"] == training
                and row["case"]["head"] == "x0"
            ]
            if not selected:
                continue
            swd = [row["swd2"] for row in selected]
            mse = [row["denoising_excess_mse"] for row in selected]
            spread = np.std(swd, ddof=1) if len(swd) > 1 else 0.0
            print(
                f"{preset:9} {training:28}: SWD {np.mean(swd):.5f} ± {spread:.5f}; "
                f"excess MSE {np.mean(mse):.5f}; warm {np.mean([r['warm_ms'] for r in selected]):.2f} ms"
            )
        spreads = []
        for preset, training, seed in itertools.product(
            PRESETS, TRAININGS, range(saved["settings"]["seeds"])
        ):
            values = [
                row["swd2"]
                for row in rows
                if row["case"]["preset"] == preset
                and row["case"]["training"] == training
                and row["case"]["seed"] == seed
            ]
            if values:
                spreads.append(np.ptp(values))
        if spreads:
            print(f"Maximum paired x0/eps/v SWD spread: {max(spreads):.8f}")
        for seed in range(saved["settings"]["seeds"]):
            count = saved["settings"]["learned_samples"]
            reference = target_sample("mixture", jax.random.key(seed + 103), count)
            independent = target_sample("mixture", jax.random.key(seed + 104), count)
            print(
                f"Reference-vs-reference SWD seed {seed}: {sample_metrics(independent, reference, target_std('mixture'))['swd2']:.5f}"
            )
    else:
        steps = max(saved["settings"]["sample_steps"])
        for kind in CATEGORICAL_KINDS:
            oracle_rows = [
                row
                for row in rows
                if row["case"]["kind"] == kind
                and row["case"]["oracle"]
                and row["case"]["steps"] == steps
            ]
            print(
                kind,
                "oracle:",
                [
                    (r["case"]["matched_terminal"], round(r["total_variation"], 5))
                    for r in oracle_rows
                ],
            )
            for training in ("uniform", "importance"):
                selected = [
                    row
                    for row in rows
                    if row["case"]["kind"] == kind
                    and not row["case"]["oracle"]
                    and row["case"]["training"] == training
                ]
                values = [r["total_variation"] for r in selected]
                if not values:
                    continue
                spread = np.std(values, ddof=1) if len(values) > 1 else 0.0
                print(
                    f"  {training:10}: TV {np.mean(values):.5f} ± {spread:.5f}; "
                    f"excess CE {np.mean([r['excess_ce'] for r in selected]):.5f}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["oracle", "learned", "categorical"], required=True
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--legacy-categorical",
        action="store_true",
        help="Reproduce the original weak-noise categorical benchmark settings",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--learned-samples", type=int, default=512)
    parser.add_argument("--learned-sample-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output = (
        args.output or Path("docs/benchmarks") / f"diffusion_configs_{args.phase}.json"
    )
    if args.summary:
        summarize(json.loads(args.output.read_text()))
        return
    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output", "resume", "limit", "summary", "legacy_categorical"}
    }
    if args.phase == "categorical":
        settings["legacy_categorical"] = args.legacy_categorical
    rows = []
    if args.resume:
        saved = json.loads(args.output.read_text())
        previous = dict(saved["settings"])
        if args.phase == "categorical":
            previous.setdefault("legacy_categorical", True)
        if previous != settings:
            raise ValueError("Resume settings differ from the saved benchmark")
        rows = saved["rows"]
    completed = {json.dumps(row["case"], sort_keys=True) for row in rows}
    metadata = {
        "settings": settings,
        "jax": jax.__version__,
        "python": platform.python_version(),
        "devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }
    for case in itertools.islice(cases(args), args.limit):
        if json.dumps(case, sort_keys=True) in completed:
            continue
        print(f"Starting {case}", flush=True)
        try:
            metrics = run_case(case, args)
            row = {"case": case, "status": "ok", **metrics}
        except (
            ValueError,
            TypeError,
            FloatingPointError,
            NotImplementedError,
        ) as error:
            row = {
                "case": case,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
        args.output.write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
