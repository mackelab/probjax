"""Inspect optimization and held-out tail failures in the NSF mixing benchmark.

Run from the repository root with ``python -m docs.scripts.diagnose_flow_mixing``.
Uses the exact benchmark initialization, data, minibatches, and warmup-cosine
schedule, while recording raw gradient norms and independent model snapshots.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from docs.scripts.benchmark_flow_mixing import MIXERS, build_flow, rotated_banana
from probjax.core import inverse_and_logabsdet
from probjax.nn import (
    CirculantDiagonal,
    FitCallback,
    FitConfig,
    FlipMixingConfig,
    RationalQuadraticSplineConfig,
)


@nnx.jit
def logpdf(model, data):
    return jax.vmap(model._logpdf)(data)


def tail_summary(values):
    nll = -np.asarray(values)
    ordered = np.sort(nll)
    return {
        "nll": float(nll.mean()),
        "median": float(np.median(nll)),
        "p99": float(np.quantile(nll, 0.99)),
        "max": float(ordered[-1]),
        "mean_without_worst": float(ordered[:-1].mean()),
        "worst_contribution_to_mean": float(ordered[-1] / len(nll)),
        "top_one_percent_contribution_to_mean": float(
            ordered[-max(1, len(nll) // 100) :].sum() / len(nll)
        ),
    }


def track_gradients(clip_norm):
    """Record [last norm, maximum norm, clipped-step count, total steps]."""

    def update(grads, state, params=None):
        del params
        norm = optax.global_norm(grads)
        return grads, jnp.array([
            norm,
            jnp.maximum(state[1], norm),
            state[2] + (norm > clip_norm),
            state[3] + 1,
        ])

    return optax.GradientTransformation(lambda _: jnp.zeros(4), update)


class Snapshots(FitCallback):
    def __init__(self):
        self.entries = []

    def on_step_end(self, state, info):
        self.entries.append((
            int(state.step),
            jax.tree.map(lambda value: value.copy(), state.params),
            jax.tree.map(lambda value: value.copy(), state.aux),
            float(info.loss),
            np.asarray(state.opt_state[0]).tolist(),
        ))


def worst_point_details(flow, point):
    value = flow._standardize(point)
    standardized = value
    stages = []
    for index in reversed(range(len(flow.transformation.layers))):
        layer = flow.transformation.layers[index]
        entry = {
            "layer": index,
            "type": type(layer).__name__,
            "input_max_abs": float(jnp.max(jnp.abs(value))),
        }
        if hasattr(layer, "predict_bij_params"):
            params = layer.predict_bij_params(value).reshape(8, layer.bijector_dim)
            x_pos, y_pos, slopes = layer.bijector.unpack(params)
            entry.update({
                "min_x_bin": float(jnp.min(jnp.diff(x_pos, axis=-1))),
                "min_y_bin": float(jnp.min(jnp.diff(y_pos, axis=-1))),
                "min_slope": float(jnp.min(slopes)),
                "max_slope": float(jnp.max(slopes)),
                "tail_dimensions": int(
                    jnp.sum((value < y_pos[:, 0]) | (value > y_pos[:, -1]))
                ),
            })
            value, logdet = layer.inverse_and_logdet(value)
        else:
            value, logdet = inverse_and_logabsdet(layer)(value)
            children = layer.layers if hasattr(layer, "layers") else (layer,)
            conditions = []
            for child in children:
                if isinstance(child, CirculantDiagonal):
                    matrix = jax.jacfwd(child)(jnp.zeros(8))
                    conditions.append(float(jnp.linalg.cond(matrix)))
            entry["cd_condition_numbers"] = conditions
        entry["inverse_logdet"] = float(logdet)
        entry["output_max_abs"] = float(jnp.max(jnp.abs(value)))
        stages.append(entry)
    recovered = flow.transformation(value)
    rotation, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(42), (8, 8)))
    original = (point @ rotation.T) / jnp.linspace(0.5, 2.0, 8)
    original = original.at[4:].add(-0.8 * (original[:4] ** 2 - 1))
    true_nll = (
        0.5 * jnp.sum(original**2)
        + 4 * jnp.log(2 * jnp.pi)
        + jnp.sum(jnp.log(jnp.linspace(0.5, 2.0, 8)))
    )
    return {
        "data_point": np.asarray(point).tolist(),
        "standardized_max_abs": float(jnp.max(jnp.abs(standardized))),
        "latent_norm": float(jnp.linalg.norm(value)),
        "gaussian_base_nll": float(0.5 * jnp.sum(value**2) + 4 * jnp.log(2 * jnp.pi)),
        "true_target_nll": float(true_nll),
        "roundtrip_max_error": float(jnp.max(jnp.abs(recovered - standardized))),
        "stages": stages,
    }


@dataclass
class TailSplineConfig(RationalQuadraticSplineConfig):
    """Diagnostic boundary-logit damping, optionally fixing tail slopes to one."""

    unit_tails: bool = False

    def unpack(self, params):
        x_pos, y_pos, slopes = super().unpack(params)
        if self.unit_tails:
            slopes = slopes.at[..., 0].set(1).at[..., -1].set(1)
        return x_pos, y_pos, slopes


def run_case(
    mixer,
    seed,
    rate,
    clip,
    steps,
    interval,
    data_seed=None,
    unit_tails=False,
    tail_bound=5.0,
    tail_logit_scale=1.0,
):
    settings = SimpleNamespace(transforms=5, bins=16)
    data_seed = seed if data_seed is None else data_seed
    train_key, test_key, fit_key = jax.random.split(jax.random.key(data_seed + 100), 3)
    train, test = rotated_banana(train_key, 8000), rotated_banana(test_key, 2000)
    baseline = build_flow(8, seed, FlipMixingConfig(), settings)
    flow = build_flow(8, seed, MIXERS[mixer](), settings)
    for target, source in zip(
        list(flow.transformation.layers)[::2],
        list(baseline.transformation.layers)[::2],
        strict=True,
    ):
        nnx.update(target, jax.tree.map(lambda value: value.copy(), nnx.state(source)))
    if unit_tails or tail_bound != 5 or tail_logit_scale != 1:
        spline = TailSplineConfig(
            num_bins=16,
            x_min=-tail_bound,
            x_max=tail_bound,
            y_min=-tail_bound,
            y_max=tail_bound,
            tail_logit_scale=tail_logit_scale,
            unit_tails=unit_tails,
        )
        flow.config.bijector = spline
        for layer in list(flow.transformation.layers)[::2]:
            layer.bijector = spline
            layer.bijector_inv = inverse_and_logabsdet(spline, invertible_arg=1)
    history = Snapshots()
    optimizer = optax.chain(
        track_gradients(clip),
        FitConfig(
            max_steps=steps,
            learning_rate=rate,
            schedule="warmup_cosine",
            clip_norm=clip,
        ).default_optimizer(),
    )
    losses = flow.fit(
        fit_key,
        train,
        config=FitConfig(
            max_steps=steps,
            batch_size=256,
            optimizer=optimizer,
            callback_every=interval,
        ),
        callbacks=[history],
    ).losses
    trajectory = []
    for step, params, model_state, batch_loss, gradient_stats in history.entries:
        nnx.update(flow, params, model_state)
        entry = {
            "step": step,
            "batch_loss": batch_loss,
            "gradient_stats": gradient_stats,
        }
        entry.update(tail_summary(logpdf(flow, test)))
        trajectory.append(entry)
    test_values = logpdf(flow, test)
    alternate_seed = 1 if data_seed == 0 else 0
    alternate_key = jax.random.split(jax.random.key(alternate_seed + 100), 3)[1]
    alternate_test = rotated_banana(alternate_key, 2000)
    alternate_summary = tail_summary(logpdf(flow, alternate_test))
    worst_index = int(jnp.argmin(test_values))
    worst_point = test[worst_index]
    details = worst_point_details(flow, worst_point)
    with jax.enable_x64(True):
        flow64 = nnx.clone(flow)
        state64 = jax.tree.map(
            lambda value: (
                value.astype(jnp.float64)
                if jnp.issubdtype(value.dtype, jnp.floating)
                else value
            ),
            nnx.state(flow64),
        )
        nnx.update(flow64, state64)
        values64 = logpdf(flow64, test.astype(jnp.float64))
        comparison64 = tail_summary(values64)
        comparison64["same_worst_point_nll"] = float(-values64[worst_index])
    return {
        "mixer": mixer,
        "seed": seed,
        "data_seed": data_seed,
        "unit_tails": unit_tails,
        "tail_bound": tail_bound,
        "tail_logit_scale": tail_logit_scale,
        "learning_rate": rate,
        "clip_norm": clip,
        "steps": steps,
        "finite_losses": bool(jnp.all(jnp.isfinite(losses))),
        "train": tail_summary(logpdf(flow, train)),
        "test": tail_summary(test_values),
        "alternate_heldout": {"data_seed": alternate_seed, **alternate_summary},
        "float64": comparison64,
        "worst_point_index": worst_index,
        "worst_point": details,
        "trajectory": trajectory,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixers",
        nargs="+",
        choices=MIXERS,
        default=["flip", "flip_cd", "circulant_diagonal"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--unit-tails", action="store_true")
    parser.add_argument("--tail-bound", type=float, default=5.0)
    parser.add_argument("--tail-logit-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument(
        "--output", default="docs/benchmarks/flow_mixing_diagnostics.json"
    )
    args = parser.parse_args()
    results = []
    for mixer in args.mixers:
        result = run_case(
            mixer,
            args.seed,
            args.learning_rate,
            args.clip_norm,
            args.steps,
            args.interval,
            args.data_seed,
            args.unit_tails,
            args.tail_bound,
            args.tail_logit_scale,
        )
        results.append(result)
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps({
                key: result[key]
                for key in (
                    "mixer",
                    "seed",
                    "learning_rate",
                    "clip_norm",
                    "tail_bound",
                    "tail_logit_scale",
                    "test",
                    "alternate_heldout",
                    "train",
                    "float64",
                )
            }),
            flush=True,
        )
        print(json.dumps(result["worst_point"]), flush=True)
        jax.clear_caches()
