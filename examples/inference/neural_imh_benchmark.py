"""Benchmark adaptive neural IMH on difficult target distributions.

Run, for example::

    uv run python examples/inference/neural_imh_benchmark.py --case all
    uv run python examples/inference/neural_imh_benchmark.py \
        --case mixture --seed-data 500
    uv run python examples/inference/neural_imh_benchmark.py \
        --case correlated --every 250 --fit-steps 250
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probjax.inference import fit_adaptor, imh, maximum_likelihood, run
from probjax.inference.mcmc.imh import imh_data
from probjax.inference.neural import density
from probjax.nn import FitConfig, nsf


@dataclass(frozen=True)
class Case:
    dimension: int
    logdensity: Callable
    sample: Callable
    metrics: Callable


def _normal_logpdf(value, mean, scale):
    return jax.scipy.stats.norm.logpdf(value, mean, scale).sum(axis=-1)


def _funnel_case() -> Case:
    def logdensity(value):
        x, v = value
        return jax.scipy.stats.norm.logpdf(v, 0.0, 3.0) + jax.scipy.stats.norm.logpdf(
            x, 0.0, jnp.exp(0.5 * v)
        )

    def sample(key, shape):
        key_v, key_x = jax.random.split(key)
        v = 3.0 * jax.random.normal(key_v, shape)
        x = jax.random.normal(key_x, shape) * jnp.exp(0.5 * v)
        return jnp.stack((x, v), axis=-1)

    def metrics(samples):
        expected_variance = jnp.array([jnp.exp(4.5), 9.0])
        return {
            "mean_error": jnp.linalg.norm(jnp.mean(samples, axis=0)),
            "variance_relative_error": jnp.mean(
                jnp.abs(jnp.var(samples, axis=0) / expected_variance - 1.0)
            ),
        }

    return Case(2, logdensity, sample, metrics)


def _mixture_case() -> Case:
    means = jnp.array([
        [-10.0, 0.0],
        [0.0, -10.0],
        [0.0, 0.0],
        [0.0, 10.0],
        [10.0, 0.0],
    ])
    scale = 0.7

    def logdensity(value):
        component_logpdf = jax.vmap(lambda mean: _normal_logpdf(value, mean, scale))(
            means
        )
        return jax.scipy.special.logsumexp(component_logpdf) - jnp.log(len(means))

    def sample(key, shape):
        key_component, key_noise = jax.random.split(key)
        indices = jax.random.randint(key_component, shape, 0, len(means))
        return means[indices] + scale * jax.random.normal(key_noise, shape + (2,))

    def metrics(samples):
        assignments = jnp.argmin(
            jnp.sum((samples[:, None, :] - means[None, :, :]) ** 2, axis=-1),
            axis=1,
        )
        frequencies = jnp.bincount(assignments, length=len(means)) / len(samples)
        return {
            "mean_error": jnp.linalg.norm(jnp.mean(samples, axis=0)),
            "mode_coverage": jnp.sum(frequencies > 0.01),
            "mode_total_variation": 0.5
            * jnp.sum(jnp.abs(frequencies - 1.0 / len(means))),
        }

    return Case(2, logdensity, sample, metrics)


def _correlated_case(dimension=16) -> Case:
    indices = jnp.arange(dimension)
    covariance = 0.95 ** jnp.abs(indices[:, None] - indices[None, :])
    precision = jnp.linalg.inv(covariance)
    logdet = jnp.linalg.slogdet(covariance)[1]
    chol = jnp.linalg.cholesky(covariance)

    def logdensity(value):
        return -0.5 * (
            dimension * jnp.log(2.0 * jnp.pi) + logdet + value @ precision @ value
        )

    def sample(key, shape):
        noise = jax.random.normal(key, shape + (dimension,))
        return noise @ chol.T

    def metrics(samples):
        sample_covariance = jnp.cov(samples, rowvar=False)
        return {
            "mean_error": jnp.linalg.norm(jnp.mean(samples, axis=0)),
            "covariance_relative_error": jnp.linalg.norm(sample_covariance - covariance)
            / jnp.linalg.norm(covariance),
        }

    return Case(dimension, logdensity, sample, metrics)


CASES = {
    "funnel": _funnel_case,
    "mixture": _mixture_case,
    "correlated": _correlated_case,
}


def _ready(value):
    jax.block_until_ready(value)
    return value


def run_case(name, args):
    case = CASES[name]()
    (
        key_initial,
        key_seed,
        key_fit,
        key_baseline,
        key_warmup,
        key_compile_sample,
        key_sample,
        key_proposal,
    ) = jax.random.split(jax.random.key(args.seed), 8)
    flow = nsf(
        case.dimension,
        args.transforms,
        rngs=nnx.Rngs(args.seed),
        num_bins=args.bins,
    )
    if args.seed_data > 0:
        # Pilot draws seed the proposal with modes the chain may never find.
        flow.fit(
            key_fit,
            case.sample(key_seed, (args.seed_data,)),
            config=FitConfig(max_steps=args.fit_steps, batch_size=args.batch_size),
        )

    started = time.perf_counter()
    proposal, params = density(flow)
    kernel = imh(case.logdensity, proposal=proposal)
    state = kernel.init(key_initial, case.sample(key_initial, (1,))[0])
    build_seconds = time.perf_counter() - started

    collect = ("position", "acceptance_rate")
    baseline = run(
        key_baseline, kernel, state, params, args.baseline_steps, collect=collect
    )
    _ready(baseline)
    baseline_acceptance = jnp.mean(baseline.trace["acceptance_rate"])

    policy = fit_adaptor(
        maximum_likelihood(proposal),
        optax.adam(args.learning_rate),
        data_fn=None if args.no_rao_blackwellize else imh_data,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        every=args.every,
        num_fit_steps=args.fit_steps,
    )
    started = time.perf_counter()
    adapted = run(
        key_warmup,
        kernel,
        state,
        params,
        args.warmup_steps,
        adaptor=policy,
        collect=("acceptance_rate",),
    )
    _ready(adapted)
    warmup_seconds = time.perf_counter() - started

    started = time.perf_counter()
    compiled_result = run(
        key_compile_sample,
        kernel,
        adapted.state,
        adapted.params,
        args.samples,
        collect=collect,
    )
    _ready(compiled_result)
    sampling_compile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    result = run(
        key_sample, kernel, adapted.state, adapted.params, args.samples, collect=collect
    )
    _ready(result)
    sampling_seconds = time.perf_counter() - started

    metrics = case.metrics(result.trace["position"])
    proposal_samples = jax.vmap(proposal.sample, (0, None))(
        jax.random.split(key_proposal, args.samples), adapted.params
    )
    proposal_metrics = {
        f"proposal_{key}": value
        for key, value in case.metrics(proposal_samples).items()
    }
    warmup_acceptance = adapted.trace["acceptance_rate"]
    output = {
        "case": name,
        "dimension": case.dimension,
        "seed_data": args.seed_data,
        "rao_blackwellized": not args.no_rao_blackwellize,
        "build_seconds": build_seconds,
        "warmup_seconds": warmup_seconds,
        "sampling_compile_seconds": sampling_compile_seconds,
        "sampling_seconds": sampling_seconds,
        "samples_per_second": args.samples / sampling_seconds,
        "baseline_acceptance": baseline_acceptance,
        # Mean acceptance over each tenth of the adaptation run.
        "warmup_acceptance": jnp.mean(
            warmup_acceptance[: len(warmup_acceptance) // 10 * 10].reshape(10, -1),
            axis=1,
        ),
        "final_acceptance": jnp.mean(result.trace["acceptance_rate"]),
        **metrics,
        **proposal_metrics,
    }
    return jax.tree.map(
        lambda value: value.tolist() if hasattr(value, "tolist") else value,
        output,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-data", type=int, default=0)
    parser.add_argument("--transforms", type=int, default=3)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--baseline-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--fit-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--no-rao-blackwellize", action="store_true")
    args = parser.parse_args()

    names = CASES if args.case == "all" else (args.case,)
    for name in names:
        print(json.dumps(run_case(name, args), sort_keys=True))


if __name__ == "__main__":
    main()
