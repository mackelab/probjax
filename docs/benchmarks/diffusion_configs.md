# Diffusion configuration benchmark

## Coverage

All 238 cases completed without exceptions or non-finite losses/samples:

- **128 oracle sampler cases:** four schedules × four solvers × ODE/SDE ×
  32/128 grid points × two distributions.
- **78 continuous training runs:** 13 valid schedule/time-sampler combinations
  × three prediction targets × two seeds.
- **32 categorical cases:** 16 exact-posterior sampler controls and 16 learned
  runs spanning all four schedule kinds and both time-sampling config classes.

Covered concrete config types:

- Schedules: `EDMNoiseSchedule`, `VENoiseSchedule`, `VPNoiseSchedule`,
  `CosineNoiseSchedule`.
- Training: `EDMTrainingConfig`, `SigmaEffEDMTrainingConfig`,
  `LogSNRTrainingConfig`, `UniformTTrainingConfig`.
- Preconditioning: `EDMPreconditioning`.
- Solvers: `BaseSolverConfig`, `EDMSolverConfig`, `DDIMSolverConfig`,
  `VSolverConfig`.
- Categorical: `MultinomialDiffusionSchedule` with linear/cosine/sigmoid/log-SNR
  factories, `CategoricalEDMPreconditioning`,
  `UniformContinuousTimeTrainingConfig`, `ImportanceContinuousTimeTrainingConfig`.

Protocols and the abstract noise-schedule scaffold are not independently
sampleable configurations. Direct log-normal physical-time `EDMTrainingConfig`
is evaluated on EDM's sigma-time schedule; other schedules use its effective-
noise counterpart. Configurable flags and numerical methods beyond the concrete
classes' defaults are not exhaustively enumerated. No diffusion defaults changed.

## Setup and metrics

Apple Silicon CPU, JAX 0.9.2, float32. Targets are a correlated 2D Gaussian
(covariance eigenvalues 0.25 and 1.75) and an equally weighted four-mode 2D
Gaussian mixture centered at `(±2, ±2)`, with within-component std 0.5.
The categorical target is one four-class token with probabilities
`[0.05, 0.15, 0.30, 0.50]`.

- Oracle denoisers use exact clean-data posterior means/probabilities.
- Continuous learners use the same `TimeMLP`: width 32, depth one, time embedding
  width 32, 16 Fourier features. Net initialization, training draws, and
  minibatch keys are matched by seed across configs.
- 4,096 training examples, 400 constant-rate Adam updates at 1e-3, batch 128,
  gradient clipping at 10. Categorical learners use a `(5, 32, 32, 4)` MLP.
- Learned continuous models are evaluated with the same 64-point EDM/Heun ODE
  sampler, separating curriculum/head effects from sampler changes. This is a
  component-wise comparison, not every possible full-system Cartesian product.
- Oracle/categorical sampling uses 2,048 events; learned continuous sampling
  uses 512. Three synchronized warm calls, with export/first-call time separate.
- `SWD2` is the root mean squared distance between sorted projections onto
  32 fixed directions, normalized by target RMS std. Lower is better.
- Common denoising validation uses effective-noise levels
  `[0.01, 0.03, 0.1, 0.3, 1, 3, 10]`. Excess MSE measures prediction error
  against the exact posterior mean, normalized by target variance.
- Categorical quality is total variation (TV) from the target class frequencies.

Finite-sample reference-vs-reference SWD is 0.0644 for the oracle mixture panel
and 0.0953 for the Gaussian panel. Learned-panel reference comparisons are
0.0780 and 0.1157 for its two seeds. These are Monte Carlo reference scales,
not hard lower bounds. Small differences near them are not a reliable ranking.

## Oracle sampler comparison

Four-mode mixture, 32 grid points, ODE mode:

| Schedule | Base/Euler SWD | EDM/Heun SWD | DDIM/Euler SWD | V/exp-AB2 SWD |
| --- | ---: | ---: | ---: | ---: |
| EDM | 0.0761 | 0.0576 | 0.0761 | 0.0555 |
| VE | 0.2715 | 0.0579 | 0.2715 | 0.0575 |
| VP | 0.0586 | 0.0576 | 0.0586 | 0.0559 |
| Cosine | 0.0556 | 0.0574 | 0.0556 | 0.0570 |

The main numerical weakness is coarse VE Euler sampling: covariance error is
45.8% at 32 points and still 14.2% at 128. V/exp-AB2 and EDM/Heun reduce the
32-point covariance error to 2.8% and 2.1%, respectively. This supports the
existing VE V-solver choice; it does not require changing the VE default.

V/exp-AB2 oracle ODE warm runtime is about 0.72–0.77 ms at 32 points and
2.51–2.59 ms at 128, for the mixture target. EDM/Heun is about 1.50–1.81 ms and
5.73–7.03 ms. These methods have different evaluation counts. Oracle SDE paths
typically take about 3–4 ms and 12–14 ms. Learned denoisers can change that
relative cost, so the oracle timing is not a learned ODE-vs-SDE conclusion.

DDIM's SDE interface has zero diffusion at default eta=0 and remains
deterministic conditional on its initial noise. It still pays SDE-driver
overhead. ODE and SDE wrappers use different initial PRNG splitting, so their
fixed-seed samples need not be identical.

## Learned continuous comparison

Mean SWD ± sample SD across two seeds, using `x0` (the other heads agree below).
All rows use the same EDM/Heun evaluation sampler.

| Schedule | Time-sampling config | SWD ↓ | Excess denoising MSE ↓ |
| --- | --- | ---: | ---: |
| EDM | EDMTrainingConfig | 0.1624 ± 0.0225 | 0.0520 |
| EDM | SigmaEffEDMTrainingConfig | 0.1624 ± 0.0225 | 0.0520 |
| EDM | LogSNRTrainingConfig | 0.2025 ± 0.0018 | 0.0138 |
| EDM | UniformTTrainingConfig | 0.2760 ± 0.0084 | 0.0143 |
| VE | SigmaEffEDMTrainingConfig | 0.1621 ± 0.0220 | 0.0520 |
| VE | LogSNRTrainingConfig | 0.2045 ± 0.0059 | 0.0140 |
| VE | UniformTTrainingConfig | 0.1899 ± 0.0078 | 0.0143 |
| VP | SigmaEffEDMTrainingConfig | 0.1552 ± 0.0262 | 0.0518 |
| VP | LogSNRTrainingConfig | 0.1775 ± 0.0082 | 0.0166 |
| VP | UniformTTrainingConfig | **0.1180 ± 0.0119** | 0.0173 |
| Cosine | SigmaEffEDMTrainingConfig | 0.1622 ± 0.0231 | 0.0520 |
| Cosine | LogSNRTrainingConfig | 0.1901 ± 0.0002 | 0.0136 |
| Cosine | UniformTTrainingConfig | **0.1088 ± 0.0101** | 0.0254 |

Uniform physical-time sampling is promising for VP and cosine on this small
mixture at this budget. Uniform EDM physical time instead spends much of its
budget at large sigma and has the worst sample quality here. These are two-
seed, one-target screens, not enough evidence for a general curriculum default
change. Global denoising validation MSE does not necessarily rank generation
quality: it weights noise levels differently from the sampling trajectory.

`x0`, `eps`, and `v` differ by at most 1.8e-7 in paired SWD. This implementation
derives epsilon and normalized v from the same underlying denoiser and applies
matching weights, making their objectives algebraically equivalent up to
floating-point effects. This is not a comparison of three independently
parameterized network architectures.

EDM direct-time and sigma-effective log-normal configs also agree on EDM,
where effective noise equals physical time. Learned sampler warm runtime is
about 13–14 ms for 512 events at 64 points on this network/machine.

## Categorical results and terminal-prior control

Exact posterior, 128 reverse transitions:

| Schedule kind | Terminal signal alpha_bar | TV with default uniform start ↓ | TV with correct noised-target start ↓ |
| --- | ---: | ---: | ---: |
| Linear | 0.95118 | 0.25703 | 0.00605 |
| Cosine | ~1e-12 | 0.01582 | 0.01582 |
| Sigmoid | 0.95118 | 0.25703 | 0.00557 |
| Log-SNR | 0.00247 | 0.02197 | 0.02197 |

The default linear/sigmoid rates retain about 95% of the signal. Their terminal
distribution on a skewed target is therefore far from the uniform base prior.
The control changes only initialization, retaining the exact same posterior
and reverse kernel; its large improvement isolates terminal-prior mismatch.
The correct start uses the known synthetic target probabilities and is an
oracle control, not an available general solution for unknown data.

Learned models, 64 transitions, mean TV ± sample SD over two seeds:

| Kind | UniformContinuousTimeTrainingConfig | ImportanceContinuousTimeTrainingConfig |
| --- | ---: | ---: |
| Linear | 0.26143 ± 0.00621 | 0.26045 ± 0.00483 |
| Cosine | 0.05073 ± 0.00863 | 0.05000 ± 0.00898 |
| Sigmoid | 0.26216 ± 0.00656 | 0.26118 ± 0.00518 |
| Log-SNR | 0.06880 ± 0.00587 | 0.11372 ± 0.02659 |

The weak-noise models have validation CE near the exact posterior's CE but
remain biased in generation. Training longer does not resolve a forward-process
terminal-prior mismatch by itself. For meaningful generation on nonuniform
categorical data, cosine/log-SNR or stronger integrated linear/sigmoid noise
are better candidates. Importance time sampling is not universally better;
it worsens log-SNR generation in this budget. No categorical rates were changed.

## Reproduce

```bash
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_diffusion_configs --phase oracle
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_diffusion_configs --phase learned
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_diffusion_configs --phase categorical --legacy-categorical

# Append --resume to skip completed cases, or --summary to inspect saved results.
```

`--limit`, `--train-steps`, `--seeds`, `--sample-steps`, and sample-count flags
support smaller smoke tests or larger follow-ups. Resume requires matching
settings. Raw cases, finite flags, denoising curves, cold/warm timings, and
quality metrics are in `diffusion_configs_{oracle,learned,categorical}.json`.

Validation covers analytic Gaussian/mixture/categorical oracles, matrix config
coverage, time bounds, and non-finite metric handling. This benchmark adds an
end-to-end learned comparison to the earlier `diffusion_audit.md`; it remains a
low-dimensional CPU screen, with no image, large-model, or GPU conclusions.

## Updated categorical defaults

Linear and sigmoid factories now use continuous rates `beta_start=0.1` and
`beta_end=20.0`. Their terminal signal is approximately 4.32e-5, independent of
reverse grid size. The cosine rate cap is 50.0 rather than the old per-step-like
0.999 cap; transition probabilities still come from bounded alpha ratios.

Matched rerun with the same four-class target and budgets:

| Kind | Old oracle TV, 128 steps | New oracle TV, 128 steps | Old learned uniform TV | New learned uniform TV |
| --- | ---: | ---: | ---: | ---: |
| Linear | 0.2570 | 0.0145 | 0.2614 | 0.0935 |
| Sigmoid | 0.2570 | 0.0165 | 0.2622 | 0.0834 |

The oracle default-start and matched-terminal-start results now agree: the
identified prior mismatch is removed. Learned models still have fitting error
at 400 updates. Cosine uniform training is unchanged (TV 0.0507); using
importance power one with the corrected rate cap is worse at this budget
(TV 0.1303), so rate correction is not a blanket recommendation for aggressive
importance sampling. The preset's importance power and variance-reduction
flags remain independently configurable.

```bash
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_diffusion_configs \
  --phase categorical --output docs/benchmarks/diffusion_defaults_categorical.json
```

[Original categorical measurements](diffusion_configs_categorical.json) and
[updated-default measurements](diffusion_defaults_categorical.json) are kept
separate. Use `--legacy-categorical` to reproduce the historical rate settings.
