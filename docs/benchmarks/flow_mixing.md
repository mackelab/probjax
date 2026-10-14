# NSF feature-mixing screen

## Decision

Keep `FlipMixingConfig` as NSF's default. Circulant–diagonal mixing and
permutation-plus-circulant–diagonal mixing remain opt-in: this small synthetic
screen does not show a consistent likelihood or runtime advantage.

## Setup

- Apple Silicon CPU, macOS 26.3, JAX 0.9.2, float32.
- Autoregressive NSF: five transforms, the default 16 spline bins, and a
  `(64, 64)` MLP conditioner chosen to keep the screen quick.
- Three seeds, 8,000 training and 2,000 independently sampled held-out examples.
- 600 Adam steps, peak learning rate `1e-3` with the flow's inherited
  warmup-cosine schedule, batch size 256, gradient clipping at 10.
- Identical initial conditioner weights and minibatch keys within each seed.
  NNX state snapshots are copied independently to prevent training one
  candidate from changing the next candidate's initial weights.
- CD uses two diagonal factors and one circulant, with `init_std=0.05` and
  `max_log_scale=3`. Random permutations are fixed after construction.
- Fit wall time includes compilation. Train-step, logpdf, and sample timings
  exclude warmup/compilation and synchronize device execution. Train-step
  timing uses extra updates after the likelihood and sampling measurements.

## Held-out negative log-likelihood

Nats per event, mean ± sample standard deviation over three seeds; lower is
better. All 36 fits and all sampled batches were finite.

| Target | Flip | Permutation | CD | Permutation + CD |
| --- | ---: | ---: | ---: | ---: |
| Crescent, 2D | 2.1908 ± 0.0366 | 2.1934 ± 0.0370 | 2.1869 ± 0.0492 | 2.1720 ± 0.0296 |
| Spiral, 2D | 2.5628 ± 0.0094 | 2.5620 ± 0.0038 | 2.6575 ± 0.0246 | 2.5748 ± 0.0139 |
| Rotated banana, 8D | 14.2038 ± 2.1549 | 51.5916 ± 66.7713 | 13.0882 ± 0.1026 | 13.1076 ± 0.2252 |

The 8D means are sensitive to poor runs: seed zero gives NLL 16.6889 with
flips and 128.6924 with permutations. The corresponding CD and permutation+CD
values are 13.1365 and 13.3424. In seeds one and two, flips give 13.0692 and
12.8533, comparable to or better than the learned alternatives. The apparent
average improvement is therefore mostly robustness to one run, not a uniform
per-seed gain. Finite training losses alone do not imply good held-out density.

Permutation+CD improves the crescent by about 0.019 nats/event on average,
but worsens the spiral by about 0.012. CD alone worsens the spiral by 0.095.

## Runtime

Mean of each seed's warmed median train-step duration, milliseconds per
256-example update:

| Target | Flip | Permutation | CD | Permutation + CD |
| --- | ---: | ---: | ---: | ---: |
| Crescent, 2D | 4.98 | 6.07 | 4.71 | 4.98 |
| Spiral, 2D | 5.04 | 4.77 | 4.96 | 5.25 |
| Rotated banana, 8D | 16.52 | 16.22 | 16.71 | 16.39 |

Some short timings show host-load outliers, particularly permutation sampling
on the crescent and permutation+CD sampling on the spiral. The small differences
are not a reliable speed ranking; these results show no clear throughput win.
CD adds only 24 trainable parameters to the 2D flows and 96 to the 8D flows.

This is a fixed-budget screen, not a convergence study or a statistically
powered comparison. It covers neither real tabular/posterior workloads nor
large-dimensional or GPU execution. Those would be needed before changing a
general-purpose default.

## Reproduce

```bash
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_flow_mixing
```

Raw settings, per-seed likelihoods, parameter counts, and timings are in
`flow_mixing.json`. The script also accepts `--steps`, `--learning-rate`,
`--targets`, `--mixers`, and `--seeds` for follow-up comparisons.

## Follow-up: flip + CD

Nine additional fits used `CompositeMixingConfig((FlipMixingConfig(),
CirculantDiagonalMixingConfig()))`, with the same settings, datasets, seeds,
and baseline conditioner initialization as above. Follow-up runs now always
obtain conditioner snapshots from an untrained flip-baseline flow, regardless
of which mixers are selected on the command line.

| Target | Flip NLL | Flip + CD NLL | Flip + CD train step (ms) |
| --- | ---: | ---: | ---: |
| Crescent, 2D | 2.1908 ± 0.0366 | 2.1812 ± 0.0351 | 5.03 |
| Spiral, 2D | 2.5628 ± 0.0094 | 2.5533 ± 0.0102 | 5.29 |
| Rotated banana, 8D | 14.2038 ± 2.1549 | 17.7389 ± 8.3230 | 16.58 |

All additional training losses and sampled batches were finite. Flip+CD makes
small improvements on the two 2D targets at similar CPU runtime. However, the
8D seed-zero NLL rises to 27.3490, compared with flip's 16.6889 and CD's 13.1365.
Its other 8D seeds give 13.0157 and 12.8521, slightly better than the corresponding
flips. This reinforces the decision to keep it opt-in rather than change the
default: the limited-budget gains do not outweigh the bad-run regression.

```bash
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_flow_mixing \
  --mixers flip_cd \
  --output docs/benchmarks/flow_mixing_flip_cd.json
```

Raw follow-up measurements are in `flow_mixing_flip_cd.json`. The full benchmark
now includes `flip_cd` by default, alongside the original four choices.

## Diagnosis of the apparent seed instability

The large 8D NLL variance is dominated by rare held-out points with badly
extrapolated spline tails, rather than a global training-loss or gradient
explosion. The diagnostic script exactly reproduces the seed-zero failure.

- Flip+CD training NLL is 12.3885, and median held-out NLL is 12.3756.
- One of 2,000 held-out points has fitted NLL 28,834.62, versus its analytically
  known target NLL 28.44. It contributes 14.4173 nats/event to the average.
  Removing it for diagnosis gives a mean of 12.9382; the reported benchmark
  mean correctly retains it and is 27.3491.
- The point starts at max absolute standardized coordinate 8.46, outside the
  `[-5, 5]` spline range. Successive inverses reach magnitudes about
  21.93, 39.04, 41.24, 100.01, and 239.40 across the five spline blocks.
  Its final latent norm is about 240, so the Gaussian-base quadratic term alone
  is 28,813.79. Conditioner predictions saturate and some bins shrink near their
  minimum widths along that inverse path.
- Training raw gradient norms remain below 6.694: the clip-at-10 threshold
  never activates in this run. Validation NLL worsens progressively while
  typical-point and training losses improve; there is no single exploding step.
- Learned CD condition numbers are only about 1.20–1.40. Evaluating identical
  learned parameters in float64 gives mean NLL 27.34906 and worst-point NLL
  28,834.60, essentially unchanged. Precision loss and FFT matrix conditioning
  do not explain the density failure, although float32 roundtrip accuracy can
  deteriorate along extreme inverse paths.
- Holding the trained model fixed and changing only the independently sampled
  held-out set changes the mean from 27.3491 to 13.0023. This directly shows
  evaluation-tail sampling is a major contributor to the apparent variability.
  Changing initialization while retaining data seed zero still gives a bad
  mean (31.3700), dominated by the same point.

Matched seed-zero flip+CD ablations, each with 600 steps:

| Change | Held-out NLL | Training NLL | Worst held-out NLL |
| --- | ---: | ---: | ---: |
| Original: peak LR 1e-3, clip 10 | 27.3491 | 12.3885 | 28,834.62 |
| Peak LR 3e-4, clip 10 | 13.4442 | 12.8578 | 522.24 |
| Peak LR 1e-3, clip 1 | 19.9665 | 12.4052 | 14,042.54 |
| Fix spline boundary slopes to one | 12.9737 | 12.3961 | 113.29 |

Reducing the learning rate mitigates the failure at this step budget but also
slows fitting. Tighter clipping helps only partially. The strongest targeted
ablation fixes the two boundary slopes to one, making extrapolation outside the
equal `[-5, 5]` input/output ranges the identity. This prevents spline tail
amplification while retaining almost the same training fit. It is currently
only a diagnostic config, not a change to the package default, and needs broader
multi-seed/target verification before adoption. The remaining tail mismatch
also shows that unit tails do not solve all generalization issues.

Reproduce with `python -m docs.scripts.diagnose_flow_mixing`; use
`--learning-rate 0.0003`, `--clip-norm 1`, `--unit-tails`, and `--data-seed`
for the ablations. Per-checkpoint validation tails, gradient norms, per-layer
inverse behavior, and precision checks are saved in the
`flow_mixing_diagnostics*.json` files.

## Learnable-tail follow-up: wider bounds and boundary-logit damping

The package now exposes `tail_logit_scale` on spline configs. It multiplies only
the first and last raw slope logits before the existing positivity transform;
interior slopes and bin positions are unaffected. With scale `a`, the boundary
slope is `exp(tanh(a * raw) * log(max_knot_slope))`. A scale of 0.1 reduces
near-initialization output sensitivity and raw-logit gradients tenfold while
preserving unit initialization and the same asymptotic slope range. It also
scales bias contributions. This is damping, not fixing the tails or a guarantee
against eventual saturation. These runs used one as the original baseline;
the package default was subsequently changed to 0.1. The runner specifies
baseline damping explicitly so the recorded comparisons remain reproducible.

On the failing flip+CD seed-zero run:

| Bounds | Boundary-logit scale | Held-out NLL | Training NLL | Worst-point NLL |
| --- | ---: | ---: | ---: | ---: |
| [-5, 5] | 1 | 27.3491 | 12.3885 | 28,834.62 |
| [-6, 6] | 1 | 12.8969 | 12.3622 | 104.44 |
| [-5, 5] | 0.1 | 12.9713 | 12.3940 | 102.35 |
| [-6, 6] | 0.1 | 12.9054 | 12.3733 | 118.26 |

Three-seed, matched-budget 8D follow-up:

| Setting | Held-out NLL, mean ± sample SD |
| --- | ---: |
| Original flip+CD | 17.7389 ± 8.3230 |
| Bounds [-6, 6] | 12.8725 ± 0.1025 |
| Bounds [-6, 6], scale 0.1 | 12.8900 ± 0.1004 |

The wider domain alone has slightly better average in-distribution scores, but
does not eliminate the extrapolation failure: the model trained with seed two
scores the independent seed-zero held-out set at NLL 21.0882, with worst-point
NLL 16,332.17. Adding damping brings that same cross-held-out comparison to
12.9192, with worst-point NLL 152.59. Its three models score the fixed seed-zero
held-out set at 12.9054, 12.9571, and 12.9192, respectively.

Thus wider bounds plus mild damping is a promising learnable-tail setting for
this workload. Domain bounds remain unchanged; boundary-logit damping now
defaults to 0.1, with 1.0 available to restore the original behavior. The new
default is [-5, 5] with damping, rather than the wider-bound combination here.
Wider bounds also trade away knot resolution at a fixed bin count.

Example:

```python
from flax import nnx
from probjax.nn import (
    CirculantDiagonalMixingConfig, CompositeMixingConfig, FlipMixingConfig, nsf,
)

mixing = CompositeMixingConfig((FlipMixingConfig(), CirculantDiagonalMixingConfig()))
flow = nsf(
    8, 5, rngs=nnx.Rngs(0), mixing=mixing,
    x_min=-6, x_max=6, y_min=-6, y_max=6,
    tail_logit_scale=0.1,
)
```

Reproduce these diagnostic variants with `--tail-bound 6` and
`--tail-logit-scale 0.1`. Results are in the `bound6`, `damped_tails`, and
`bound6_damped` diagnostic JSON files, including the seed-one and seed-two
follow-ups. Boundary slopes affect the edge bins as well as the outside-domain
tails, so a boundary head can receive gradients even without outside-domain
examples; shared conditioner feature updates can subsequently alter those
learned predictions.

## Adopted damping default

The package now defaults to `tail_logit_scale=0.1`, retaining flips and the
`[-5, 5]` domain. The diagnosed seed-zero banana case was rerun with this
flip-only default: held-out NLL improves from 16.6889 to 13.0049, and worst-point
NLL drops from 6,323.63 to 113.96. Training NLL is 12.4605, compared with 12.4547
for the original setting. Results are saved in
`flow_mixing_default_tail_damping.json`.

This validates the change on the identified failure, not a universal guarantee
against outliers. The runner continues to use explicit `tail_logit_scale=1.0`
for its original-baseline labels; use the public NSF constructor's default to
obtain the newly adopted damping, or pass 1.0 to restore the old behavior.
