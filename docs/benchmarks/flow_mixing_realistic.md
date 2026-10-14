# NSF mixing: representative synthetic workloads

## Coverage and execution status

Six deterministic, eight-dimensional generators are available in
`docs/scripts/benchmark_flow_mixing.py`:

| Case | Motivation | Construction |
| --- | --- | --- |
| Correlated Gaussian | Correlated sensor/tabular features | Rotated Gaussian; covariance condition number 100 |
| Gaussian mixture | Multiple operating regimes or subpopulations | Four weighted, separated Gaussian modes |
| Student-t | Heavy-tailed measurements or returns | Elliptical multivariate Student-t, four degrees of freedom |
| Correlated lognormal | Positive, skewed measurements | Exponential of a correlated Gaussian |
| Bounded beta | Proportions and bounded continuous measurements | Eight differently shaped beta marginals |
| Funnel | Hierarchical posteriors and heteroscedasticity | Log-variance N(0, 1.5²), conditional Gaussian features |

These are synthetic proxies, not results on collected real-world datasets.
All six reference log-densities pass independent SciPy checks.

The first five cases have a complete matched comparison: four configurations
and three seeds, totaling 60 fits. The funnel has a validated generator and
one completed baseline fit. Repeated long-run timeouts prevented its full
comparison from completing; no comparative funnel conclusion is claimed.

## Matched setup

- Five autoregressive NSF transforms, 16 spline bins, `(64, 64)` conditioner.
- 8,000 training and 5,000 independently sampled held-out examples per seed.
- 600 Adam updates, peak learning rate 1e-3, inherited warmup-cosine schedule,
  batch size 256, global gradient clipping at 10.
- Conditioner weights, training/test draws, and minibatch keys matched across
  candidates within each seed. Independent snapshots prevent state sharing.
- Apple Silicon CPU, JAX 0.9.2. These measurements use explicit original/tuned
  settings. Subsequently the package damping default changed to 0.1; flips
  and the [-5, 5] domain remain unchanged. Benchmark baseline settings stay
  explicit to preserve reproducibility.

The four configurations are original flips, flip+CD, flips with wider/damped
tails, and flip+CD with wider/damped tails. Wider/damped means equal input and
output bounds `[-6, 6]` and `tail_logit_scale=0.1`; original bounds are `[-5, 5]`.

## Held-out negative log-likelihood

Nats/event, mean ± sample standard deviation over three seeds; lower is better.
Negative values for beta data are valid for continuous densities.

| Case | Flip | Flip+CD | Flip, wider/damped | Flip+CD, wider/damped |
| --- | ---: | ---: | ---: | ---: |
| Correlated Gaussian | 7.8330 ± 0.0202 | 7.8377 ± 0.0221 | **7.8136 ± 0.0208** | 7.8174 ± 0.0241 |
| Gaussian mixture | 8.9286 ± 0.0204 | 8.9382 ± 0.0402 | **8.9121 ± 0.0239** | 8.9173 ± 0.0428 |
| Student-t | 14.4312 ± 0.0929 | 14.3154 ± 0.0770 | 14.4087 ± 0.0972 | **14.3076 ± 0.0947** |
| Correlated lognormal | 3.2553 ± 0.0684 | 3.0866 ± 0.0635 | 3.2254 ± 0.0682 | **3.0855 ± 0.0625** |
| Bounded beta | -3.8346 ± 0.0259 | -3.8244 ± 0.0295 | **-3.8471 ± 0.0261** | -3.8444 ± 0.0267 |

Mean reference NLLs on the same held-out draws are 7.7044, 8.6519, 14.0534,
2.9280, and -3.9575, respectively. Every completed fit, evaluated log-density,
and sampled batch was finite. The standalone funnel baseline scores 12.2300,
against reference NLL 11.7619; this is one seed, not a mixer comparison.

## Interpretation

- CD's clearest gains are on Student-t (about 0.116 nats/event over original
  flips) and lognormal features (about 0.169), with the same direction in all
  three paired seeds. It is not a consistent improvement on the other cases.
- Wider/damped tails slightly improve average likelihood in all five cases
  relative to their corresponding original-tail configuration, but the gains
  are small on ordinary Gaussian, mixture, and bounded data.
- Better mean Student-t likelihood does not imply better worst-tail behavior:
  original flip+CD's worst held-out NLLs are about 64–106; adding wider/damped
  tails gives about 97–255. Do not rank robustness using means alone.
- Gaussian-base NSF does not enforce positive or bounded support. Lognormal
  runs generate about 0.4–1.0% invalid-support events; beta runs about 1.2–1.7%.
  Log/logit preprocessing or an appropriate support transform is relevant for
  such applications; mixing changes alone do not address the support mismatch.
- Warmed CPU updates are mostly about 16–18 ms per batch. Wider/damped-tail
  sampling is slower in this run (roughly 50–56 ms per 5,000 samples, versus
  38–40 ms for original tails). Timing is a local CPU screen, not a GPU claim.

The expanded evidence supports retaining these options as tunable choices,
rather than switching the general NSF default to CD. A longer convergence
study and actual tabular/posterior datasets would be needed for a default change.

## Reproduce or resume

```bash
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_flow_mixing \
  --suite realistic --test-size 5000 \
  --output docs/benchmarks/flow_mixing_realistic.json

# Resume the unfinished funnel rows without repeating the completed 60 fits.
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_flow_mixing \
  --suite realistic --test-size 5000 --resume \
  --output docs/benchmarks/flow_mixing_realistic.json

# Summarize the saved checkpoint without training.
python -m docs.scripts.benchmark_flow_mixing --summary-only \
  --output docs/benchmarks/flow_mixing_realistic.json
```

Per-seed NLL, reference gap, training NLL, median/p99/worst-point NLL, support
violations, finite-value flags, and timings are saved in
`flow_mixing_realistic.json`. The standalone funnel result is saved in
`flow_mixing_funnel_baseline.json`.
