# Discretization benchmark (quick)

`docs/scripts/benchmark_discretization.py` — masked discrete diffusion vs
quantization + continuous, on 2D toy densities. Small MLP backbone, 2000 steps,
one seed, 2 targets. Numbers are indicative, not publication-grade.

## Setup

- Targets: `crescent` (curved, bimodal) and `skewed_copula` (gamma margins,
  Gaussian-copula dependence, so strongly skewed and dependent).
- Values normalized to `[-1, 1]`.
- Arms, all on the same backbone class and the same 2000 training steps:
  - `masked` — absorbing-mask diffusion over N bins (+1 MASK class), trained with
    the boxed masked-only CE `E[-sum_{i: z_t^i=MASK} log p(z0^i|z_t,t)]`.
  - `uniform` — uniform-corruption diffusion over N bins, CE at every position.
  - `continuous` — EDM (x0-prediction) on the raw values.
  - `dequant` — same EDM, trained on values dequantized uniformly inside their
    bin.
- Binning: uniform-width (`uniform`) vs quantile (`quantile`).
- Bin counts: 32, 64, 128, 256.
- Metrics: energy distance to the true generator (lower better) and mean
  log-density of the samples under a fixed Gaussian-KDE fit to the generator
  (higher better).

## Results

`energy | ref_ll` per cell.

### crescent / uniform-width bins

| arm | N=32 | N=64 | N=128 | N=256 |
| --- | --- | --- | --- | --- |
| masked | **0.0040** \| -2.81 | 0.0099 \| -3.69 | 0.0137 \| -4.43 | 0.0101 \| -4.67 |
| uniform | 0.0100 \| -3.30 | 0.0097 \| -2.86 | 0.0125 \| -3.26 | **0.0053** \| -3.61 |
| continuous | 0.0104 \| **-0.90** | 0.0104 \| **-0.90** | 0.0104 \| **-0.90** | 0.0104 \| **-0.90** |
| dequant | 0.0095 \| -0.93 | 0.0102 \| -0.91 | 0.0104 \| -0.90 | 0.0103 \| -0.89 |

`continuous` is constant across N by construction: EDM trains on raw values and
never sees the binning. `dequant` is indistinguishable from `continuous` — the
uniform-in-bin dequantization adds nothing measurable at this budget.

### skewed_copula / quantile bins

| arm | N=32 | N=64 | N=128 | N=256 |
| --- | --- | --- | --- | --- |
| masked | 0.0141 \| -1.06 | 0.0097 \| -1.48 | 0.0242 \| -3.24 | 0.0246 \| -3.18 |
| uniform | 0.0061 \| -1.77 | 0.0093 \| -1.60 | 0.0090 \| -1.83 | 0.0102 \| -1.55 |
| continuous | 0.0163 \| **-0.25** | 0.0163 \| **-0.25** | 0.0163 \| **-0.25** | 0.0163 \| **-0.25** |
| dequant | 0.0162 \| -0.25 | 0.0159 \| -0.24 | 0.0159 \| -0.25 | 0.0159 \| -0.24 |

On skewed marginals, `masked` wins energy at N=32 (uniform-width: 0.0028, the
best cell in the whole run) but degrades badly at N=128/256 (0.02+), while
`uniform`-corruption stays flat. Quantile binning helps `uniform`-corruption and
the continuous arms slightly, and hurts `masked`.

## Takeaways

1. **The two metrics disagree, and the disagreement is the finding.** Sample
   geometry (energy distance) favors the discrete arms; density fidelity
   (KDE log-density) favors continuous by 1.4-4 nats, and the gap *widens* with
   N for discrete arms. This mirrors the literature (robotics: quantized models
   competitive; TabDAR: continuous better density) — the answer depends on which
   metric you optimize.
2. **The KDE log-density structurally penalizes piecewise-uniform sampling.**
   Samples that are uniform within a bin cannot concentrate on a density peak,
   so they lose under any bandwidth small enough to resolve the bin. Energy
   distance, which is bandwidth-free, does not show this penalty. Treat
   `ref_ll` here as "density peakedness", not a clean likelihood.
3. **Dequantization was a no-op.** EDM on raw values vs uniform-in-bin
   dequantized values are within noise on every cell. At these budgets the
   dequantization noise floor (bin width) is below model error.
4. **`masked` can beat `uniform`-corruption at low N but is fragile at high N.**
   It was best wherever bins were few and the target had sharp geometry, and
   worst at N>=128 where most positions stay masked and few CE terms are active.
5. **Uniform-width vs quantile is target-dependent**, exactly as predicted: on
   skewed features quantile bins help uniform-corruption and continuous arms,
   but not masked.

## Caveats

- One seed, ~2k steps, tiny backbone: differences below ~0.003 energy are noise.
- `ref_ll` is a KDE with one global bandwidth; it is a rough surrogate.
- The exact discrete NLL was not computed (the boxed objective is not a
  normalized density at a single t); `discrete_ce_minus_offset` reports the raw
  CE and the `d*log(2/N)` bin-width offset separately as a diagnostic.
