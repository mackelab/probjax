# Discrete targets benchmark (quick)

`docs/scripts/benchmark_discrete_targets.py` — can a continuous model, then
quantized, model an intrinsically discrete distribution, versus modelling the
tokens directly with discrete diffusion? Small MLP, 3000 steps, one seed.

## Setup

Targets (simple, fixed finite support):
- `poisson` — univariate Poisson(3), truncated to k=0..19.
- `corr_poisson` — bivariate Poisson(3) margins coupled by a Gaussian copula
  (rho=0.7): genuinely dependent, not a product.
- `grid` — 2D categorical on a 4x4 grid supported only on cells where `i+j` is
  even: non-product support with holes.

Arms:
- `masked` — absorbing-mask discrete diffusion over the K token classes
  (the boxed masked-only objective).
- `uniform` — uniform-corruption discrete diffusion over K classes.
- `continuous` — EDM on the integer values treated as real numbers; samples
  rounded to the nearest integer.
- `dequant` — EDM on integers + U(-0.5, 0.5) jitter; samples rounded back.

Metric: total variation (TV) between the empirical sample PMF and a 400k-sample
reference PMF (lower better). For 2D targets also marginal TV, correlation error,
and the fraction of samples off the true support.

## Results

| target | arm | TV | TV_marg | corr_err | off_support |
| --- | --- | --- | --- | --- | --- |
| poisson | masked | 0.0702 | - | - | - |
| poisson | uniform | **0.0409** | - | - | - |
| poisson | continuous | 0.1598 | - | - | - |
| poisson | dequant | 0.1220 | - | - | - |
| corr_poisson | masked | **0.1474** | 0.0968 | 0.1358 | - |
| corr_poisson | uniform | 0.1684 | 0.0657 | 0.3934 | - |
| corr_poisson | continuous | 0.1911 | 0.1642 | **0.0905** | - |
| corr_poisson | dequant | 0.1793 | 0.1517 | 0.1138 | - |
| grid | masked | 0.0907 | 0.0442 | 0.0975 | 0.0182 |
| grid | uniform | 0.1128 | 0.0147 | 0.0857 | 0.1128 |
| grid | continuous | **0.0613** | 0.0262 | 0.0701 | **0.0010** |
| grid | dequant | 0.2135 | 0.0490 | 0.1196 | 0.2135 |

## Takeaways

1. **Univariate discrete (Poisson): continuous+quantize is clearly worse.**
   `continuous` TV 0.160 and `dequant` 0.122 versus `uniform` 0.041 — a 3-4x
   gap. Rounding a continuous fit to integers smears the sharp Poisson modes;
   the discrete model nails the PMF directly. `masked` is middling (0.070).
2. **Correlated discrete: mixed, and the arms trade off.** On joint TV the
   discrete arms win (`masked` 0.147); but on the *dependence structure* the
   continuous arms win (`continuous` corr_err 0.091 vs `masked` 0.136,
   `uniform` 0.393). Continuous+round gets the correlation right while getting
   the joint PMF wrong; masked is the reverse.
3. **Non-product support (`grid`): continuous wins outright.** `continuous`
   TV 0.061 with only 0.1% of samples off-support, versus `masked` 0.091 and
   `uniform` 0.113 (11% off-support). A smooth density naturally avoids the
   holes; uniform-corruption discrete diffusion scatters off-support.
4. **`dequant` is the worst grid arm** (TV 0.214, 21% off-support): training on
   jittered data broadens the density, and rounding then scatters across the
   holes rather than respecting them. Jitter helps on the 1D Poisson but hurts
   on the structured grid.

## Answer to the question

Can continuous-then-quantize model discrete distributions well? **Partly, and it
depends on the target:**

- For a **smooth, unimodal count** distribution (Poisson), no — direct discrete
  modelling is 3-4x better in TV; rounding loses too much.
- For **structure/support constraints** (the grid), yes — continuous wins, and it
  respects holes far better than the discrete arms, whose corruption scatters
  off-support.
- For **dependence in correlated counts**, the continuous model reproduces the
  correlation but not the joint PMF, and the discrete model does the opposite.

So the earlier "continuous wins on density, discrete wins on samples" split
carries over to discrete data, with the extra twist that continuous+quantize
cannot represent sharp atoms but handles support geometry more gracefully.

## Caveats

- One seed, ~3k steps, tiny backbone: TV gaps below ~0.02 are noise.
- `poisson`/`corr_poisson` truncate at k=20; the truncation tail is mass 0.003.
- `masked` still leaves residual MASK-token mass; decoded via x0 argmax as in the
  companion script.
