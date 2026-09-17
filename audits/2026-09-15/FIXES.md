# Audit fixes

Implemented in the isolated `codex/complementary-special` worktree. The original
`README.md`, `findings.json`, and reproduction logs describe the **pre-fix** state
and remain historical evidence. This document records the subsequent fixes.

## Confirmed findings

| Finding | Correction |
| --- | --- |
| I01 | Gaussian IMH proposal log density uses the square root of diagonal covariance, matching its sampler. |
| I02 | Merwe/Julier points use Cholesky columns. Simplex points use a centered, whitened Helmert basis. |
| I03 | UKF regenerates observation sigma points after adding process covariance. |
| I04 | UKF observation log likelihood includes the dimension-dependent Gaussian normalization. |
| I05 | UKF evaluates callable observation covariance at observation time. |
| S01 | Pareto samples no longer have an extra lower-bound shift. |
| S02 | Generalized Pareto implements the exponential limit at shape zero, masks support, handles endpoints, and preserves the quantile's shape derivative at zero. |
| S03 | Scalar transformed CDF/quantiles account for decreasing bijections using survival functions. |
| S04 | Binomial CDF floors fractional thresholds. |
| S05 | Binomial entropy uses normalized mode-relative probability recurrences for small variance and a corrected asymptotic expansion for large variance. |
| S06 | Poisson entropy sums probability-weighted log masses for small rates and uses an asymptotic expansion for large rates. |
| S07 | Truncated-normal quantiles and moments no longer call missing JAX functions. Infinite bounds remain constant under differentiation. Bounded-interval quadrature stabilizes narrow/tail moments. |
| U01 | Sample entropy estimators use corrected Van Es, Ebrahimi, and Correa formulas; default window selection works under JIT. |
| U02 | PCG no longer rejects positive denominators merely because they are smaller than absolute machine epsilon. |
| U03 | Lanczos logdet handles depth one and uses multiple trace probes, with full-basis evaluation for small matrices. |
| U04 | Upper-triangular and batched diagonal predicates test the intended matrix entries. |
| U05 | Optional multistep solver module imports successfully after moving its future import. |
| C01 | Nested JIT propagation preserves named-site dictionaries and scalar log potentials instead of treating every state as per-variable inverse metadata. |
| N01 | Transformer derives distinct explicit RNG keys for layers and stochastic operations, while repeated calls with the same key remain reproducible. |
| N02 | Flash3 forward/residual/backward adapters now invoke the actual backend functions with the expected argument and result structure. |

The interpolation schedule protocol also moved to `utils.protocols`, retaining
its config-module import path and runtime-checkable behavior. Loss annotations
can now import the shared protocol without a generative-model import cycle.

## Numerical behavior and costs

- `lanczos_logdet(..., num_probes=16)` separates quadrature depth from trace
  accuracy. A full Krylov depth alone does not remove stochastic trace error.
  At least `dimension` probes select the coordinate basis; full basis plus full
  depth recovers the dense reference up to roundoff. More probes and
  reorthogonalization cost additional work and memory compared with the old
  single all-ones probe. No runtime speedup is claimed.
- Binomial entropy uses a static 1025-point window when variance is at most 256;
  above that it uses the leading corrected normal approximation. Poisson sums
  1024 masses through rate 256 and uses an inverse-cubic asymptotic expansion
  above it. These are numerical approximations, not symbolic exact sums for
  arbitrary parameters. Tests cover endpoints, broadcasting/JIT through existing
  suites, and large/skewed inputs. Mode-relative normalization avoids the severe
  FP32 log-factorial cancellation observed for `n=1_000_000, p=1e-5`.
- Truncated-normal moments combine recurrence with 64-point Gauss–Legendre
  quadrature for finite intervals of moderate density variation. Centering in
  quadrature coordinates avoids cancellation in small variances. Very extreme
  regimes outside those covered remain subject to floating-point limits.
- Quantiles use fixed-iteration bisection with an implicit first-derivative
  correction. First derivatives are tested; this does not establish higher-order
  derivative accuracy.
- Transformed scalar CDF/PPF infer orientation from transformed base quartiles;
  they require a scalar monotone bijection.

## Validation

CPU test logs are in [validation](validation/). The main test environment has
JAX 0.10.0. Original reproduction scripts were also rerun with the checkout's
supported JAX 0.9.2 interpreter.

- Dedicated audit regressions: **65 passed**, covering all 20 findings, including
  sigma-point covariance reconstruction at dimensions 1, 3, 5, and 20. See
  `validation/audit_regressions.log`.
- Distribution, transform, inverse, and propagation suites: **1,115 passed,
  12 existing expected failures**.
- Filtering, MCMC/SMC, core backend, losses, and DiffusionTransformer suites:
  **205 passed**.
- Existing sample-entropy coverage: **12 passed**.
- After the FP32 refinements, affected distribution coverage rerun:
  **73 passed** (overlaps the broader distribution suite).
- Syntax/undefined-name checks passed for the changed audit implementation files.

SciPy's default Poisson entropy summation did not converge at rate 10,000, so the
regression reference explicitly sums a sufficiently wide support. SciPy returned
NaN for entropy with infinite truncation bounds; those cases use the analytic
normal/half-normal values instead.

Flash3 adapter plumbing is covered with CPU backend stubs. **No actual Flash3
GPU kernel execution was validated.** The old NN reproduction deliberately calls
its backend with `None` inputs; its remaining invalid-config exception is not a
GPU correctness test. Time-pair embedding symmetry remains an architectural
observation from the audit, not one of the 20 confirmed bugs.
