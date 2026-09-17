# Codebase audit — 2026-09-15

## Scope and audit status

This audits the isolated `codex/complementary-special` worktree at base commit `629d9da`, **including all prior uncommitted session changes**. Production code was not edited in this audit. Three parallel agents were launched at the user’s request; they then hit a service usage limit. Their surviving reproductions were retained, checked, and supplemented locally. This is a broad, evidence-backed audit, not a claim of exhaustive line-by-line review or proof that all remaining code is correct.

All Python source files were syntax-checked and scanned for undefined/redefined-name errors. All default-selected tests were collected. Runtime execution was targeted by subsystem rather than all 4,350 selected tests. No GPU, multi-device, clean-install or exhaustive benchmark run was performed in this audit.

Primary environment: Python 3.13.12, JAX/jaxlib 0.10.0, Flax 0.12.8. **The declared JAX range is >=0.9,<0.10**, so the full targeted suite is not a supported-version certification. All four reproduction scripts were also run using the existing JAX 0.9.2 environment; the reported runtime failures reproduce there. Syntax failure is version-independent. See `environment.json` and the per-area `results_jax092.txt` files.

## Priority summary

P1 = high-impact wrong numerical/probabilistic result or unusable named feature. P2 = incorrect bounded API behavior, runtime failure on an optional path, or training-quality issue. Findings are not scored as security vulnerabilities.

| ID | Severity | Finding | Source |
|---|---|---|---|
| I01 | P1 | Gaussian IMH samples and evaluates different proposals | `probjax/inference/mcmc/imh.py:370` |
| I02 | P1 | UKF sigma-point rules do not reproduce the input covariance | `probjax/inference/filtering/unscented_kalman_filter.py:63` |
| I03 | P1 | UKF observation update omits process noise from sigma points | `probjax/inference/filtering/unscented_kalman_filter.py:286` |
| I04 | P2 | UKF log-likelihood is missing the Gaussian normalization | `probjax/inference/filtering/unscented_kalman_filter.py:310` |
| I05 | P2 | UKF callable observation covariance is not evaluated | `probjax/inference/filtering/unscented_kalman_filter.py:293` |
| S01 | P1 | Pareto random samples have an extra location shift | `probjax/stats/continuous/pareto.py:87` |
| S02 | P1 | Generalized Pareto defaults are singular and support is not enforced | `probjax/stats/continuous/genpareto.py:60` |
| S03 | P1 | Transformed CDF and quantile assume increasing bijectors | `probjax/stats/transformed.py:191` |
| S04 | P2 | Binomial CDF treats non-integer thresholds as continuous | `probjax/stats/discrete/binomial.py:47` |
| S05 | P2 | Binomial entropy ignores the trial count | `probjax/stats/discrete/binomial.py:181` |
| S06 | P2 | Poisson entropy can be negative | `probjax/stats/discrete/poisson.py:148` |
| S07 | P2 | Truncated normal delegates public methods to nonexistent backend APIs | `probjax/stats/continuous/truncnorm.py:76` |
| U01 | P1 | Default differential entropy and two alternative estimators are incorrect | `probjax/utils/stats.py:135` |
| U02 | P1 | PCG is not invariant to right-hand-side scaling | `probjax/utils/linalg.py:588` |
| U03 | P1 | Lanczos logdet has a one-step dimension bug and an invalid exactness guarantee | `probjax/utils/linalg.py:505` |
| U04 | P2 | Matrix structure predicates return wrong answers | `probjax/utils/linalg.py:87` |
| U05 | P2 | Optional multistep solver module cannot be imported | `probjax/utils/odeutil/solvers/multistep.py:34` |
| C01 | P1 | Nested JIT loses stochastic sites and breaks log-joint evaluation | `probjax/core/interpreters/ppl/joint_sample.py:39` |
| N01 | P2 | Explicit transformer RNG is reused across dropout layers | `probjax/nn/nets/transformer.py:409` |
| N02 | P1 | FlashAttention3 primitive implementations call undefined functions | `probjax/nn/pallas_kernels/kernels/flash_attention3.py:160` |

## Detailed findings

### I01 — Gaussian IMH samples and evaluates different proposals (P1)

Source: `probjax/inference/mcmc/imh.py:370`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** For diagonal covariance [4], the sampler uses standard deviation 2 but proposal_logpdf(0) returns -2.305233 instead of -1.612086.

**Cause:** The sampling branch takes sqrt(cov); the density branch passes cov directly as the normal scale.

**Impact:** The Hastings correction is wrong, so the chain can target the wrong stationary distribution. This is not merely a normalization offset: the quadratic term also changes.

**Recommended fix:** Use sqrt(cov) in the diagonal proposal density, and validate positive covariance.

**Regression coverage to add:** Compare dense/diagonal representations of the same Gaussian, including off-center densities and a stationary-moment test.

### I02 — UKF sigma-point rules do not reproduce the input covariance (P1)

Source: `probjax/inference/filtering/unscented_kalman_filter.py:63`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** For P=[[2,1],[1,2]], Merwe and Julier reconstruct [[2.5,.866],[.866,1.5]]. The simplex rule reconstructs approximately 1.333 I.

**Cause:** Lower-Cholesky rows are used as point displacements rather than columns. The simplex construction also does not have the claimed whitened covariance.

**Impact:** Even the identity transform corrupts a correlated covariance; downstream nonlinear filtering and uncertainty estimates are wrong.

**Recommended fix:** Transpose Cholesky displacements; construct and verify a centered, whitened simplex before applying the covariance factor.

**Regression coverage to add:** For every rule, reconstruct mean/covariance for random correlated SPD matrices at dimensions 1, 2, 5 and 20.

### I03 — UKF observation update omits process noise from sigma points (P1)

Source: `probjax/inference/filtering/unscented_kalman_filter.py:286`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** Identity transition/observation, initial P=1, Q=1, R=1, y=1: UKF gives posterior mean .5 and covariance 1.5; the exact Kalman update gives 2/3 and 2/3.

**Cause:** The prediction adds Q to cov1_, but observations and cross-covariance still use the pre-Q propagated points.

**Impact:** Incorrect gains and posterior uncertainty even in a one-dimensional linear problem.

**Recommended fix:** Regenerate sigma points from the predicted mean/covariance or implement a consistent augmented-noise UKF.

**Regression coverage to add:** Compare UKF to the linear Kalman filter for nonzero Q, correlated states and multi-step observations.

### I04 — UKF log-likelihood is missing the Gaussian normalization (P2)

Source: `probjax/inference/filtering/unscented_kalman_filter.py:310`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** With Q=0, P=R=1, residual 1, reported log-likelihood is -.596574; the normalized value is -1.515512.

**Cause:** The expression omits observation_dim * log(2*pi).

**Impact:** Incorrect evidence/log-likelihood values, especially across models with different observation dimensions.

**Recommended fix:** Include the normalization, or explicitly expose the result as an unnormalized potential instead of log_likelihood.

**Regression coverage to add:** Compare to multivariate_normal.logpdf in dimensions 1 and greater than 1.

### I05 — UKF callable observation covariance is not evaluated (P2)

Source: `probjax/inference/filtering/unscented_kalman_filter.py:293`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** Providing observation_covariance=lambda t: eye(1) raises TypeError when an array is added to a function.

**Cause:** Transition covariance callables are evaluated, but the observation covariance is passed through unchanged.

**Impact:** Time-dependent observation noise cannot be used through the kernel.

**Recommended fix:** Resolve callable observation covariance at the observation time, or reject it explicitly if unsupported.

**Regression coverage to add:** Compare constant-array and equivalent callable covariance over several times.

### S01 — Pareto random samples have an extra location shift (P1)

Source: `probjax/stats/continuous/pareto.py:87`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** For b=1, alpha=3, 20,000 draws have minimum about 2 and mean 2.4997; support starts at 1 and the mean is 1.5.

**Cause:** jax.random.pareto already has support [1,infinity); adding one shifts every draw.

**Impact:** Sampling disagrees with the distribution density/CDF/moments and biases any simulation using it.

**Recommended fix:** Return samples*b; pin the random backend support convention with a regression.

**Regression coverage to add:** Check support, analytic moments with Monte Carlo tolerance, and CDF probability-integral transforms.

### S02 — Generalized Pareto defaults are singular and support is not enforced (P1)

Source: `probjax/stats/continuous/genpareto.py:60`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** At the default c=0, PDF/CDF/SF raise ZeroDivisionError and PPF/ISF return NaN. At c=.5 and x=-1, PDF=8 and CDF=-3 instead of zero.

**Cause:** Expressions divide by c without the exponential-limit branch and evaluate outside distribution support.

**Impact:** Default construction is unusable for core probability operations; out-of-support probabilities can be invalid.

**Recommended fix:** Implement stable c->0 limits and support-aware branches, including c<0 finite endpoints.

**Regression coverage to add:** Test zero, tiny positive/negative shape and both support boundaries, under JIT and differentiation.

### S03 — Transformed CDF and quantile assume increasing bijectors (P1)

Source: `probjax/stats/transformed.py:191`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** For transformed(norm(), lambda x:-x), CDF(1)=.158655 rather than .841345, and PPF(.9)=-1.281551 rather than +1.281551.

**Cause:** CDF always uses the base CDF and quantile always uses q, without orientation handling.

**Impact:** Wrong probabilities/quantiles for decreasing univariate transformations, even a negated standard normal.

**Recommended fix:** Use a monotonicity contract to select CDF/SF and q/1-q; reject unknown orientation for these operations.

**Regression coverage to add:** Test positive and negative affine transforms, reversed support and CDF/PPF round trips.

### S04 — Binomial CDF treats non-integer thresholds as continuous (P2)

Source: `probjax/stats/discrete/binomial.py:47`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** Binomial(n=4,p=.5).cdf(.5)=.1604695 rather than .0625.

**Cause:** The incomplete-beta formula receives k without first flooring it to a discrete threshold.

**Impact:** The CDF is not the required step function; interval probabilities at fractional bounds are wrong.

**Recommended fix:** Floor finite thresholds and retain correct behavior outside [0,n].

**Regression coverage to add:** Check equality of CDF(k+fraction) and CDF(k) across integer intervals.

### S05 — Binomial entropy ignores the trial count (P2)

Source: `probjax/stats/discrete/binomial.py:181`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** For p=.5, entropy is 1.386294 for both n=1 and n=4; correct values are .693147 and 1.407532.

**Cause:** The formula is Bernoulli entropy plus log(2), with no dependence on n.

**Impact:** Incorrect entropy-based objectives and diagnostics; degenerate probabilities also need zero-times-log-zero handling.

**Recommended fix:** Implement a justified discrete summation/approximation with documented accuracy and endpoint handling.

**Regression coverage to add:** Test n=1, several larger n, p=0/1 and SciPy parity.

### S06 — Poisson entropy can be negative (P2)

Source: `probjax/stats/discrete/poisson.py:148`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** At rate=10, entropy is -13.025851 instead of 2.561410.

**Cause:** The formula omits the expectation of log(k!).

**Impact:** Invalid discrete entropy and incorrect entropy gradients.

**Recommended fix:** Use a stable summation or a validated small/large-rate approximation.

**Regression coverage to add:** Test nonnegativity and reference accuracy across rates including zero.

### S07 — Truncated normal delegates public methods to nonexistent backend APIs (P2)

Source: `probjax/stats/continuous/truncnorm.py:76`. Reproduction: [stats_inference/reproduce.py](stats_inference/reproduce.py); outputs: [stats_inference/results.txt](stats_inference/results.txt) and [stats_inference/results_jax092.txt](stats_inference/results_jax092.txt).

**Observed:** PPF, ISF, mean, variance and entropy all raise AttributeError for a=-1,b=1 in both tested JAX versions.

**Cause:** jax.scipy.stats.truncnorm does not provide those methods, although the wrapper calls them.

**Impact:** Advertised distribution operations are unusable.

**Recommended fix:** Implement stable truncated-normal formulas or make unsupported operations explicit until implemented.

**Regression coverage to add:** Exercise every public method against the installed supported backend, including tail truncations.

### U01 — Default differential entropy and two alternative estimators are incorrect (P1)

Source: `probjax/utils/stats.py:135`. Reproduction: [numerics/reproduce.py](numerics/reproduce.py); outputs: [numerics/results.txt](numerics/results.txt) and [numerics/results_jax092.txt](numerics/results_jax092.txt).

**Observed:** On linspace(.001,.999,100), auto/Ebrahimi gives -3.735061 vs SciPy .008048; Van Es gives -1.436549 vs .063873; Correa gives NaN vs -.031353. Vasicek agrees.

**Cause:** Ebrahimi uses adjacent gaps instead of the requested window gaps; Van Es takes logsumexp instead of the harmonic sum; Correa indexes an unpadded window.

**Impact:** The default estimator is badly biased on a simple nearly uniform sample.

**Recommended fix:** Implement each estimator from its defining window formula and add independent reference checks.

**Regression coverage to add:** Compare all methods on sorted/unsorted samples, axes, window lengths and large-n near-uniform data.

### U02 — PCG is not invariant to right-hand-side scaling (P1)

Source: `probjax/utils/linalg.py:588`. Reproduction: [numerics/reproduce.py](numerics/reproduce.py); outputs: [numerics/results.txt](numerics/results.txt) and [numerics/results_jax092.txt](numerics/results_jax092.txt).

**Observed:** For A=I and B=1e-4*ones((2,1)), PCG runs 200 iterations and returns about 4e-10 with relative residual .999996. The same problem at unit scale converges in one iteration.

**Cause:** Valid small dot products are replaced with 1 using an absolute machine-epsilon threshold.

**Impact:** Legitimate small systems fail. The large-operator Kalman solve currently discards PCG convergence diagnostics.

**Recommended fix:** Use scale-aware breakdown handling that distinguishes valid nonzero denominators from zero/nonpositive curvature; propagate nonconvergence.

**Regression coverage to add:** Rescale identical SPD problems over many orders of magnitude and assert solution/residual invariance.

### U03 — Lanczos logdet has a one-step dimension bug and an invalid exactness guarantee (P1)

Source: `probjax/utils/linalg.py:505`. Reproduction: [numerics/reproduce.py](numerics/reproduce.py); outputs: [numerics/results.txt](numerics/results.txt) and [numerics/results_jax092.txt](numerics/results_jax092.txt).

**Observed:** logdet([[2]],num_steps=1) returns -33.845627 vs log(2)=.693147. Full two-step evaluation of [[2,1],[1,2]] returns 2.197225 vs log(3)=1.098612.

**Cause:** One-step construction creates a nonempty off-diagonal and an unintended 2x2 matrix. Separately, one deterministic probe estimates a quadratic form, not an exact trace even at full Lanczos depth.

**Impact:** Incorrect determinants and potentially biased large-operator Kalman likelihoods, whose default path uses this deterministic probe.

**Recommended fix:** Fix tridiagonal sizes and breakdown; use explicit multi-probe stochastic trace estimation with accuracy diagnostics, removing the full-depth exactness claim.

**Regression coverage to add:** Test dimension/step=1, diagonal matrices, rotated SPD matrices and probe-averaged convergence.

### U04 — Matrix structure predicates return wrong answers (P2)

Source: `probjax/utils/linalg.py:87`. Reproduction: [numerics/reproduce.py](numerics/reproduce.py); outputs: [numerics/results.txt](numerics/results.txt) and [numerics/results_jax092.txt](numerics/results_jax092.txt).

**Observed:** An upper-triangular [[1,2],[0,1]] returns False with lower=False. A batch [I,2I] is reported [False,False] by is_diagonal_matrix.

**Cause:** The upper branch omits the equality comparison; jnp.diag does not reconstruct a batched diagonal.

**Impact:** Wrong dispatch/validation decisions for structured matrices; scalar tests do not cover batching.

**Recommended fix:** Compare both triangular projections to A and construct batched diagonals using an identity mask with the requested axes.

**Regression coverage to add:** Cover upper/lower, batched, rectangular and non-default-axis cases.

### U05 — Optional multistep solver module cannot be imported (P2)

Source: `probjax/utils/odeutil/solvers/multistep.py:34`. Evidence: [syntax_checks.json](syntax_checks.json).

**Observed:** Compiling the source raises SyntaxError: from __future__ imports must occur at the beginning of the file (line 34).

**Cause:** Coefficient helpers were placed before a future import.

**Impact:** Direct use of this optional module fails. Normal test collection succeeds because it does not import this path.

**Recommended fix:** Move the future import to the top and ensure annotation types are defined.

**Regression coverage to add:** Compile all source files and smoke-import optional pure-Python modules.

### C01 — Nested JIT loses stochastic sites and breaks log-joint evaluation (P1)

Source: `probjax/core/interpreters/ppl/joint_sample.py:39`. Reproduction: [core/reproduce.py](core/reproduce.py); outputs: [core/results.txt](core/results.txt) and [core/results_jax092.txt](core/results_jax092.txt).

**Observed:** A one-site model gives {x: sample} when plain, but joint_sample(jax.jit(model)) returns {}. log_joint_fn of the jitted model raises AttributeError about .items. An explicit rv_p.bind reproduces it too; conditioning provides a passing control.

**Cause:** The stochastic interpreter/reducer composition does not preserve site collection and state types across nested call/JIT boundaries.

**Impact:** Common model compilation changes probabilistic transformation semantics.

**Recommended fix:** Propagate the same stochastic visitor and compatible reducer state into nested jaxprs; test both high-level sampling and explicit primitives.

**Regression coverage to add:** Compare plain/JIT/nested-JIT joint_sample, trace, log_joint and condition for identical named-site programs.

### N01 — Explicit transformer RNG is reused across dropout layers (P2)

Source: `probjax/nn/nets/transformer.py:409`. Reproduction: [nn/reproduce.py](nn/reproduce.py); outputs: [nn/results.txt](nn/results.txt) and [nn/results_jax092.txt](nn/results_jax092.txt).

**Observed:** A three-layer transformer forwards the identical key [0,8] to each dense dropout call.

**Cause:** The supplied key is forwarded repeatedly without splitting/folding in layer and operation indices.

**Impact:** Equal-shaped layers receive correlated dropout masks, unlike independent per-layer dropout. This affects the explicit-key path, not necessarily stored RNG streams.

**Recommended fix:** Derive separate keys per layer and stochastic suboperation while preserving deterministic replay of a whole call.

**Regression coverage to add:** Check whole-call reproducibility and within-call mask independence with explicit RNG.

### N02 — FlashAttention3 primitive implementations call undefined functions (P1)

Source: `probjax/nn/pallas_kernels/kernels/flash_attention3.py:160`. Reproduction: [nn/reproduce.py](nn/reproduce.py); outputs: [nn/results.txt](nn/results.txt) and [nn/results_jax092.txt](nn/results_jax092.txt).

**Observed:** The forwarding implementation raises NameError for _run_flash_forward_raw; _run_flash_backward_raw is likewise undefined. Static symbol checks confirm no definitions.

**Cause:** The new primitive adapters reference missing raw forward/backward wrappers.

**Impact:** The affected Flash3 backend cannot execute these implementations. GPU execution itself was not tested; the missing Python names were reproduced on CPU in both JAX versions.

**Recommended fix:** Restore/import raw implementations and test eligible GPU forward/backward execution, not only backend availability guards.

**Regression coverage to add:** Add symbol/linkage smoke checks plus actual GPU forward/gradient tests.

## Coverage and test interpretation

- Test collection: 4,350 selected out of 4,415, with 65 deselected by the default benchmark/mesh/GPU policy. Collection is not execution.
- Core/inverse/RV selection: 450 passed, 12 expected failures; one float64-to-float32 warning. See `core/tests.log`.
- NN/training/generative selection: 71 passed. See `nn/tests.log`.
- Filtering/inference/distribution selection: 235 passed. See `stats_inference/tests.log`.
- Total executed in these three selections: 756 passed and 12 expected failures. This excludes historical tests from earlier turns.
- Surviving agent probes were independently rerun, including under JAX 0.9.2. These scripts print expected/actual values and are audit evidence, not a substitute for committed regression assertions.
- No claim that the passing tests cover the defects above: the discrepancies explicitly demonstrate missing property/reference tests.

## Additional imperfections and unresolved questions

1. **Environment/CI mismatch:** most targeted tests currently run against excluded JAX 0.10.0. Run the suite on declared supported versions before release; separately decide whether to broaden the supported range.
2. **Type-hint name resolution:** `InterpolationScheduleProtocol` is referenced as a string without definition/import in flow/mean-flow losses. This is lower priority than numerical defects, but runtime type-hint resolution can fail. Static results are in `static_checks.json`.
3. **Mean-flow time conditioning:** DiffusionTransformer (and the analogous TimeMLP pattern) adds embeddings of t and r, making the representation invariant to swapping the times. Whether that restriction is intended should be documented; it is not counted as a confirmed wrong-result bug here.
4. **Previously documented limitations remain:** lazy distribution export inside outer JIT, coarse diffusion SDE instability, weak terminal mixing in categorical linear/sigmoid defaults, and unverified nonuniform-grid order of exponential AB3. See the prior diffusion benchmark report; this audit did not re-run that performance study.
5. **GPU/sharding coverage:** Pallas/Mosaic backends were statically inspected and the Flash3 adapter failure reproduced in Python, but GPU kernel numerical correctness, multi-device transformations and performance remain unverified.
6. **Breadth limitations:** SMC/VI and special-function tests were inspected/selected where useful; not every kernel, support family, dtype, backend, shape, derivative order, or optimizer state transition was exhaustively exercised.

## Suggested repair sequence

1. Repair Gaussian IMH and UKF first: they can silently invalidate posterior inference.
2. Repair numerical utility contracts (PCG/logdet/entropy), including diagnostics propagation into filtering.
3. Repair distribution sampling, support, quantiles and entropy; use cross-method identities and reference values.
4. Repair nested probabilistic JIT transformations and optional-module/backend linkage failures.
5. Split explicit NN RNGs per stochastic operation, resolve annotation issues, then expand declared-version and GPU CI.

For each repair, add the indicated failing regression before changing production code. Keep independent fixes in separately reviewable changes; do not bundle all numerical semantics into one refactor.

## Reproduce

Run from the worktree root:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python audits/2026-09-15/core/reproduce.py
PYTHONPATH=. JAX_PLATFORMS=cpu python audits/2026-09-15/nn/reproduce.py
PYTHONPATH=. JAX_PLATFORMS=cpu python audits/2026-09-15/numerics/reproduce.py
PYTHONPATH=. JAX_PLATFORMS=cpu python audits/2026-09-15/stats_inference/reproduce.py
```

Replace `python` with `/home/manuel/probjax/.venv/bin/python` to reproduce the JAX 0.9.2 checks on this machine. Raw logs and machine-readable findings are retained alongside this report.
