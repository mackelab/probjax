# Re-audit fixes

All nine findings are addressed in the isolated `codex/complementary-special`
worktree. The original re-audit logs remain unchanged as pre-fix evidence.

| Finding | Implemented correction |
| --- | --- |
| R01: NaN logdet gradients | Use a safe norm at Krylov breakdown and a custom matrix-log quadratic-form JVP with logarithmic divided differences. Equal eigenvalues use the analytic limit instead of differentiating eigenvectors. Dense inputs use a symmetric extension, giving entrywise gradients consistent with the SPD log determinant. |
| R02: generalized-Pareto sampling | Move `shape` after distribution parameters and generate sample dimensions followed by broadcast parameter dimensions. |
| R03: NaN quantiles at a bound | Return the bracketed quantile directly, removing the unguarded Newton correction. Probabilities too small to resolve an interior point return the representable bound. |
| R04: incorrect quantile curvature | Supply an implicit custom JVP depending on the quantile itself. Its parameter and probability derivatives compose under higher-order AD. |
| R05: false PCG convergence | Normalize each RHS/initial-guess column before iterations and diagnostics, then rescale its solution. Squared residuals no longer underflow or overflow solely because RHS magnitudes differ. |
| R06: one-sided tail variance | Extend centered quadrature to one-sided tails beyond five standard deviations, truncating after a relative density drop of at least `exp(-48)`. Central moments avoid subtraction of large raw moments. |
| R07: optional RNG functions | Inspect the actual callable signature. Functions, methods and partials bypass the per-type cache; positional-only `rng` parameters do not imply keyword support. |
| R08: forwarding constructors | Preserve options accepted through `**kwargs` and inspect partials' effective signatures. Custom MLP forwarding layers receive their metadata. |
| R09: generalized-Pareto mode | Return the upper endpoint for shape below -1, with broadcast support. |

## Verification

[tests/test_reaudit_fixes.py](../../../tests/test_reaudit_fixes.py) contains the
nine original cases plus wider regression coverage:

- Dense and matrix-free logdet gradients at dimensions 1, 3, and 20, including
  repeated spectra; a generic SPD dense gradient against the inverse; and a
  finite-depth stochastic directional derivative against finite differences.
- First, second, and third quantile derivatives for the normal limit, bounded
  central intervals, and a tail interval.
- Tiny PPF/ISF probabilities, finite support, and one-sided tail variances on
  both sides at 5, 8, 20, and 50 standard deviations.
- Mixed RHS scales `1e-20`, `1`, `1e20`, and zero, including initial guesses and JIT.
- Frozen, batched, named, and JIT generalized-Pareto sampling.
- Distinct functions/partials in RNG dispatch and custom MLP metadata forwarding.

Results:

- **106 audit regressions passed**: [105-case run](validation/regressions.log)
  plus the additional [matrix-free filter case](validation/filter_operator.log).
  The permanent suites now contain 41 re-audit cases and 65 previous audit cases.
- All **nine original reproductions pass on JAX 0.9.2**:
  [log](validation/jax092.txt). The main pytest environment uses JAX 0.10.0.
- **1,491 passed** across filtering, NN overrides/modules/fitting, JIT fitting,
  distributions, and transforms: [log](validation/integration.log). Four existing
  pytest parametrization deprecation warnings were emitted.
- Syntax/undefined-name checks and `git diff --check` pass.

## Numerical scope

The SLQ forward estimate, probe count, and finite-depth approximation remain as
before. Its new derivative follows that estimator; finite depth and finite
probe counts are not an exact logdet or exact trace guarantee. First derivatives
are tested at repeated spectra; higher-order logdet differentiation at repeated
spectra is not established by this change.

Quantile derivatives are tested through third order at interior probabilities.
At endpoints and below representable interior spacing, usual floating-point and
one-sided-derivative limitations apply. The quantile's implicit derivative does
not differentiate the finite bisection control flow.

PCG scaling addresses RHS dynamic range, not arbitrary operator ill-conditioning
or every possible underflow inside a user-supplied matvec/preconditioner. The
preconditioner is assumed linear, as required by PCG.

Tail quadrature has fixed shape under JIT. It retains recurrence for other wide
or unbounded intervals; this does not claim uniform error bounds for all extreme
parameters. No GPU runtime or speedup claim is made.
