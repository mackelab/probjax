# Test quality and efficiency audit

Scope: static inventory of 78 test files / 1,104 test functions before
parametrization, followed by focused execution of numerical, inverse, kernel
and test-runner paths. This is not a complete execution or semantic review of
every test in the repository.

## Findings and changes

1. **Invalid performance workload.** Eight compositions of `2*exp(x)+1`
   overflowed the benchmark inputs. Replaced this with `2*exp(x)-2` on a small
   neighborhood of its zero fixed point. Added ordinary, untimed finite-value,
   inverse recovery and analytic log-determinant checks. Runtime comparisons
   validate outputs before timing, warm both executables, alternate timing order,
   synchronize device completion and compare medians. No performance thresholds
   were relaxed. Benchmarks remain opt-in.
2. **Silent expected pass.** The vector-parameter kernel-matrix test passes
   without its non-strict xfail marker on the supported CPU environment. Removed
   the marker; future failures now fail CI. Added a strict, TypeError-limited
   expected failure for the open adaptive ODE adjoint blocker with analytic
   gradient assertions; it must be revisited when repaired.
3. **GPU CLI mismatch.** `--gpu` was accepted but ignored. It now selects the GPU
   backend like `--device gpu`; explicit contradictory options raise UsageError.
   Removed an unused module-level key and premature forced-CPU configuration.
   Collection checks verify the alias and conflict handling. GPU execution is
   unavailable locally and is not claimed as validated.
4. **Oracle independence.** Linear ODE references now use SciPy matrix
   exponentials in host FP64 instead of staging another JAX exponential. The
   seeded random matrix keeps the existing values but no longer depends on
   unrelated global NumPy RNG draws. Both ordinary and split-drift references
   use this independent oracle.
5. **CI waste and missing evidence.** Superseded PR runs are cancelled; main runs
   are preserved. Matrix fail-fast is disabled so one Python failure does not
   erase the other versions' evidence. Duration summaries and retained JUnit
   reports make slow tests and failures reviewable. Coverage execution remains
   a full suite, and Codecov failure handling is unchanged.

## Measurements

Fresh-process CPU runs in the same environment, pytest-reported runtime:

| Selection | Before | After | Coverage |
| --- | --- | --- | --- |
| Linear ODE matrix | 11.69 s | 11.45 s | 104 passed, 8 skipped in both |
| Beta/gamma inverse sweep | 2.40 s | unchanged | 200 passing cases retained |

The ODE timing difference is small and may be noise; the supported improvement
is oracle independence and avoiding reference-only JAX compilation. Do not infer
a full-suite speedup. The special-function sweep was already compilation-cached,
so reducing its 200 cases would sacrifice coverage for little gain.

Before final integration, the inverse/special/RK4 selection passed 230 tests,
including all opt-in inverse timings. The formerly xfailed kernel case passed
with `--runxfail` in 4.07 s. Final combined results are recorded below.

## Further work

- Use retained CI durations to prioritize large attention, inference and training
  sweeps; avoid shrinking statistical sample sizes without checking power.
- Some data-loader tests still use tight timing assertions. The earlier prefetch
  tests now use events; extend that pattern to end-to-end loader overlap tests.
- Mesh selection still uses legacy expression heuristics. Complex `-m`/`-k`
  expressions need a dedicated runner-selection regression suite.
- Resolve the adaptive ODE release blocker; current broad forward-error tests
  did not cover its backward path. Strict expected failure makes it visible.
- Full Python/GPU/mesh execution and CI runtime comparisons remain outstanding.

Final combined validation: **360 passed, 112 skipped, 230 deselected, 1 xfailed
in 48.54 s**. Selection covered inverse benchmarks, special functions, release
numerics, kernel matrix products, and ordinary/split linear ODE cases. The only
expected failure was the documented adaptive adjoint blocker.
