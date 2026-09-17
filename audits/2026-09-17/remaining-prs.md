# Remaining PR consolidation

PR #29 was merged at bf1c1b7 after the three required Python CI jobs passed,
using the explicitly authorized per-PR administrator review override. The full
coverage run passed 3,616 tests (837 skipped, 10 expected failures, one
pre-existing non-strict unexpected pass) and measured 73.57% coverage. Its
upload failed because Codecov returned HTTP 404, Repository not found.
Repository protection settings were not changed.

PR #18's autoregressive/MoE feature files already match the integrated code;
its remaining commit is reconciled without changing those files. PR #28's
curated notebook conversion, documentation navigation, and workflow changes
are included in this follow-up. Tutorial generation also runs in the coverage
job, and coverage.xml is retained as a GitHub artifact even if Codecov fails.
The external upload remains strict; the service setup failure is not suppressed.

Executing the Kalman tutorial exposed LinearOperator multiplication errors:
JIT tracers were rejected by a runtime ArrayLike check, right-hand dense
composition reversed the factors, left-vector multiplication used the wrong
operator, and rectangular shape metadata was transposed. Products now preserve
linear algebra order and dimension conventions, with dtype retained in dense
conversion and transpose construction.

Validation: 93 operator/filter/reaudit tests passed; all six selected Kalman
documentation checks passed after the fix. The earlier full documentation run
passed 240 cases and skipped four, with only the now-fixed tutorial failing.
New regressions cover rectangular vector/matrix products in eager/JIT mode,
and compiled operator-vs-dense Kalman predictions, updates, and gradients.
