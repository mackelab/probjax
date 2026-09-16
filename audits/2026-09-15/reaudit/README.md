# Re-audit of the current worktree

**Follow-up:** all nine findings have now been addressed. See [fixes and validation](FIXES.md). The observations and failure logs below describe the pre-fix state.

This pass found **nine additional reproducible issues**. These are newly
identified findings, not a claim that all nine were introduced by the previous
fixes. Production code was not changed during this audit.

The earlier statement that the 20 reported findings were fixed applies to their
covered cases. The re-audit exposes missing coverage, including remaining defects
in those same numerical APIs.

## Findings, ordered by priority

### R01 — P1: logdet differentiation fails on a scaled identity

Source: `probjax/utils/linalg.py:473` and `:483`.

`grad(lambda s: lanczos_logdet(s * eye(2), num_steps=2))(2.)` returns **NaN**;
the exact derivative is **1**. Reproduced in FP64 and with dimensions 1, 2, 3,
and 20 in FP32. Forward values are correct, so forward-only tests miss this.

The recurrence differentiates `norm(w)` at zero during Krylov breakdown; masking
later values does not make that derivative defined. Eigensystem differentiation
also needs care at repeated Ritz values. This affects differentiable logdet use,
including the matrix-free filtering route in `kalman_filter.py:90`.

Recommended fix: define breakdown-safe differentiation, or provide a deliberate
logdet JVP/VJP based on linear solves and the selected trace estimator. Validate
identity, repeated eigenvalues, and generic SPD matrices against dense gradients.

### R02 — P1: generalized-Pareto sampling rejects explicit parameters

Source: `probjax/stats/continuous/genpareto.py:114`.

`genpareto.rvs(key, c=.5, shape=(5,))` raises
`TypeError: ... got multiple values for argument 'shape'`.

The private sampler places `shape` before distribution parameters; the random
variable primitive supplies distribution parameters positionally and `shape` by
keyword. An AST check found this ordering only in this distribution. Batched
parameters fail too, before their broadcasting can even be tested.

Recommended fix: align the sampler signature with the other distributions and
explicitly generate sample shape plus broadcast parameter shape. Test direct,
frozen, named, JIT, and batched sampling.

### R03 — P2: valid truncated-normal quantiles become NaN near a bound

Source: `probjax/stats/continuous/truncnorm.py:185`.

In FP64, `truncnorm.ppf(1e-100, a=-1., b=1.)` returns **NaN**, whereas the
representable quantile is **-1**. The same failure occurs for bounds `[8, 9]`.
The probability itself is representable in FP64; this is not probability
underflow.

Bisection reaches a representable boundary with `logcdf=-inf`; the unconditional
Newton correction evaluates an infinite residual times a zero factor.

Recommended fix: safeguard the correction and retain an in-support bracketed
answer when no representable interior quantile exists. Add symmetric ISF and
finite-density endpoint tests.

### R04 — P2: truncated-normal quantile curvature is wrong

Source: `probjax/stats/continuous/truncnorm.py:182`.

For the untruncated standard normal at `q=.7`, the second derivative of PPF is
**-4.1087195**, but the analytic result is **+4.3378265**. The error persists in
FP64. At `.1`, it returns **-56.9806** instead of **-41.6093**.

Stopping differentiation through the root and applying one Newton correction
supplies a first derivative, but its derivative omits the root's dependence.
The prior fix report explicitly left higher-order accuracy unvalidated; this
pass confirms a correctness failure, rather than merely missing coverage.

Recommended fix: use a custom implicit differentiation rule whose root dependence
remains correct under higher-order AD; otherwise explicitly reject unsupported
higher-order differentiation rather than returning wrong values silently.

### R05 — P2: PCG falsely reports convergence after norm underflow

Source: `probjax/utils/linalg.py:538` and the final residual calculation.

For identity `A` and FP32 `b=[[1e-20], [1e-20]]`, PCG returns **zero** in zero
iterations with **converged=True, rel_residual=0**. The actual relative residual
is **1**. Both the input and correct solution are representable.

Squared residuals underflow, which disables the iteration and also corrupts
final diagnostics. At `b=1e20`, the squared norms overflow and no iterations
execute either, though that case reports failure.

Recommended fix: normalize RHS columns or use scaled norms/recurrences, retaining
accurate convergence checks across representable scales.

### R06 — P2: one-sided truncated-normal variance remains inaccurate in FP32

Source: `probjax/stats/continuous/truncnorm.py:275`.

For bounds `[20, inf)`, variance is **0.00332642**, versus reference
**0.00246326**, about **35% relative error**. `[8, inf)` is also inaccurate,
although less severely.

The finite-interval quadrature improvement is bypassed at infinite bounds;
subtracting the large raw moments still loses the small variance. This is a
concrete instance of the tail precision limitation noted in the prior report.

Recommended fix: use centered tail moments, stable Mills-ratio expansions, or a
controlled tail integration scheme. Test both one-sided tails in FP32 and FP64.

### R07 — P2: optional-RNG helper breaks plain functions

Source: `probjax/nn/utils.py:194` and `:202`.

`call_with_optional_rng(lambda x: x + 1, 1., rng=key)` raises an unexpected-`rng`
`TypeError`, despite its documented module/function support.

Inspecting the function's generic `__call__` exposes `**kwargs`, not the function's
actual signature. Caching support by `type(function)` would also conflate distinct
function signatures after fixing introspection.

Recommended fix: inspect functions/partials directly and cache by callable identity
where signatures can vary within one type. Retain per-type caching for ordinary
module classes where appropriate.

### R08 — P2: constructor filtering drops valid forwarded sharding options

Source: `probjax/nn/utils.py:123`; used by MLP builders in `nn/nets/simple.py`.

For a layer constructor `__init__(self, **kwargs)`, filtering
`kernel_sharding=('data', None)` returns **{}**. Such forwarding subclasses accept
the option, so custom modules silently lose metadata that standard layers receive.

The filter checks only parameter names and ignores `VAR_KEYWORD` acceptance.

Recommended fix: preserve options when the effective callable accepts `**kwargs`;
inspect partials' effective signatures instead of discarding bindings. Add an
integration test with a forwarding custom dense layer.

### R09 — P2: generalized-Pareto mode selects the wrong endpoint

Source: `probjax/stats/continuous/genpareto.py:149`.

For `c=-2, loc=0, scale=1`, `mode()` returns **0**, where the density is 1.
The density increases to infinity at the upper endpoint **0.5**, which is the
mode. Returning `loc` is only appropriate for the decreasing-density regime
(with nonunique modes at `c=-1`).

Recommended fix: select `loc-scale/c` for `c < -1` and preserve broadcasting.

## Evidence and scope

- [Expected-behavior tests](test_new_findings.py): **9 failures**, each reproducing
  one finding. These intentionally failing audit tests live outside the normal
  `tests/` collection; no expected-failure markers conceal the defects.
- [Failure log](new_findings_tests.log). All nine expected-behavior checks also
  fail under [supported JAX 0.9.2](new_findings_jax092.txt).
- [Existing regression rerun](existing_regressions.log): the previous 65 audit
  regression cases still pass, demonstrating the coverage gaps.
- [Numerical/core probes, JAX 0.10](probe.txt) and [JAX 0.9.2](probe_jax092.txt).
- [API/module probes, JAX 0.10](edge_probe.txt) and
  [JAX 0.9.2](edge_probe_jax092.txt).

This pass targeted recently edited numerical routines, distribution sampling,
first/higher-order AD, nested named-site composition, callback stopping, and NN
constructor/RNG helpers. Nested dependent named-site sampling/log density and
host/IO stop-on-begin checks did not reveal new failures. This was not another
exhaustive sweep of every source file. No GPU execution or performance benchmark
was performed.
