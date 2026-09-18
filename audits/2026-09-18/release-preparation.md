# 0.2.0 release preparation

Prepared on `codex/release-docs` from main `dfcba8e`.

## Documentation and metadata

- Align package/runtime version at 0.2.0 (previously 0.1.1 versus 0.1.0).
- Replace stale Sphinx release claims, summarize changes since the actual v0.1.1
  tag, add migration and numerical solver guides, and document release checks.
- Correct training callback/JIT descriptions, dependency bounds and installation
  instructions. Distinguish main-branch docs from the published version and
  checked Python examples from stored notebook outputs.
- Document actual solver APIs. Earlier discussions of selectable adaptive ODE
  gradients and Jacobian stride controls are not implemented in this tree.

## RK4 correction

Executing the analytic decay gradient example exposed an incorrect third row
in the classical RK4 tableau: `[0, 1, 0.5, 0]` instead of `[0, 0.5, 0, 0]`.
Corrected it and removed the explicit RK loop's out-of-bounds extra iteration.
Regression checks use the single-step stability polynomial and analytic
parameter derivatives for traced and terminal-only solutions.

## Remaining release blocker: adaptive ODE adjoint

This minimal reproduction currently fails, independently of the documentation:

```python
import jax
import jax.numpy as jnp
from probjax.utils import odeint

def objective(rate):
    return odeint(
        lambda t, y, r: -r * y,
        jnp.array([1.0, 2.0]), jnp.linspace(0.0, 1.0, 17), rate,
        method="dopri5",
    )[-1].sum()

jax.jit(jax.grad(objective))(jnp.asarray(0.5))
```

`_odeint_fwd` places a Python StepSizeAdaptor object in the custom-VJP residuals,
which JAX rejects. An experimental local removal of that static residual exposed
another error: `_odeint_rev` calls the two-argument raveled filter with one
argument. That incomplete change was reverted. Further inspection shows the
reverse code indexes a trace without its initial state using the full time
array, and returns zero cotangents for drift parameter leaves. These latter
findings require dedicated reproduction and repair. Terminal-only adaptive
reverse gradients are explicitly unsupported in the current reverse rule.
Do not publish the candidate as having working adaptive ODE gradients.

## Validation

- 365 passed, 29 skipped, 311 deselected: complete documentation suite, new RK4
  regressions, and selected linear/RK4/Dormand–Prince ODE checks.
- All generated tutorial pages pass the converter freshness check.
- Strict Zensical build passed within the documentation suite.
- Isolated PEP 517 wheel and source build passed; both pass `twine check`.
- Wheel installed with its declared dependencies into a fresh Python 3.11
  environment outside the checkout; runtime/metadata versions match 0.2.0,
  distribution sampling and analytic ODE smoke checks pass.
- GPU execution was not tested. A complete release CI matrix is still required.
- No release tag, GitHub release or PyPI upload was created.

Local artifacts: `/tmp/probjax-release-020-final/`; validation log:
`/tmp/probjax-release-validation.log`. The candidate has an open release blocker
and is not publication-ready.
