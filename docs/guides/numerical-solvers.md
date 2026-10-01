# ODE and SDE integration

Both solvers take a state array or pytree and return a trajectory with a leading
axis for the requested times. Pass `collect_trace=False` for just the terminal
state. Parameters go through positional `*args`; use a closure or partial for
configuration. Keep method names and controller configuration static under JIT.

## Choosing an ODE grid

Fixed-step methods such as `rk4` step on the supplied grid. Adaptive methods such
as `dopri5` choose internal steps and return values at the requested times.
`StepSizeAdaptor` controls tolerances for adaptive methods; fixed-step methods
ignore it.

```python
import jax
import jax.numpy as jnp
from probjax.utils import odeint
from probjax.utils.odeutil import StepSizeAdaptor

def decay(t, y, rate):
    return -rate * y

ts = jnp.linspace(0.0, 1.0, 17)
y0 = jnp.array([1.0, 2.0])
trajectory = odeint(
    decay, y0, ts, 0.5, method="dopri5",
    step_size_adaptor=StepSizeAdaptor(rtol=1e-4, atol=1e-5),
)
assert trajectory.shape == (17, 2)
assert jnp.allclose(trajectory[-1], y0 * jnp.exp(-0.5), rtol=1e-3)
terminal = odeint(decay, y0, ts, 0.5, collect_trace=False)
assert terminal.shape == y0.shape
```

The default computation dtype is float32. For float64, enable JAX x64 before
creating arrays and pass `dtype=jnp.float64`. Tight tolerances alone cannot
recover precision unavailable in the chosen dtype. Compare against a finer grid
or tighter tolerances for the quantity you actually need.

## Gradients

Fixed-grid integration differentiates the discrete stepping computation and
supports the `check_points` option for checkpointed grid scans when collecting
the trajectory. Its entries are nested scan lengths whose product equals the
number of grid intervals, not checkpoint indices. Adaptive ODE
integration currently uses a custom reverse-mode adjoint, but release preparation
found failures in its backward path (controller residuals and filtered-state
handling). Treat adaptive gradients as a release blocker; use a converged
fixed-grid solve for differentiable examples below. There is **no public
`adjoint=` or gradient-mode selector** in this release, and forward-mode AD
through that custom VJP is unsupported. Any repaired adjoint also needs gradient convergence checks separate from the
forward solution.

```python
def terminal_sum(rate):
    return jnp.sum(odeint(decay, y0, ts, rate, collect_trace=False))

gradient = jax.jit(jax.grad(terminal_sum))(jnp.asarray(0.5))
assert jnp.allclose(gradient, -jnp.sum(y0) * jnp.exp(-0.5), rtol=1e-3)
```

## SDEs and adaptive noise

For fixed-step integration, select a method such as `euler_maruyama` or
`milstein`. Diffusion outputs determine noise layout: scalar/vector outputs
represent diagonal noise; matrices can represent full, rectangular noise.

Supplying an SDE adaptor enables adaptive Euler–Maruyama step doubling,
regardless of `method`. This adaptive path currently supports diagonal noise
only.

| Controller | Behavior on rejection | Use |
| --- | --- | --- |
| `StrongStepSizeAdaptor` | Refines one keyed Brownian path | Pathwise comparisons and sample-wise objectives |
| `WeakStepSizeAdaptor` | Draws fresh noise | Distributional experiments; check bias and convergence for your problem |

```python
from probjax.utils import sdeint
from probjax.utils.sdeutil import StrongStepSizeAdaptor

def noise(t, y, rate):
    return jnp.full_like(y, 0.1)

paths = sdeint(
    jax.random.key(0), decay, noise, y0, ts, 0.5,
    step_size_adaptor=StrongStepSizeAdaptor(rtol=1e-2, atol=1e-3),
)
assert paths.shape == (17, 2)
assert jnp.all(jnp.isfinite(paths))
```

The adaptive SDE inner loop is bounded by `max_inner_steps` and supports reverse
AD through its scan. A larger budget costs more work, even after an interval
finishes. Budget or step-size boundary hits can force completion outside the
requested tolerance. Inspect eager warnings and test convergence; host warnings
are skipped while tracing under `jit`/`vmap`. Reusing a key alone does not make
weak-controller retries path-consistent.

Use `return_brownian=True` with `collect_trace=True` to also return the Brownian
trace. Low-level Brownian paths and iterated stochastic integrals live in
`probjax.utils.sdeutil.brownian`; their layout and Itô/Stratonovich conventions
must agree with the solver.

## Inverse log-determinants

`inverse_and_logabsdet` can invert a terminal ODE map by integrating backward.
The inverse log-determinant uses the integrated drift divergence. This is a
numerical continuous-flow estimate, not an exact Jacobian determinant of the
discrete stepping map. Inversion is useful only when the flow and its backward
integration are well behaved.

`trace_estimator="exact"` computes full Jacobians. `"hutchinson"` uses JVPs
with fixed random probes along the trajectory, trading computation for variance.
Pass `logdet_rng` and increase `num_samples` to reduce variance. These options
affect inverse log-determinant computation, not an ordinary forward solve.

```python
from probjax.core import inverse_and_logabsdet

def flow(x):
    return odeint(
        decay, x, ts, 0.5, collect_trace=False,
        trace_estimator="hutchinson", logdet_rng=jax.random.key(1),
        num_samples=2,
    )

recovered, inverse_logdet = inverse_and_logabsdet(flow)(flow(y0))
assert jnp.allclose(recovered, y0, rtol=1e-3)
assert jnp.allclose(inverse_logdet, 1.0, atol=1e-3)
```

A callable `trace_estimator(drift_flat, t, x_flat, args)` can exploit an analytic
or structured divergence. There is no public Jacobian-evaluation stride option;
skipping evaluations is not a supported accuracy/performance control.
