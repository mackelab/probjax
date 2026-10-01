# Utilities

Numerical solvers, linear-algebra helpers and shared type aliases.

## Solvers

See [ODE and SDE integration](../guides/numerical-solvers.md) for adaptive
controllers, gradients and inverse log-determinants.

::: probjax.utils.odeint
::: probjax.utils.sdeint
::: probjax.utils.root
::: probjax.utils.newton_raphson

## Special functions

`betainccinv(a, b, q)` and `gammainccinv(a, q)` invert the complementary
regularized incomplete beta and gamma functions. They accept the upper-tail
probability directly, avoiding the loss of tiny probabilities when forming
`1 - q`. Beta and gamma distribution `isf` methods use them directly.

```python
from probjax.utils import gammainccinv, log1mexp, logdiffexp

x = gammainccinv(1.0, 1e-30)  # Approximately 69.07755, including in float32.
log_survival = log1mexp(-1e-8)
log_difference = logdiffexp(1000.0, 999.0)
```

Inverse functions require finite positive shapes and clip probabilities to
[0, 1]. Invalid shapes return NaN. At upper-tail probabilities zero and one,
beta returns one and zero; gamma returns infinity and zero.

Beta inversion uses cached safeguarded Halley solvers, reflecting according
to the root's location. Its total iteration budget is
`2 + max_halley_steps + max_bisection_steps`. Reverse differentiation uses the
inverse density for probability gradients and central differences for shape
gradients; forward AD and higher shape derivatives are unsupported.
Gamma uses log-space initialization and an implicit JVP, avoiding differentiation
through solver iterations. Higher shape derivatives depend on JAX's
`igamma_grad_a` support.

FP32 accuracy remains limited by the underlying CDF for some extremely unequal
beta shapes: the seeded stress test includes a case near `a=0.051, b=784` with
about 2% relative quantile error. Increasing the iteration budget does not fix
that case. Use FP64 for demanding tail calculations. The originally identified
FP32 regression near `a=0.070, b=626` is covered separately at 0.5% tolerance.

`log1mexp(x)` computes `log(1-exp(x))` for `x <= 0`; `logdiffexp(a, b)` computes
`log(exp(a)-exp(b))` for `a >= b`. They avoid cancellation and intermediate
overflow, support broadcasting, and differentiate on the finite interior.
Zero differences return negative infinity; invalid domains return NaN.

::: probjax.utils.betaincinv
::: probjax.utils.betainccinv
::: probjax.utils.gammaincinv
::: probjax.utils.gammainccinv
::: probjax.utils.log1mexp
::: probjax.utils.logdiffexp
::: probjax.utils.digammainv

## Interpolation and linear algebra

::: probjax.utils.linear_interpolation
::: probjax.utils.polynomial_interpolation
::: probjax.utils.cholesky_update
::: probjax.utils.mv_diag_or_dense
