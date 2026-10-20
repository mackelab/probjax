# Utilities

Numerical solvers, linear-algebra helpers and shared type aliases.

## Kernel API

The base class every algorithm API derives from, and the value that binds an
API's options before the model is known.

::: probjax.utils.KernelAPI
::: probjax.utils.Configured

## Solvers

See [ODE and SDE integration](../guides/numerical-solvers.md) for adaptive
controllers, gradients and inverse log-determinants.

::: probjax.utils.odeint
::: probjax.utils.sdeint
::: probjax.utils.root
::: probjax.utils.newton_raphson

## Special functions

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

## Linear operators

::: probjax.utils.linear_operator.Diagonal
::: probjax.utils.linear_operator.Dense
::: probjax.utils.linear_operator.Function
::: probjax.utils.linear_operator.Jacobian
::: probjax.utils.linear_operator.solve
::: probjax.utils.linear_operator.logdet
::: probjax.utils.linear_operator.as_operator
::: probjax.utils.linear_operator.phi_matrices

## ODE integration internals

::: probjax.utils.odeutil.ODEInfo
::: probjax.utils.odeutil.ODESolver
::: probjax.utils.odeutil.ODEState
::: probjax.utils.odeutil.get_method
::: probjax.utils.odeutil.get_methods
::: probjax.utils.odeutil.StepSizeAdaptor
::: probjax.utils.odeutil.euler
::: probjax.utils.odeutil.implicit_euler
::: probjax.utils.odeutil.TraceFilter
::: probjax.utils.odeutil.TraceEverything
::: probjax.utils.odeutil.TraceNothing

## SDE integration internals

::: probjax.utils.sdeutil.SDEStepSizeAdaptor
::: probjax.utils.sdeutil.StrongStepSizeAdaptor
::: probjax.utils.sdeutil.WeakStepSizeAdaptor
::: probjax.utils.sdeutil.SDEInfo
::: probjax.utils.sdeutil.SDESolver
::: probjax.utils.sdeutil.SDEState
::: probjax.utils.sdeutil.get_method
::: probjax.utils.sdeutil.get_methods
::: probjax.utils.sdeutil.euler_maruyama
::: probjax.utils.sdeutil.exp_euler_maruyama
::: probjax.utils.sdeutil.milstein
::: probjax.utils.sdeutil.sri1
::: probjax.utils.sdeutil.sri2
::: probjax.utils.sdeutil.linear_exact_sde

## Special functions (subpackage)

::: probjax.utils.special.betaincinv
::: probjax.utils.special.betainccinv
::: probjax.utils.special.gammaincinv
::: probjax.utils.special.gammainccinv
::: probjax.utils.special.digammainv
::: probjax.utils.special.log1mexp
::: probjax.utils.special.logdiffexp
