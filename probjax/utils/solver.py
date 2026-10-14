from typing import Any

import jax
import jax.numpy as jnp

from probjax.utils.linear_operator import Dense, Function, Jacobian, solve


def newton_raphson(
    f,
    x0,
    tol=1e-6,
    max_iter=50,
    *,
    jacobian: str = "dense",
    linear_solver: Any = "auto",
):
    """Newton-Raphson root finding for ``f(x) = 0`` with ``f(x).shape == x.shape``.

    Gradients of the returned root use the implicit function theorem
    (``jax.lax.custom_root``) instead of differentiating through the
    iterations, so they cost one extra linear solve. The gradient is only
    valid when the solve converged: always check the returned flag before
    differentiating (callers that keep stepping, like the implicit ODE
    solvers, record it in their info).

    Args:
        f: Function whose root is sought.
        x0: Initial guess.
        tol: Stop once a Newton step is shorter than this (in 2-norm),
            floored at machine epsilon.
        max_iter: Maximum number of Newton steps (at least 1).
        jacobian: ``"dense"`` materializes the Jacobian with forward mode,
            which is best for small states. ``"matrix_free"`` applies it with
            one JVP per product, for large states with an iterative solver.
        linear_solver: Solver for each Newton step, passed to
            :func:`probjax.utils.linear_operator.solve`: ``"auto"`` (LU for
            dense, GMRES for matrix-free Jacobians), another solver name or a
            lineax solver. Inner solves use :func:`solve` defaults. Newton
            Jacobians are generally nonsymmetric, so ``"cg"``/``"cholesky"``
            are opt-in at the caller's risk.

    Returns:
        ``(x, converged)``: the root estimate and whether a step was shorter
        than ``tol``.
    """
    if jacobian not in ("dense", "matrix_free"):
        raise ValueError(f"jacobian must be 'dense' or 'matrix_free', got {jacobian!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}.")
    x0 = jnp.asarray(x0)
    tol = max(tol, jnp.finfo(x0.dtype).eps)
    shape = x0.shape

    def flat_f(x):
        return f(x.reshape(shape)).ravel()

    def jacobian_at(g, x):
        return Dense.jacfwd(g, x) if jacobian == "dense" else Jacobian(g, x)

    def newton_solve(g, x):
        def scan_fn(carry, _):
            converged, x = carry

            def newton_step(x):
                delta = solve(jacobian_at(g, x), -g(x), linear_solver)
                return x + delta, jnp.linalg.norm(delta)

            x, step_size = jax.lax.cond(converged, lambda x: (x, 0.0), newton_step, x)
            return (converged | (step_size < tol), x), None

        (converged, x), _ = jax.lax.scan(
            scan_fn, (jnp.asarray(False), x), None, length=max_iter
        )
        return x, converged

    def tangent_solve(g, y):
        # g is the Jacobian-vector product of f at the root.
        op = Function(g, y.shape, dtype=y.dtype)
        return solve(op.as_dense() if jacobian == "dense" else op, y, linear_solver)

    x, converged = jax.lax.custom_root(
        flat_f, x0.ravel(), newton_solve, tangent_solve, has_aux=True
    )
    return x.reshape(shape), converged


def root(
    fun,
    x0,
    args=(),
    method="newton-raphson",
    tol=1e-3,
    max_iter=20,
    **newton_kwargs,
):
    """Find a root of ``fun(x, *args)``.

    Args:
        fun (Callable): Function to find root of.
        x0 (Array): Initial value.
        args (tuple, optional): Extra arguments to pass to function. Defaults to ().
        method (str, optional): Method to use. Only ``"newton-raphson"``.
        tol (float, optional): Tolerance, floored at machine epsilon. Defaults to 1e-3.
        max_iter (int, optional): Maximum number of iterations. Defaults to 20.
        **newton_kwargs: ``jacobian``/``linear_solver`` for :func:`newton_raphson`.

    Returns:
        ``(x, converged)``.
    """

    # Dtype constraints on tolerance
    dtype = x0.dtype
    tol = max(tol, jnp.finfo(dtype).eps)

    _f = lambda x: fun(x, *args)

    if method == "newton-raphson":
        return newton_raphson(_f, x0, tol=tol, max_iter=max_iter, **newton_kwargs)
    else:
        raise NotImplementedError(f"Method {method} not implemented.")
