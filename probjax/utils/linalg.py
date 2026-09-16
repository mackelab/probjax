from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import expm
from jaxtyping import Array, Float


def cholesky_update(L, u):
    """
    Update the Cholesky decomposition of a matrix after a rank-1 update i.e.

    C = L @ L.T + multiplier * u @ u.T

    Args:
    L: A [D, D] lower triangular matrix, the Cholesky factor of the original matrix.
    u: A [D,] vector, the update vector.

    Returns:
    The updated [D, D] lower triangular matrix.
    """
    D = L.shape[0]
    indices = jnp.arange(D)

    def body_fun(i, vals):
        L, u = vals
        r = jnp.sqrt(L[i, i] ** 2 + u[i] ** 2)
        c = r / L[i, i]
        s = u[i] / L[i, i]
        L = L.at[i, i].set(r)

        mask = indices > i
        col_update = (L[:, i] + s * u) / c
        col_update = jnp.where(mask, col_update, L[:, i])
        L = L.at[:, i].set(col_update)
        u_update = c * u - s * L[:, i]
        u = jnp.where(mask, u_update, u)

        return (L, u)

    L, u = jax.lax.fori_loop(0, D, body_fun, (L, u))

    return L


@partial(jax.jit, static_argnames=("precission",), inline=True)
def mv_diag_or_dense(
    A_diag_or_dense: Array, b: Array, precission=jax.lax.Precision.DEFAULT
) -> Array:
    """Dot product of a diagonal matrix and a dense matrix

    Args:
        A (Array): Diagonal matrix
        B (Array): Dense matrix

    Returns:
        Array: Dot product
    """
    A_diag_or_dense = jnp.asarray(A_diag_or_dense)
    dtype = jnp.result_type(A_diag_or_dense.dtype, b.dtype)
    A_diag_or_dense = A_diag_or_dense.astype(dtype)
    b = b.astype(dtype)
    ndim = A_diag_or_dense.ndim

    if ndim <= 1:
        return jax.lax.mul(A_diag_or_dense, b)
    else:
        return jax.lax.dot(
            A_diag_or_dense, b, precision=precission, preferred_element_type=dtype
        )


def is_matrix(A: Array) -> bool:
    """Check if input is a matrix

    Args:
        A (Array): Input array

    Returns:
        bool: True if A is a matrix, or a batch of matrices
    """
    return len(A.shape) >= 2


def is_diagonal_matrix(A: Array, axis1=-2, axis2=-1) -> bool:
    """Check if input is a diagonal matrix

    Args:
        A (Array): Input array

    Returns:
        bool: True if A is a diagonal matrix, or a batch of diagonal matrices
    """
    if not is_matrix(A):
        return False
    matrix = jnp.moveaxis(A, (axis1, axis2), (-2, -1))
    mask = jnp.eye(matrix.shape[-2], matrix.shape[-1], dtype=bool)
    return jnp.all(jnp.where(mask, 0, matrix) == 0, axis=(-2, -1))


def is_triangular_matrix(A: Array, lower: bool = True) -> bool:
    """Check if input is a triangular matrix

    Args:
        A (Array): Input array
        lower (bool, optional): True if lower triangular. Defaults to True.

    Returns:
        bool: True if A is a triangular matrix, or a batch of triangular matrices
    """
    return is_matrix(A) and jnp.all(
        jnp.tril(A) == A if lower else jnp.triu(A) == A, axis=(-2, -1)
    )


def batch_mv(bmat: Array, bvec: Array) -> Array:
    """
    Performs a batched matrix-vector product, with compatible but different batch
    shapes.

    This function takes as input `bmat`, containing n x n matrices, and
    `bvec`, containing length n vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return jnp.matmul(bmat, bvec[..., jnp.newaxis])[..., 0]


def batch_mahalanobis(bL: Array, bx: Array) -> Array:
    """
    Computes the squared Mahalanobis distance x^T M^-1 x for a factored M = LL^T.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the
    same batch shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    bL = jnp.broadcast_to(bL, bx.shape[:-1] + bL.shape[-2:])

    sol = lax.linalg.triangular_solve(bL, bx, lower=True, transpose_a=True)
    return jnp.sum(sol**2, axis=-1)


def transition_matrix(A: Array, t0: Float, t1: Float) -> Array:
    """Transition matrix

    Args:
        A (Array): Drift matrix
        t (float): New time point
        t0 (float): Old time point

    Returns:
        Array: Transition matrix
    """
    if A.shape[-1] == 1:
        return jnp.exp(A * (t1 - t0))
    else:
        return expm(A * (t1 - t0))


def matrix_fraction_decomposition(
    t0: Float, t1: Float, A: Array, B: Array
) -> Tuple[Array, Array]:
    """Matrix fraction decomposition

    Returns the transition matrix and covariance. Is exact if A and B are truely
    time independent

    Args:
        t0 (float): New time point
        t1 (float): Old time point
        A (Array): Drift matrix
        B (Array): Diffusion matrix

    Returns:
        Tuple[Array]: Transition matrix and covariance
    """
    d = A.shape[-1]
    blockmatrix = jnp.block([[A, jnp.dot(B, B.T)], [jnp.zeros((d, d)), -A.T]])
    M = expm(blockmatrix * (t1 - t0))
    Phi = M[:d, :d]
    Q = jnp.dot(M[:d, d:], Phi.T)
    return Phi, Q


def tsvd(A: Array, r: int) -> Tuple[Array, Array, Array]:
    """Truncated singular value decomposition

    Args:
        A (Array): Input matrix
        r (int): Rank

    Returns:
        Tuple[Array]: U, S, V
    """
    U, S, V = jnp.linalg.svd(A)
    U = U[:, :r]
    S = S[:r]
    V = V[:r, :]
    return U, S, V


def symmetrize_matrix(A: Array) -> Array:
    """Return the symmetrized version of a matrix

    Args:
        A (Array): Input matrix

    Returns:
        Array: Symmetrized matrix
    """
    return 0.5 * (A + A.T)


def K_step_lyapunov(
    K: Array, drift_matrix: Array, V_n: Array, diffusion_matrix: Array, t: Float
) -> Array:
    """One step of the K-update in the Lyapunov equation integrator.

    This corresponds to the step:
    K_dot = F*K + K*(V_n' * F' * V_n) + C*V_n

    Args:
        K (Array): Current K matrix
        drift_matrix (Array): Drift matrix
        V_n (Array): Current U matrix
        diffusion_matrix (Array): Diffusion matrix
        t (float): Time point

    Returns:
        Array: K_dot
    """
    return (
        drift_matrix @ K + K @ (V_n.T @ drift_matrix.T @ V_n) + diffusion_matrix @ V_n
    )


def S_step_lyapunov(
    S: Array, drift_matrix: Array, U_hat: Array, diffusion_matrix: Array, t: Float
) -> Array:
    """One step of the S-update in the Lyapunov equation integrator.

    This corresponds to:
    S_dot = S*U_hat'*F'*U_hat + (S*U_hat'*F'*U_hat)' + U_hat'*C*U_hat

    Args:
        S (Array): current S matrix
        drift_matrix (Array): Drift matrix
        U_hat (Array): Current U matrix
        diffusion_matrix (Array): Diffusion matrix
        t (Float): Time parameter

    Returns:
        Array: S_dot
    """
    S_Ut_Ft_U = S @ U_hat.T @ drift_matrix.T @ U_hat
    return S_Ut_Ft_U.T + S_Ut_Ft_U + U_hat.T @ diffusion_matrix @ U_hat


def closed_form_K_step_lyapunov(
    drift_matrix: Array, V: Array, diffusion_matrix: Array, K_n: Array, h: Float
) -> Array:
    """Closed form solution for the K-step in the Lyapunov equation integrator.

    This corresponds to the step:
    Based on:
        K_next = (Mexp_11 * K_n + Mexp_12) / Mexp_22

    where Mexp = exp(h * [F       C*V
                        0   -V'*F'*V])

    Args:
        drift_matrix (Array): Drift matrix
        V (Array): Current U matrix
        diffusion_matrix (Array): Diffusion matrix
        K_n (Array): Current K matrix
        h (Float): Time step

    Returns:
        Array: K_next
    """

    d_select = drift_matrix.shape[0]
    # Construct block matrix for exp
    block_top_right = diffusion_matrix @ V
    block_bottom_left = jnp.zeros((
        V.T.shape[0],
        V.T.shape[1],
    ))  # zero(V') has shape (V.shape[1], V.shape[0])
    block_bottom_right = -V.T @ drift_matrix.T @ V
    M = jnp.block([
        [drift_matrix, block_top_right],
        [block_bottom_left, block_bottom_right],
    ])
    Mexp = expm(h * M)
    Mexp_11 = Mexp[:d_select, :d_select]
    Mexp_12 = Mexp[:d_select, d_select:]
    Mexp_22 = Mexp[d_select:, d_select:]
    # Knext = (Mexp_11 * Kₙ + Mexp_12) / Mexp_22
    # Solve for Knext: (Mexp_11 @ K_n + Mexp_12) * inv(Mexp_22)
    # Mexp_22 is square, so we can solve the linear system:
    Knext = jnp.linalg.solve(Mexp_22.T, (Mexp_11 @ K_n + Mexp_12).T).T
    return Knext


def closed_form_S_step_lyapunov(drift_matrix, U_hat, diffusion_matrix, S_hat_n, h):
    """Closed form solution for the S-step in the Lyapunov equation integrator.

    This corresponds to the step:
    Based on:
        Snext = (Mexp_11 * S_hat_n + Mexp_12) * Mexp_11'

    where Mexp = exp(h * [F       C*V
                        0   -V'*F'*V])

    Args:
        drift_matrix (Array): Drift matrix
        U_hat (Array): Current U matrix
        diffusion_matrix (Array): Diffusion matrix
        S_hat_n (Array): Current S matrix
        h (Float): Time step

    Returns:
        Array: S_next
    """

    # hAₛ = h * (Û' * (drift_matrix * Û))
    hA_s = h * (U_hat.T @ (drift_matrix @ U_hat))
    hP_s = h * (U_hat.T @ (diffusion_matrix @ U_hat))

    d_select = hA_s.shape[0]
    # Construct block matrix for exp:
    # Mexp = exp([hA_s     hP_s;
    #             zero(hA_s)  -hA_s'])
    # zero(hA_s) is a zero matrix with same shape as hA_s
    zero_hA_s = jnp.zeros_like(hA_s)
    block_matrix = jnp.block([[hA_s, hP_s], [zero_hA_s, -hA_s.T]])
    Mexp = expm(block_matrix)
    Mexp_11 = Mexp[:d_select, :d_select]
    Mexp_12 = Mexp[:d_select, d_select:]
    # Snext = (Mexp_11 * Ŝₙ + Mexp_12) * Mexp_11'
    # In Python: Snext = (Mexp_11 @ S_hat_n + Mexp_12) @ Mexp_11.T
    Snext = (Mexp_11 @ S_hat_n + Mexp_12) @ Mexp_11.T
    return Snext


# def solve_lyapunov_dlr_with_factor(proc, Y, r0, t_span, num_steps):
#     # This corresponds to the first solve_lyapunov_dlr definition
#     # In Julia:
#     # QR_L = qr(Y.factor)
#     # Q_L = Matrix(QR_L.Q)
#     # R_L = Matrix(QR_L.R)
#     # Y_tpl = (Q_L, symmetrize_matrix(R_L * R_L'))
#     # solve_lyapunov_dlr(proc, Y_tpl, r0, t_span, num_steps)
#     Q_L, R_L = np.linalg.qr(Y.factor)
#     Y_tpl = (Q_L, symmetrize_matrix(R_L @ R_L.T))
#     return solve_lyapunov_dlr(proc, Y_tpl, r0, t_span, num_steps)


def solve_lyapunov_dlr(proc, Y, r0, dt, num_steps):
    # If num_steps == 1, splitting_integrator_steps = [dt], else use linspace and skip
    if num_steps == 1:
        splitting_integrator_steps = [dt]
        h = dt
    else:
        _rg = jnp.linspace(0.0, dt, num_steps + 1)
        _st = _rg[1] - _rg[0]
        splitting_integrator_steps = _rg[
            1:
        ]  # equivalent to _rg[2:end] in Julia 1-based indexing
        h = _st

    # Assert dimensions
    U_init, S_init = Y
    assert U_init.shape[1] == S_init.shape[0] == S_init.shape[1] == r0

    # small_matexp = exp(proc.drift_matrix_1d * h)
    # If drift_matrix_1d is scalar:
    small_matexp = jnp.exp(proc.drift_matrix_1d * h)

    # eAh = kronecker(I(wiener_process_dimension(proc)), small_matexp)
    w_dim = proc.wiener_process_dimension()
    eAh = jnp.kron(jnp.eye(w_dim), small_matexp)

    # phi_factor = kronecker(I(w_dim), (small_matexp - I) / (proc.drift_matrix_1d * h))
    phi_factor = jnp.kron(
        jnp.eye(w_dim), (small_matexp - 1.0) / (proc.drift_matrix_1d * h)
    )

    current_Y = (U_init, S_init)

    for _ in splitting_integrator_steps:
        U_n, S_n = current_Y
        V_n = U_n

        # K STEP
        K_n = U_n @ S_n

        if jnp.allclose(S_n, 0):
            # If S_n is zero
            K_next = h * phi_factor @ (proc.diffusion_matrix @ V_n)
        else:
            K_next = eAh @ K_n + h * phi_factor @ (
                K_n @ (V_n.T @ (proc.drift_matrix.T @ V_n))
                + proc.diffusion_matrix @ V_n
            )

        # QR decomposition
        Q, R = jnp.linalg.qr(K_next)
        U_hat = Q
        M_hat = U_hat.T @ U_n
        N_hat = M_hat
        S_hat_n = M_hat @ S_n @ N_hat.T

        S_next = closed_form_S_step_lyapunov(
            proc.drift_matrix,
            U_hat,
            proc.diffusion_matrix,
            S_hat_n,
            h,
        )

        current_Y = (U_hat, symmetrize_matrix(S_next))

    chol_S = jnp.linalg.cholesky(current_Y[1])
    return current_Y[0], chol_S


@jax.custom_jvp
def _log_quadratic_form(matrix):
    """e_0.T log(matrix) e_0 with a repeated-eigenvalue-safe first derivative."""
    values, vectors = jnp.linalg.eigh(matrix)
    return jnp.sum(vectors[0]**2 * jnp.log(values))


@_log_quadratic_form.defjvp
def _log_quadratic_form_jvp(primals, tangents):
    (matrix,), (tangent,) = primals, tangents
    values, vectors = jnp.linalg.eigh(matrix)
    left, right = values[:, None], values[None, :]
    difference = left-right
    equal = difference == 0
    safe_difference = jnp.where(equal, 1, difference)
    relative = difference/right
    # Use log1p near equal eigenvalues to avoid cancellation.
    close = jnp.abs(relative) < .5
    numerator = jnp.where(close, jnp.log1p(jnp.where(close, relative, 0)),
                          jnp.log(left)-jnp.log(right))
    divided_difference = jnp.where(equal, 1/right, numerator/safe_difference)
    weights = vectors[0]
    derivative = jnp.sum(weights[:, None]*weights[None, :]*divided_difference
                         * (vectors.T @ tangent @ vectors))
    return _log_quadratic_form(matrix), derivative


def lanczos_logdet(S, num_steps=20, key=None, *, num_probes=16):
    """Estimate an SPD log determinant using multi-probe Lanczos quadrature.

    ``num_steps`` controls quadrature depth, not trace-estimation accuracy.
    ``num_probes`` controls independent Rademacher probes; their average is a
    stochastic trace estimate. With at least ``dim`` probes an orthogonal basis
    is used instead, eliminating trace variance. A full basis and full depth
    recover the exact log determinant up to floating-point error. Omitting
    ``key`` uses seed zero for reproducibility, never a fixed all-ones probe.
    """
    if num_steps < 1 or num_probes < 1:
        raise ValueError("num_steps and num_probes must be positive")
    if hasattr(S, "operator"):
        matvec, dim = S.operator, S.out_dim
    elif callable(S):
        if not hasattr(S, "shape"):
            raise ValueError("Callable S must have a shape attribute")
        matvec, dim = S, S.shape[0]
    else:
        S = jnp.asarray(S)
        # The estimator is defined on symmetric positive-definite matrices.
        # Use the symmetric extension so dense entrywise gradients agree with
        # slogdet, rather than depending on off-domain asymmetric perturbations.
        S = (S + S.T) / 2
        matvec, dim = lambda v: S @ v, S.shape[0]
    if dim < 1:
        raise ValueError("S must have positive dimension")
    steps = min(num_steps, dim)
    dtype = jnp.asarray(matvec(jnp.ones(dim))).dtype
    if num_probes >= dim:
        probes = jnp.eye(dim, dtype=dtype)
    else:
        key = jax.random.key(0) if key is None else key
        probes = jax.random.rademacher(key, (num_probes, dim), dtype=dtype) / jnp.sqrt(dim)

    def quadrature(v0):
        # Reorthogonalize to avoid spurious repeated Ritz values after
        # Krylov convergence. Zero padding is decoupled from the first basis vector.
        basis = jnp.zeros((steps, dim), dtype=dtype)
        def body(carry, i):
            v, previous, beta_previous, basis, active = carry
            basis = basis.at[i].set(v)
            w = matvec(v) - beta_previous * previous
            alpha = jnp.dot(v, w)
            w = w - alpha * v
            w = w - basis.T @ (basis @ w)
            squared_norm = jnp.dot(w, w)
            beta = jnp.where(squared_norm > 0, jnp.sqrt(jnp.where(squared_norm > 0, squared_norm, 1)), 0)
            threshold = jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(alpha), jnp.finfo(dtype).tiny) * dim
            next_active = active & (beta > threshold)
            next_v = jnp.where(next_active, w / jnp.where(next_active, beta, 1), 0)
            return (next_v, v, jnp.where(next_active, beta, 0), basis, next_active), (
                jnp.where(active, alpha, 1), jnp.where(next_active, beta, 0))
        _, (diag, beta) = jax.lax.scan(
            body, (v0, jnp.zeros_like(v0), jnp.asarray(0, dtype), basis, True),
            jnp.arange(steps))
        T = jnp.diag(diag) + jnp.diag(beta[:-1], 1) + jnp.diag(beta[:-1], -1)
        return dim * _log_quadratic_form(T)
    return jnp.mean(jax.vmap(quadrature)(probes))


# ---------------------------------------------------------------------------
# Batched Preconditioned Conjugate Gradient (PCG)
# ---------------------------------------------------------------------------


class PCGInfo(NamedTuple):
    converged: Array  # (nrhs,) bool
    rel_residual: Array  # (nrhs,) float
    iters_per_block: Array  # (nblocks,) int


def _identity_precond(x):
    return x


def _matmat_from_matvec(matvec, x):
    """Apply matvec column-wise: x shape (n, bs) -> (n, bs)."""
    return jax.vmap(matvec, in_axes=1, out_axes=1)(x)


def _solve_rhs_block_pcg(
    matvec: Callable,
    B: Array,  # (n, bs)
    *,
    M: Callable = _identity_precond,
    x0: Optional[Array] = None,
    tol: float = 1e-4,
    maxiter: int = 200,
):
    """Batched PCG for a block of RHS columns.

    Solves A @ X = B where A is accessed only via matvec. All bs columns
    run independent CG recurrences in parallel. Converged columns are
    frozen so they don't pollute the shared matvec.

    Args:
        matvec: v -> A @ v, shape (n,) -> (n,).
        B: RHS block, shape (n, bs).
        M: Preconditioner, maps (n, bs) -> (n, bs). Default: identity.
        x0: Initial guess, shape (n, bs). Default: zeros.
        tol: Relative residual tolerance per column.
        maxiter: Maximum CG iterations.

    Returns:
        (X, iters, converged, rel_residual) where X is (n, bs).
    """
    # Solve in units of each RHS magnitude so squared norms cannot underflow
    # or overflow merely because the right-hand side has a different scale.
    rhs_scale = jnp.max(jnp.abs(B), axis=0)
    if x0 is not None:
        rhs_scale = jnp.where(rhs_scale > 0, rhs_scale, jnp.max(jnp.abs(x0), axis=0))
    rhs_scale = jnp.where(rhs_scale > 0, rhs_scale, 1)
    B = B / rhs_scale[None, :]
    X = jnp.zeros_like(B) if x0 is None else x0 / rhs_scale[None, :]
    R = B - _matmat_from_matvec(matvec, X)
    Z = M(R)

    b2 = jnp.sum(B * B, axis=0)
    b2_safe = jnp.where(b2 > 0, b2, 1.0)
    # A zero RHS retains the original absolute-residual tolerance.
    tol2 = jnp.where(b2 > 0, (tol**2) * b2_safe, (tol / rhs_scale)**2)

    # Zero-RHS columns are already converged
    active = jnp.sum(R * R, axis=0) > tol2
    R = jnp.where(active[None, :], R, 0.0)
    Z = jnp.where(active[None, :], Z, 0.0)
    P = Z
    rho = jnp.sum(R * Z, axis=0)


    def cond_fn(state):
        k, _X, _R, _Z, _P, _rho, active = state
        return (k < maxiter) & jnp.any(active)

    def body_fn(state):
        k, X, R, Z, P, rho, active = state

        Q = _matmat_from_matvec(matvec, P)
        denom = jnp.sum(P * Q, axis=0)
        denom = jnp.where(denom > 0, denom, 1.0)

        alpha = jnp.where(active, rho / denom, 0.0)
        X = X + P * alpha[None, :]
        R = R - Q * alpha[None, :]

        r2 = jnp.sum(R * R, axis=0)
        active_new = r2 > tol2

        # Freeze converged columns
        R = jnp.where(active_new[None, :], R, 0.0)
        Z = M(R)
        Z = jnp.where(active_new[None, :], Z, 0.0)

        rho_new = jnp.sum(R * Z, axis=0)
        rho_safe = jnp.where(rho > 0, rho, 1.0)
        beta = jnp.where(active_new, rho_new / rho_safe, 0.0)

        P = Z + P * beta[None, :]
        P = jnp.where(active_new[None, :], P, 0.0)
        return (k + 1, X, R, Z, P, rho_new, active_new)

    init_state = (jnp.array(0, dtype=jnp.int32), X, R, Z, P, rho, active)
    k, X, R, Z, P, rho, active = lax.while_loop(cond_fn, body_fn, init_state)

    # Recompute true final residual for accurate diagnostics
    R_true = B - _matmat_from_matvec(matvec, X)
    residual_scale = jnp.max(jnp.abs(R_true), axis=0)
    normalized_residual = R_true / jnp.where(residual_scale > 0, residual_scale, 1)[None, :]
    residual_norm = residual_scale * jnp.sqrt(jnp.sum(normalized_residual**2, axis=0))
    rel_residual = jnp.where(b2 > 0, residual_norm / jnp.sqrt(b2_safe), residual_norm * rhs_scale)
    converged = rel_residual <= tol

    return X * rhs_scale[None, :], k, converged, rel_residual


def batched_pcg_solve(
    matvec: Callable,
    B: Array,  # (n, nrhs)
    *,
    M: Optional[Callable] = None,
    x0: Optional[Array] = None,
    block_size: int = 128,
    tol: float = 1e-4,
    maxiter: int = 200,
):
    """Solve A @ X = B for many RHS using chunked matrix-free PCG.

    Processes columns of B in blocks of `block_size`, running batched CG
    within each block. Uses `lax.map` over blocks so the total trace size
    is O(block_size) rather than O(nrhs).

    Args:
        matvec: function v -> A @ v, with v shape (n,).
        B: RHS matrix, shape (n, nrhs).
        M: Optional preconditioner mapping (n, bs) -> (n, bs).
            Default: identity.
        x0: Optional initial guess, shape (n, nrhs).
        block_size: Number of RHS columns per CG block.
        tol: Relative residual tolerance per column.
        maxiter: Max CG iterations per block.

    Returns:
        (X, info) where X has shape (n, nrhs) and info is a PCGInfo.
    """
    if M is None:
        M = _identity_precond

    n, nrhs = B.shape
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    pad = (-nrhs) % block_size

    def pad_cols(arr):
        if arr is None or pad == 0:
            return arr
        return jnp.pad(arr, ((0, 0), (0, pad)))

    B_pad = pad_cols(B)
    x0_pad = pad_cols(x0)
    nblocks = (nrhs + pad) // block_size
    # (nblocks, n, block_size)
    B_blocks = B_pad.reshape(n, nblocks, block_size).transpose(1, 0, 2)

    if x0_pad is None:

        def solve_block(Bi):
            return _solve_rhs_block_pcg(
                matvec, Bi, M=M, x0=None, tol=tol, maxiter=maxiter
            )

        X_blocks, iters, conv_blocks, rel_blocks = lax.map(solve_block, B_blocks)
    else:
        X0_blocks = x0_pad.reshape(n, nblocks, block_size).transpose(1, 0, 2)

        def solve_block_with_x0(args):
            Bi, X0i = args
            return _solve_rhs_block_pcg(
                matvec, Bi, M=M, x0=X0i, tol=tol, maxiter=maxiter
            )

        X_blocks, iters, conv_blocks, rel_blocks = lax.map(
            solve_block_with_x0, (B_blocks, X0_blocks)
        )

    X_pad = X_blocks.transpose(1, 0, 2).reshape(n, nrhs + pad)
    X = X_pad[:, :nrhs]

    converged = conv_blocks.reshape(nrhs + pad)[:nrhs]
    rel_residual = rel_blocks.reshape(nrhs + pad)[:nrhs]

    info = PCGInfo(
        converged=converged,
        rel_residual=rel_residual,
        iters_per_block=iters,
    )
    return X, info
