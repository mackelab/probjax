from functools import partial
from typing import Tuple

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
    return is_matrix(A) and jnp.all(
        jnp.diag(jnp.diagonal(A, axis1=axis1, axis2=axis2)) == A, axis=(axis1, axis2)
    )


def is_triangular_matrix(A: Array, lower: bool = True) -> bool:
    """Check if input is a triangular matrix

    Args:
        A (Array): Input array
        lower (bool, optional): True if lower triangular. Defaults to True.

    Returns:
        bool: True if A is a triangular matrix, or a batch of triangular matrices
    """
    return is_matrix(A) and jnp.all(
        jnp.tril(A) == A if lower else jnp.triu(A), axis=(-2, -1)
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
