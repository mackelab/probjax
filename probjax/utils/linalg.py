import jax
import jax.numpy as jnp
from jax import lax

from jaxtyping import Array, Float
from typing import Tuple

from jax.scipy.linalg import expm


def batch_mv(bmat: Array, bvec: Array) -> Array:
    """
    Performs a batched matrix-vector product, with compatible but different batch shapes.

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

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    sol = lax.linalg.triangular_solve(bL, bx)
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
        return jnp.exp(A*(t1-t0))
    else:
        return expm(A*(t1-t0))


def matrix_fraction_decomposition(
    t: Float, t0: Float, A: Array, B: Array
) -> Tuple[Array, Array]:
    """Matrix fraction decomposition

    Returns the transition matrix and covariance. Is exact A and B are truely time independent

    Args:
        t (float): New time point
        t0 (float): Old time point
        A (Array): Drift matrix
        B (Array): Diffusion matrix

    Returns:
        Tuple[Array]: Transition matrix and covariance
    """
    d = A.shape[-1]
    blockmatrix = jnp.block([[jnp.zeros((d, d)), A], [-A.T, jnp.dot(B, B.T)]])
    M = expm(blockmatrix * (t - t0))
    Phi = M[:d, :d]
    Q = jnp.dot(M[:d, d:], Phi.T)
    return Phi, Q
