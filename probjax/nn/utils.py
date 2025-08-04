from functools import partial
from typing import Callable, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax import lax
from jax.ops import segment_max  # segment reduction (available in JAX)
from jaxtyping import Array
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from probjax.core.custom_primitives.custom_inverse import custom_inverse


def extract_permutation(M: jnp.ndarray) -> jnp.ndarray:
    """
    Given a square matrix M (e.g. a Sinkhorn coupling),
    returns a permutation (as a vector of column indices) for the rows.

    For each row we precompute a sorted list (in descending order) of candidate columns.
    Then we assign each row its best candidate. In case multiple rows choose the same column,
    only the row with the highest score (with a small tie–breaker) keeps it; the others
    move on to their next candidate.
    """
    n = M.shape[0]
    # For each row, sort the candidate column indices in descending order of M.
    sorted_indices = jnp.argsort(-M, axis=1)  # shape (n, n)
    # For each row, candidate pointer (initially 0 = best candidate)
    candidate_idx = jnp.zeros(n, dtype=jnp.int32)
    # Current assignment: for each row i, assignment[i] is its candidate column.
    assignment = sorted_indices[jnp.arange(n), candidate_idx]

    # We define the loop condition: while the assignment is not a permutation.
    def cond_fn(state):
        _, assignment = state
        # Check if assignment is a permutation by comparing sorted(assignments) to 0,1,...,n-1.
        return ~jnp.all(jnp.sort(assignment) == jnp.arange(n))

    def body_fn(state):
        candidate_idx, assignment = state
        row_indices = jnp.arange(n)
        # Compute a "score" for each row's candidate.
        # We subtract a very small multiple of the row index so that if two rows have
        # the same value for a column, the row with the smaller index wins.
        candidate_scores = (
            M[row_indices, assignment] - row_indices.astype(M.dtype) * 1e-6
        )
        # For each column, compute the maximum candidate score among rows that selected that column.
        # (Rows that did not select a given column do not contribute.)
        best_score_per_col = segment_max(candidate_scores, assignment, num_segments=n)
        # For each row, get the best candidate score for the column it chose.
        best_for_row = best_score_per_col[assignment]
        # Identify rows that lost the conflict (i.e. whose candidate score is not the maximum
        # for the chosen column).
        conflict = candidate_scores < best_for_row
        # For rows in conflict, advance to their next candidate.
        candidate_idx = candidate_idx + conflict.astype(jnp.int32)
        # (Clip candidate indices so they remain in [0, n-1].)
        candidate_idx = jnp.minimum(candidate_idx, n - 1)
        # Update the assignment accordingly.
        assignment = sorted_indices[jnp.arange(n), candidate_idx]
        return candidate_idx, assignment

    candidate_idx, assignment = lax.while_loop(
        cond_fn, body_fn, (candidate_idx, assignment)
    )
    return assignment


def ot_copula(
    x,
    y,
    p: float = 2,
    epsilon: float = 1e-1,
    threshold: float = 0.1,
    inner_iterations: int = 4,
    min_iterations: int = 0,
    max_iterations: int = 100,
):
    # (These objects are assumed to be defined/imported elsewhere.)
    geom = pointcloud.PointCloud(x, y, cost_fn=costs.PNormP(p), epsilon=epsilon)
    ot_prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn(
        threshold=threshold,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        inner_iterations=inner_iterations,
    )
    ot = solver(ot_prob)
    # Instead of simply taking argmax along axis=1 (which can assign the same y multiple times),
    # we extract a valid permutation.
    permutation = extract_permutation(ot.matrix)
    y_permuted = y[permutation]
    return x, y_permuted


class Sequential(nnx.Module):
    def __init__(self, *layers):
        """Sequential module.

        Args:
            layers (nnx.Module): List of layers.
        """
        self.layers = layers

    def __call__(self, x, *args, **kwargs) -> Array:
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x


class Affine(nnx.Module):
    def __init__(self, in_out_dim: int, rngs):
        """This module applies an affine transformation to the input.

        Args:
            in_out_dim (int): Input and output dimension.
            rngs (rngs): Random generator stream.
        """
        self.scale = nnx.Variable(
            nnx.initializers.normal(1.0)(rngs.next(), shape=(in_out_dim,))
        )
        self.bias = nnx.Variable(
            nnx.initializers.normal(1.0)(rngs.next(), shape=(in_out_dim,))
        )

    def __call__(self, x: Array, *args) -> Array:
        return x * self.scale.value + self.bias.value


class Flip(nnx.Module):
    def __init__(self, axis: int = -1, rngs=None):
        """Flip the array along an axis.

        Args:
            axis (int, optional): Axis to flip. Defaults to -1.
        """
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.flip(x, axis=self.axis)


class Permute(nnx.Module):
    def __init__(self, permutation: Array, axis: int = -1, rngs=None):
        """Permutes the array along an axis.

        Args:
            permutation (Array): An array of indices to permute.
            axis (int, optional): Axis to permute. Defaults to -1.
        """
        self.permutation = nnx.Variable(permutation)
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.take(x, self.permutation, axis=self.axis)


class Rotate(nnx.Module):
    def __init__(
        self,
        in_out_dim: int,
        rngs,
        *,
        rotation_matrix: Optional[Array] = None,
        learnable: bool = False,
    ):
        """Rotate the array.

        Args:
            rotation_matrix (Array): Rotation matrix.
            name (str, optional): Name of the module. Defaults to "rotate".
        """
        self.in_out_dim = in_out_dim
        self.learnable = learnable
        if not learnable:
            if rotation_matrix is None:
                self.rotation_matrix = nnx.Variable(
                    nnx.initializers.orthogonal()(
                        rngs.next(), shape=(in_out_dim, in_out_dim)
                    )
                )

            else:
                self.rotation_matrix = nnx.Variable(rotation_matrix)
        else:
            raise NotImplementedError(
                "Learnable rotation matrix is not implemented yet."
            )
            # TODO: Matrix exponetial of any skew symetric matrix is orthogonal
            # Use for reparameterization

    def __call__(self, x: Array, *args) -> Array:
        if not self.learnable:
            rotation_matrix = jax.lax.stop_gradient(self.rotation_matrix.value)
        return rotate(rotation_matrix, x)


@partial(custom_inverse, inv_argnum=1)
def rotate(R, x):
    return jnp.matmul(R, x.T).T


rotate.definv_and_logdet(lambda R, x: (jnp.matmul(R.T, x.T).T, 0.0))


class GaussianFourierEmbedding(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rngs,
        *,
        learnable=True,
    ):
        """Gaussian Fourier embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
            name (str, optional): Name of the module. Defaults to
            "gaussian_fourier_embedding".
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learnable = learnable
        half_dim = self.output_dim // 2 + 1
        if not learnable:
            self.B = nnx.Variable(
                nnx.initializers.normal(1.0)(rngs.next(), shape=(half_dim, input_dim))
            )
        else:
            self.B = nnx.Param(
                nnx.initializers.normal(1.0)(rngs.next(), shape=(half_dim, input_dim))
            )

    def __call__(self, inputs):
        B = self.B.value
        if not self.learnable:
            B = jax.lax.stop_gradient(B)
        term1 = jnp.cos(2 * jnp.pi * jnp.dot(inputs, B.T))
        term2 = jnp.sin(2 * jnp.pi * jnp.dot(inputs, B.T))
        out = jnp.concatenate([term1, term2], axis=-1)
        return out[..., : self.output_dim]


class OneHot(nnx.Module):
    """One hot encoding module."""

    def __init__(self, num_tokens: int, rngs=None):
        """Represents a one hot encoding module.

        Args:
            num_tokens (int): Number of distinct tokens.
        """
        self.num_tokens = num_tokens

    def __call__(self, x: Array, *args) -> Array:
        """One hot encodes the input.

        Args:
            x (jax.Array): Input array of shape [B, T]
        """
        return jax.nn.one_hot(x, self.num_tokens)


class AdditiveFuse(nnx.Module):
    def __init__(self, input_dim: int, context_dim: int, rngs):
        """This module applies an additive transformation to the input.

        Args:
            in_out_dim (int): Input and output dimension.
            rngs (rngs): Random generator stream.
        """
        self.linear = nnx.Linear(context_dim, input_dim, rngs=rngs)

    def __call__(self, x: Array, context: Array) -> Array:
        return x + self.linear(context)


class AffineFuse(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        rngs,
        scale_activation: Callable | None = None,
        use_bias: bool = False,
    ):
        """This module applies an affine transformation to the input.

        Args:
            in_out_dim (int): Input and output dimension.
            rngs (rngs): Random generator stream.
        """
        self.linear_scale = nnx.Linear(
            context_dim,
            input_dim,
            rngs=rngs,
            use_bias=use_bias,
            kernel_init=nnx.initializers.zeros,
        )
        self.linear_bias = nnx.Linear(
            context_dim,
            input_dim,
            rngs=rngs,
            use_bias=use_bias,
            kernel_init=nnx.initializers.zeros,
        )
        self.scale_activation = scale_activation

    def __call__(self, x: Array, context: Array) -> Array:
        scale = 1 + self.linear_scale(context)
        if self.scale_activation is not None:
            scale = self.scale_activation(scale)
        bias = self.linear_bias(context)
        return x * scale + bias


class ConcatFuse(nnx.Module):
    def __init__(self, input_dim: int, context_dim: int, rngs):
        """This module applies an additive transformation to the input.

        Args:
            in_out_dim (int): Input and output dimension.
            rngs (rngs): Random generator stream.
        """
        self.linear = nnx.Linear(context_dim, input_dim, rngs=rngs)

    def __call__(self, x: Array, context: Array) -> Array:
        return jnp.concatenate([x, self.linear(context)], axis=-1)
