from functools import partial
from typing import Callable, Optional
import inspect

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


def filter_precision_kwargs(cls: type[nnx.Module], **kwargs):
    """Utility function to filter out unsupported precision kwargs.

    Note:
      - We unwrap functools.partial only for the BUGGED-class check.
      - We inspect the callable (class or partial) directly to get the effective parameters.
    """

    # Unwrap only for BUGGED membership check
    def unpack(cls):
        while isinstance(cls, partial):
            cls = cls.func
        return cls

    target_cls = unpack(cls)

    # Unsupported precision kwargs due to bug
    # JAX does not support backward pass with preferred_element_type!=input dtype
    # see JAX #31592
    BUGGED = {nnx.Conv, nnx.ConvTranspose}

    # Inspect the callable to get its effective parameters (works for class and partial)
    try:
        param_names = inspect.signature(cls.__init__).parameters.keys()
    except (ValueError, TypeError):
        # Fallback to known precision-related keys
        param_names = {"dtype", "precision", "param_dtype", "preferred_element_type"}

    if target_cls in BUGGED and "preferred_element_type" in param_names:
        kwargs.pop("preferred_element_type", None)

    # Filter out unsupported precision kwargs
    return {key: kwargs[key] for key in kwargs if key in param_names}


def get_active_precision_kwargs(dtype, precision, param_dtype, preferred_element_type) -> dict:
    """Utility function to get the active precision kwargs."""
    precision_kwargs = {}
    if dtype is not None:
        precision_kwargs["dtype"] = dtype
    if precision is not None:
        precision_kwargs["precision"] = precision
    if param_dtype is not None:
        precision_kwargs["param_dtype"] = param_dtype
    if preferred_element_type is not None:
        precision_kwargs["preferred_element_type"] = preferred_element_type
    return precision_kwargs


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
