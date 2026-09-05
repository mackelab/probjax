import inspect
from functools import partial
from typing import Optional, Sequence, Tuple

import flax.nnx as nnx
import jax.numpy as jnp
from jax import lax
from jax.ops import segment_max  # segment reduction (available in JAX)

from probjax.utils.optional import require_ott
from probjax.utils.typing import Array, ArrayLike, ModuleLikeType


_RNG_SUPPORT_BY_TYPE: dict[type, bool] = {
    nnx.Linear: False,
}


def identity_1x1(_, shape: Sequence[int], dtype=jnp.float32):
    """Kernel init for an all-ones-kernel Conv that starts as identity.

    Shape is (*spatial, C_in, C_out) for any number of spatial dims — 1-D
    convs give (1, C_in, C_out), 2-D give (1, 1, C_in, C_out). If C_in ≠ C_out
    the extra channels are zero-filled.
    """
    k = jnp.zeros(shape, dtype)
    diag = jnp.arange(min(shape[-2], shape[-1]))
    # set W[0, ..., 0, i, i] = 1
    spatial_origin = (0,) * (len(shape) - 2)
    k = k.at[(*spatial_origin, diag, diag)].set(1.0)
    return k


def pad_to_power_of_2(arr: Array, min_size: int = 16, axis=(-1,)) -> Array:
    """Pad the array to the next power of 2 greater than min_size along given axis."""

    def next_power_of_2(x):
        return 1 << (x - 1).bit_length()

    target_shape = list(arr.shape)
    for ax in axis:
        seq_len = arr.shape[ax]
        if seq_len < min_size:
            target_shape[ax] = min_size
        else:
            target_shape[ax] = next_power_of_2(seq_len)

    pad_width = [
        (0, target - current)
        for current, target in zip(arr.shape, target_shape, strict=False)
    ]
    return jnp.pad(arr, pad_width)


def flatten_to_btd(x: ArrayLike) -> Tuple[Array, Tuple[int, ...]]:
    """Flatten leading batch dims to (B, T, D); returns flattened tensor and original shape."""
    x = jnp.asarray(x)
    if x.ndim < 2:
        raise ValueError(
            f"Expected tensor with at least 2 dims (time, dim); got shape {x.shape}"
        )
    orig = tuple(x.shape)
    time_dim = x.shape[-2]
    feature_dim = x.shape[-1]
    x = x.reshape((-1, time_dim, feature_dim))
    return x, orig


def restore_from_btd(x: Array, orig_shape: Optional[tuple[int, ...]]) -> Array:
    """Restore tensor from (B, T, D) back to original leading batch dims."""
    if orig_shape is None:
        return x
    return x.reshape(orig_shape)


def normalize_attn_mask(mask: Array | None) -> Array | None:
    """Normalize attention mask shapes to broadcast with [B, H, T, T]."""
    if mask is None:
        return None
    mask = jnp.asarray(mask)
    if mask.ndim == 2:
        return mask[None, None, :, :]
    if mask.ndim == 3:
        return mask[:, None, :, :]
    if mask.ndim == 4:
        return mask
    raise ValueError(f"Mask must have ndim 2, 3, or 4; got {mask.ndim}.")


def normalize_attn_bias(bias: Array | None) -> Array | None:
    """Normalize attention bias shapes to broadcast with attention logits [B, H, T, T]."""
    if bias is None:
        return None
    bias = jnp.asarray(bias)
    if bias.ndim == 2:
        return bias[None, None, :, :]
    if bias.ndim == 3:
        return bias[:, None, :, :]
    if bias.ndim == 4:
        return bias
    raise ValueError(f"Bias must have ndim 2, 3, or 4; got {bias.ndim}.")


def filter_supported_kwargs(ctor, **kwargs) -> dict:
    """Keep only kwargs the constructor's signature accepts.

    Works for classes, functions, and functools.partial wrappers. Used to
    pass optional metadata (e.g. sharding) to layers that support it while
    remaining compatible with custom layer classes that don't.
    """
    target = ctor
    while isinstance(target, partial):
        target = target.func
    try:
        fn = target.__init__ if isinstance(target, type) else target
        param_names = inspect.signature(fn).parameters.keys()
    except (ValueError, TypeError):
        return {}
    return {key: kwargs[key] for key in kwargs if key in param_names}


def filter_precision_kwargs(cls: ModuleLikeType, **kwargs):
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
        param_names = inspect.signature(target_cls.__init__).parameters.keys()
    except (ValueError, TypeError):
        # Fallback to known precision-related keys
        param_names = {"dtype", "precision", "param_dtype", "preferred_element_type"}

    if target_cls in BUGGED and "preferred_element_type" in param_names:
        kwargs.pop("preferred_element_type", None)

    # Filter out unsupported precision kwargs
    return {key: kwargs[key] for key in kwargs if key in param_names}


def get_active_precision_kwargs(
    dtype, precision, param_dtype, preferred_element_type
) -> dict:
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


def call_with_optional_rng(module, *args, rng=None, **kwargs):
    """Call a module/function and pass `rng` only if supported."""
    if rng is None:
        return module(*args, **kwargs)

    if not module_accepts_rng(module):
        return module(*args, **kwargs)

    return module(*args, rng=rng, **kwargs)


def module_accepts_rng(module) -> bool:
    module_type = type(module)
    cached = _RNG_SUPPORT_BY_TYPE.get(module_type)
    if cached is not None:
        return cached

    call_target = module.__call__ if hasattr(module, "__call__") else module
    try:
        params = inspect.signature(call_target).parameters.values()
        supports_rng = any(
            p.name == "rng" or p.kind == inspect.Parameter.VAR_KEYWORD for p in params
        )
    except (TypeError, ValueError):
        supports_rng = True
    _RNG_SUPPORT_BY_TYPE[module_type] = supports_rng
    return supports_rng


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
    costs, pointcloud, _, linear_problem, sinkhorn = require_ott()
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
