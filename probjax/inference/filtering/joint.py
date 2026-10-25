"""Joint posterior draws of parameters and latent paths after temporal SMC."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax

from probjax.inference.filtering.trajectory import sample_paths
from probjax.inference.particles import take_batch
from probjax.inference.runner import run
from probjax.utils.guards import is_tracing
from probjax.utils.typing import Array


class JointPathSamples(NamedTuple):
    parameters: Any
    paths: Any
    parameter_indices: Array


def sample_joint_paths(
    rng,
    state: Any,
    filter: Any,
    model: Any,
    *,
    num_samples: int = 100,
    method: str | None = None,
    batch_size: int = 0,
) -> JointPathSamples:
    """Draw parameters from a temporal SMC state, then a path for each.

    Each drawn ``theta`` re-runs ``filter`` over the observations the state
    has assimilated (it keeps them), then draws one path with
    :func:`sample_paths`. Paths have shape ``(num_samples,
    num_observations + 1, D)``. Gaussian paths are exact given each
    parameter; particle paths carry the filter's finite-particle error. Must
    be called outside ``jit``, since the number of observations sets the
    shapes.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if is_tracing(state.num_observations):
        raise ValueError("sample_joint_paths must be called outside jit.")
    count = int(state.num_observations)
    data = jax.tree.map(lambda buffer: buffer[:count], state.data)
    select_key, path_key = jax.random.split(rng)
    indices = jax.random.categorical(
        select_key, state.log_weights, shape=(num_samples,)
    )
    parameters = take_batch(state.parameters, indices)

    def draw(args: Any) -> Any:
        key, theta = args
        init_key, filter_key, sample_key = jax.random.split(key, 3)
        initial = filter.init(init_key, theta, state.t0)
        trace = run(
            filter_key,
            filter,
            initial,
            theta,
            count,
            args=data,
            collect=("state", "info"),
        ).trace
        paths = sample_paths(
            sample_key,
            model,
            theta,
            initial,
            trace["state"],
            trace["info"],
            method=method,
        )
        return paths[:, 0]

    keys = jax.random.split(path_key, num_samples)
    if batch_size:
        paths = jax.lax.map(draw, (keys, parameters), batch_size=batch_size)
    else:
        paths = jax.vmap(draw)((keys, parameters))
    return JointPathSamples(parameters, paths, indices)
