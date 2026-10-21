"""Joint posterior draws of parameters and latent paths after temporal SMC."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
from blackjax.smc.base import map_fn

from probjax.inference.filtering.observations import validate_observations
from probjax.inference.filtering.temporal import run_temporal_filter
from probjax.inference.filtering.trajectory import sample_paths
from probjax.inference.particles import take_batch
from probjax.utils.guards import is_tracing
from probjax.utils.typing import Array, RngKey


class JointPathSamples(NamedTuple):
    parameters: Any
    paths: Any
    parameter_indices: Array


def sample_joint_paths(
    key: RngKey,
    state: Any,
    model: Any,
    ts: Any,
    observations: Any,
    *,
    observed: Any = None,
    num_samples: int = 100,
    num_particles: int | None = None,
    method: str | None = None,
    batch_size: int = 0,
) -> JointPathSamples:
    """Draw parameters from a temporal SMC state, then a path for each.

    Pass the same data as to :func:`run_temporal_smc`; only the observations
    the state has assimilated are used. Paths have shape
    ``(num_samples, num_assimilated + 1, D)``. Gaussian paths are exact given
    each drawn parameter; particle paths re-run a filter with
    ``num_particles`` and so add a finite-particle approximation. Must be
    called outside ``jit``, since the assimilated count sets array shapes.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if is_tracing(state.num_observations):
        raise ValueError("sample_joint_paths must be called outside jit.")
    count = int(state.num_observations)
    ts, observations, observed = validate_observations(ts, observations, observed)
    data = (ts[:count], observations[:count], observed[:count])
    select_key, path_key = jax.random.split(key)
    indices = jax.random.categorical(
        select_key, state.log_weights, shape=(num_samples,)
    )
    parameters = take_batch(state.parameters, indices)

    def draw(args: Any) -> Any:
        key, theta = args
        filter_key, sample_key = jax.random.split(key)
        result = run_temporal_filter(
            filter_key,
            model,
            theta,
            data[0],
            data[1],
            t0=state.t0,
            num_particles=num_particles,
            observed=data[2],
        )
        return sample_paths(sample_key, result.trace, model, theta, method=method)[:, 0]

    paths = map_fn(draw, batch_size)((
        jax.random.split(path_key, num_samples),
        parameters,
    ))
    return JointPathSamples(parameters, paths, indices)
