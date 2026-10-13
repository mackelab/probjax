"""Joint posterior draws of parameters and latent paths after temporal SMC."""

from typing import NamedTuple

import jax
from blackjax.smc.base import map_fn

from probjax.inference.filtering.temporal import _observations, run_temporal_filter
from probjax.inference.filtering.trajectory import sample_paths


class JointPathSamples(NamedTuple):
    parameters: object
    paths: object
    parameter_indices: object


def sample_joint_paths(
    key,
    state,
    model,
    ts,
    observations,
    *,
    observed=None,
    num_samples=100,
    num_particles=None,
    method=None,
    batch_size=0,
):
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
    if isinstance(state.num_observations, jax.core.Tracer):
        raise ValueError("sample_joint_paths must be called outside jit.")
    count = int(state.num_observations)
    ts, observations, observed = _observations(ts, observations, observed)
    data = (ts[:count], observations[:count], observed[:count])
    select_key, path_key = jax.random.split(key)
    indices = jax.random.categorical(
        select_key, state.log_weights, shape=(num_samples,)
    )
    parameters = jax.tree.map(lambda x: x[indices], state.parameters)

    def draw(args):
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
        return sample_paths(sample_key, result.trace, model, theta, method=method)[
            :, 0
        ]

    paths = map_fn(draw, batch_size)((
        jax.random.split(path_key, num_samples),
        parameters,
    ))
    return JointPathSamples(parameters, paths, indices)
