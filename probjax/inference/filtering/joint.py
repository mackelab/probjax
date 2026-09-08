"""Posterior parameter/trajectory draws using existing filter and smoother backends."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from blackjax.smc.base import map_fn

from probjax.inference.filtering.temporal import run_temporal_filter
from probjax.inference.filtering.trajectory import (
    sample_gaussian_paths,
    sample_particle_paths,
)


class JointPathSamples(NamedTuple):
    parameters: object
    paths: object
    parameter_indices: object
    valid: object


def sample_joint_paths(
    key,
    state,
    backend,
    replay,
    *,
    num_samples=100,
    transition_fn=None,
    transition_logdensity_fn=None,
    method='backward',
    batch_size=0,
):
    """Sample parameters then conditional paths; paths have shape (M,T+1,D).

    ReplayData must contain exactly the assimilated prefix (no future suffix).
    transition_fn(theta,t0,t1)->Phi selects Gaussian backward sampling.
    Otherwise transition_logdensity_fn(theta,new,old,t0,t1) supplies FFBSi density.
    Gaussian conditional draws are exact given the SMC parameter approximation;
    rerun particle-filter smoothing is an additional finite-particle approximation,
    not an exact joint draw from the SMC² extended state. Use PGAS for further moves.
    """
    if num_samples < 1:
        raise ValueError('num_samples must be positive.')
    if (
        not isinstance(state.num_observations, jax.core.Tracer)
        and int(state.num_observations) != replay.ts.shape[0]
    ):
        raise ValueError('Joint sampling requires exactly the assimilated data prefix.')
    if (
        transition_fn is None
        and method == 'backward'
        and transition_logdensity_fn is None
    ):
        raise ValueError(
            'Supply a Gaussian transition matrix or a particle transition density.'
        )
    select_key, path_key = jax.random.split(key)
    indices = jax.random.categorical(
        select_key, state.log_weights, shape=(num_samples,)
    )
    parameters = jax.tree.map(lambda x: x[indices], state.parameters)

    def draw(args):
        key, theta = args
        init_key, filter_key, sample_key = jax.random.split(key, 3)
        result = run_temporal_filter(
            backend,
            filter_key,
            backend.init(init_key, theta, state.t0),
            theta,
            replay.ts,
            replay.observations,
            observed=replay.observed,
        )
        if transition_fn is not None:
            return sample_gaussian_paths(
                sample_key, result.trace, lambda a, b: transition_fn(theta, a, b)
            )[:, 0]
        density = (
            None
            if transition_logdensity_fn is None
            else lambda new, old, a, b: transition_logdensity_fn(theta, new, old, a, b)
        )
        return sample_particle_paths(
            sample_key, result.trace, method=method, transition_logdensity_fn=density
        )[:, 0]

    paths = map_fn(draw, batch_size)((
        jax.random.split(path_key, num_samples),
        parameters,
    ))
    valid = state.num_observations == replay.ts.shape[0]
    valid = valid & (state.t == (replay.ts[-1] if replay.ts.shape[0] else state.t0))
    return JointPathSamples(
        parameters, jnp.where(valid, paths, jnp.nan), indices, valid
    )
