"""Sampling helpers shared by the generative families.

Data is a pytree: every event is described by a *spec*, a pytree of
:class:`jax.ShapeDtypeStruct` leaves giving the trailing event shape and
dtype of each array. :func:`normalize_spec` builds one from a plain shape
tuple or any pytree of shapes, so structured data works wherever a flat array
does.

The integrators take the model and a ``build(model, inputs)`` function rather
than a ready drift: the model's state travels through the solver as an
argument, never as a closure, which is what lets the solvers' custom
primitives treat the drift as static.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.utils.functions import generic_drift, split_drift
from probjax.utils.jaxutils import leading_dim
from probjax.utils.odeint import odeint
from probjax.utils.sdeint import sdeint
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey

__all__ = ["integrate_ode", "integrate_sde", "normalize_spec", "sample_normal"]


def _is_shape(value) -> bool:
    return isinstance(value, tuple) and all(isinstance(dim, int) for dim in value)


def normalize_spec(value, dtype=jnp.float32) -> PyTree[jax.ShapeDtypeStruct]:
    """Normalize a shape, shape/dtype struct, or pytree of either."""

    def is_leaf(node):
        return isinstance(node, jax.ShapeDtypeStruct) or _is_shape(node)

    def to_struct(node):
        if isinstance(node, jax.ShapeDtypeStruct):
            return node
        return jax.ShapeDtypeStruct(tuple(int(dim) for dim in node), jnp.dtype(dtype))

    return jax.tree.map(to_struct, value, is_leaf=is_leaf)


def sample_normal(
    rng: RngKey,
    sample_shape: tuple[int, ...],
    spec: PyTree[jax.ShapeDtypeStruct],
    *,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> PyTree[Array]:
    """Draw a normal pytree matching ``sample_shape + spec``."""
    spec_leaves, treedef = jax.tree.flatten(spec)
    keys = jax.random.split(rng, len(spec_leaves))
    noise_leaves = [
        jax.random.normal(key, sample_shape + leaf.shape, leaf.dtype)
        for key, leaf in zip(keys, spec_leaves, strict=True)
    ]

    def parameter_leaves(value):
        try:
            return treedef.flatten_up_to(value)
        except (TypeError, ValueError):
            return [value] * len(spec_leaves)

    return treedef.unflatten([
        value * leaf_scale + leaf_loc
        for value, leaf_loc, leaf_scale in zip(
            noise_leaves,
            parameter_leaves(loc),
            parameter_leaves(scale),
            strict=True,
        )
    ])


def integrate_ode(
    model: nnx.Module,
    build_drift: Callable,
    noise: PyTree[Array],
    ts: Array,
    *,
    inputs=None,
    method: str,
    trace: bool = False,
) -> PyTree[Array]:
    """Integrate ``build_drift(model, inputs)`` from ``noise`` over ``ts``.

    A :class:`~probjax.utils.functions.split_drift` keeps its split, so
    exponential integrators still see the linear part.
    """
    graphdef, state = nnx.split(model)
    prototype = build_drift(model, inputs)
    nonlinear = isinstance(prototype, split_drift)

    def evaluate(t, value, state, inputs):
        drift = build_drift(nnx.merge(graphdef, state), inputs)
        return drift.nonlin(t, value) if nonlinear else drift(t, value)

    drift = (
        split_drift(prototype.lin_coeff, evaluate)
        if nonlinear
        else generic_drift(evaluate)
    )
    return odeint(
        drift, noise, ts, state, inputs, solver=method, dtype=None, collect_trace=trace
    )


def integrate_sde(
    model: nnx.Module,
    build_terms: Callable,
    rng: RngKey,
    noise: PyTree[Array],
    ts: Array,
    *,
    inputs=None,
    method: str,
    trace: bool = False,
) -> PyTree[Array]:
    """Integrate the SDE ``build_terms(model, inputs) -> (drift, diffusion)``.

    Every sample gets its own key and path; a trace is time-first,
    ``(num_steps, batch, ...)``.
    """
    graphdef, state = nnx.split(model)

    def drift_fn(t, value, state, inputs):
        drift, _ = build_terms(nnx.merge(graphdef, state), inputs)
        return drift(t, value)

    def diffusion_fn(t, value, state, inputs):
        _, diffusion = build_terms(nnx.merge(graphdef, state), inputs)
        return diffusion(t, value)

    def one_sample(key, initial, inputs):
        return sdeint(
            key,
            generic_drift(drift_fn),
            generic_drift(diffusion_fn),
            initial,
            ts,
            state,
            inputs,
            solver=method,
            dtype=None,
            collect_trace=trace,
        )

    keys = jax.random.split(rng, leading_dim(noise))
    result = jax.vmap(one_sample)(keys, noise, inputs)
    if trace:
        return jax.tree.map(lambda value: jnp.moveaxis(value, 0, 1), result)
    return result
