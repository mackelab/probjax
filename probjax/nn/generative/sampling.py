"""Cached, shape-polymorphic samplers for NNX generative models.

Data is a pytree: every event is described by a *spec*, a pytree of
:class:`jax.ShapeDtypeStruct` leaves giving the trailing event shape and
dtype of each array. :func:`probjax.nn.export.normalize_spec` builds one from
a plain shape tuple or any pytree of shapes, so structured data works wherever
a flat array does.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.export import (
    NNXExportedFunction,
    export_symbolic_batch,
    flatten_broadcast_batch,
    flatten_spec_batch,
    restore_batch,
    zeros_from_spec,
)
from probjax.utils.functions import generic_drift, split_drift
from probjax.utils.odeint import odeint
from probjax.utils.sdeint import sdeint
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey


@runtime_checkable
class SamplerLike(Protocol):
    """Callable sampling operation bound to a generative model.

    Every family provides its own sampler (an exported program, an eager
    callable, ...); the distribution view only needs to call it and to know
    whether its cached export is still valid.
    """

    @property
    def invalidated(self) -> bool: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


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


def _unflatten_trace(output, batch_shape):
    """Unflatten a trace where each leaf has shape (num_steps, flat_batch, ...)."""
    return jax.tree.map(
        lambda leaf: leaf.reshape((leaf.shape[0],) + batch_shape + leaf.shape[2:]),
        output,
    )


class _ExportedSampler(NNXExportedFunction):
    """An exported sampler bound weakly to an NNX model.

    The exported program captures model structure but receives current model
    state and initial noise as dynamic arguments. Its leading noise dimension
    is symbolic, so one export accepts every concrete batch size. Noise and
    output are pytrees matching ``spec``.
    """

    def __init__(
        self,
        model: nnx.Module,
        graphdef: nnx.GraphDef,
        exported,
        spec: PyTree[jax.ShapeDtypeStruct],
        *,
        stochastic: bool = False,
        inputs_spec: PyTree[jax.ShapeDtypeStruct] | None = None,
        trace: bool = False,
        base_sample_kwargs=None,
    ) -> None:
        super().__init__(model, graphdef, exported)
        self.spec = spec
        self.stochastic = stochastic
        self.inputs_spec = inputs_spec
        self.trace = trace
        self.base_sample_kwargs = dict(base_sample_kwargs or {})

    def from_noise(
        self,
        eps: PyTree[Array],
        *,
        rng: RngKey | None = None,
        inputs: PyTree[Array] | None = None,
    ) -> PyTree[Array]:
        """Generate from initial noise with arbitrary leading batch dimensions."""
        _, state = self.model_and_state()
        return self.from_noise_with_state(state, eps, rng=rng, inputs=inputs)

    def from_noise_with_state(
        self,
        state,
        eps: PyTree[Array],
        *,
        rng: RngKey | None = None,
        inputs: PyTree[Array] | None = None,
    ) -> PyTree[Array]:
        """Generate from noise using an explicit model state."""
        batch_shape, flat_eps = flatten_spec_batch(eps, self.spec, name="initial")
        flat_inputs = flatten_broadcast_batch(
            inputs, batch_shape, self.inputs_spec, name="inputs"
        )
        if self.stochastic:
            if rng is None:
                raise ValueError("rng is required for stochastic sampling.")
            call_args = (state, rng, flat_eps)
        else:
            call_args = (state, flat_eps)
        if flat_inputs is not None:
            call_args += (flat_inputs,)
        output = self.exported.call(*call_args)
        if self.trace:
            return _unflatten_trace(output, batch_shape)
        return restore_batch(output, batch_shape)

    def __call__(
        self,
        rng: RngKey,
        sample_shape: tuple[int, ...] = (),
        *,
        inputs: PyTree[Array] | None = None,
    ) -> PyTree[Array]:
        """Draw base noise and generate ``sample_shape`` samples."""
        model, _ = self.model_and_state()
        sample_shape = tuple(sample_shape)
        if self.stochastic:
            noise_key, sample_key = jax.random.split(rng)
        else:
            noise_key, sample_key = rng, None
        eps = model._sample_base(  # type: ignore
            noise_key, sample_shape, self.spec, **self.base_sample_kwargs
        )
        return self.from_noise(eps, rng=sample_key, inputs=inputs)

    def sample_with_state(
        self,
        state,
        rng: RngKey,
        sample_shape: tuple[int, ...] = (),
        *,
        inputs: PyTree[Array] | None = None,
    ) -> PyTree[Array]:
        """Draw samples using an explicit model state."""
        if self.invalidated:
            raise RuntimeError("This compiled operation has been invalidated.")
        model = nnx.merge(self._graphdef, state, copy=True)
        sample_shape = tuple(sample_shape)
        if self.stochastic:
            noise_key, sample_key = jax.random.split(rng)
        else:
            noise_key, sample_key = rng, None
        eps = model._sample_base(  # type: ignore
            noise_key, sample_shape, self.spec, **self.base_sample_kwargs
        )
        return self.from_noise_with_state(state, eps, rng=sample_key, inputs=inputs)


def make_map_sample_fn(
    graphdef: nnx.GraphDef,
    with_inputs: bool,
    *,
    transform: Callable,
) -> Callable:
    """Build an exported function that maps a transform over samples."""

    def sample_fn(current_state, samples, *rest):
        inputs = rest[0] if with_inputs else None
        model = nnx.merge(graphdef, current_state)
        if inputs is None:
            return jax.vmap(lambda value: transform(model, value, None))(samples)
        return jax.vmap(lambda value, condition: transform(model, value, condition))(
            samples, inputs
        )

    return sample_fn


def make_scan_sample_fn(
    graphdef: nnx.GraphDef,
    with_inputs: bool,
    *,
    xs: PyTree[Array],
    step: Callable,
) -> Callable:
    """Build an exported deterministic scan sampler."""

    def sample_fn(current_state, initial, *rest):
        inputs = rest[0] if with_inputs else None

        def scan_step(value, item):
            model = nnx.merge(graphdef, current_state)
            return step(model, value, item, inputs), None

        final, _ = jax.lax.scan(scan_step, initial, xs)
        return final

    return sample_fn


def make_ode_sample_fn(
    graphdef: nnx.GraphDef,
    with_inputs: bool,
    *,
    ts: Array,
    prototype,
    build_drift: Callable,
    method: str,
    collect_trace: bool,
) -> Callable:
    """Build an exported ODE sampler while preserving specialized drifts."""

    def evaluate(t, value, current_state, inputs=None, *, nonlinear=False):
        model = nnx.merge(graphdef, current_state)
        drift = build_drift(model, inputs)
        if nonlinear:
            return drift.nonlin(t, value)
        return drift(t, value)

    nonlinear = isinstance(prototype, split_drift)

    def drift_fn(t, value, current_state, *rest):
        inputs = rest[0] if with_inputs else None
        return evaluate(t, value, current_state, inputs, nonlinear=nonlinear)

    if nonlinear:
        drift = split_drift(prototype.lin_coeff, drift_fn)
    else:
        drift = generic_drift(drift_fn)

    def sample_fn(current_state, initial, *rest):
        extra = (rest[0],) if with_inputs else ()
        return odeint(
            drift,
            initial,
            ts,
            current_state,
            *extra,
            solver=method,
            dtype=None,
            collect_trace=collect_trace,
        )

    return sample_fn


def make_sde_sample_fn(
    graphdef: nnx.GraphDef,
    with_inputs: bool,
    *,
    ts: Array,
    build_drift_and_diffusion: Callable,
    method: str,
    collect_trace: bool,
) -> Callable:
    """Build an exported SDE sampler with one independent path per sample."""

    def drift_fn(t, value, current_state, inputs=None):
        model = nnx.merge(graphdef, current_state)
        drift, _ = build_drift_and_diffusion(model, inputs)
        return drift(t, value)

    def diffusion_fn(t, value, current_state, inputs=None):
        model = nnx.merge(graphdef, current_state)
        _, diffusion = build_drift_and_diffusion(model, inputs)
        return diffusion(t, value)

    drift = generic_drift(drift_fn)
    diffusion = generic_drift(diffusion_fn)

    def one_sample(current_state, rng, initial, *rest):
        extra = (rest[0],) if with_inputs else ()
        return sdeint(
            rng,
            drift,
            diffusion,
            initial,
            ts,
            current_state,
            *extra,
            solver=method,
            dtype=None,
            collect_trace=collect_trace,
        )

    def sample_fn(current_state, rng, initial, *rest):
        batch_size = jax.tree.leaves(initial)[0].shape[0]
        keys = jax.random.split(rng, batch_size)
        in_axes = (None, 0, 0, 0) if with_inputs else (None, 0, 0)
        result = jax.vmap(one_sample, in_axes=in_axes)(
            current_state, keys, initial, *rest
        )
        if collect_trace:
            return jax.tree.map(lambda value: jnp.moveaxis(value, 0, 1), result)
        return result

    return sample_fn


def _export_sampler(
    state: nnx.State,
    spec: PyTree[jax.ShapeDtypeStruct],
    sample_fn: Callable,
    *,
    stochastic: bool = False,
    inputs_spec: PyTree[jax.ShapeDtypeStruct] | None = None,
):
    """Export ``sample_fn(state, [rng,] eps, [inputs])`` with a symbolic
    leading batch axis shared by every ``eps`` and ``inputs`` leaf."""
    if stochastic:
        args = (state, jax.random.key(0), zeros_from_spec(spec))
        shape_specs = (None, None, "b, ...")
    else:
        args = (state, zeros_from_spec(spec))
        shape_specs = (None, "b, ...")
    if inputs_spec is not None:
        args += (zeros_from_spec(inputs_spec),)
        shape_specs += ("b, ...",)
    return export_symbolic_batch(
        sample_fn,
        args,
        shape_specs,
    )
