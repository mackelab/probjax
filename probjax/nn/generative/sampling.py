"""Cached, shape-polymorphic samplers for NNX generative models."""

from __future__ import annotations

import weakref
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import export as jax_export

from probjax.utils.typing import Array, RngKey

_SAMPLER_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class BuiltSampler:
    """An exported sampler bound weakly to an NNX model.

    The exported program captures model structure but receives current model
    state and initial noise as dynamic arguments. Its leading noise dimension
    is symbolic, so one export accepts every concrete batch size.
    """

    def __init__(
        self,
        model: nnx.Module,
        graphdef: nnx.GraphDef,
        exported: jax_export.Exported,
        event_shape: tuple[int, ...],
        dtype,
        *,
        base: str,
        stochastic: bool = False,
    ) -> None:
        self._model_ref = weakref.ref(model)
        self._graphdef = graphdef
        self.exported = exported
        self.event_shape = event_shape
        self.dtype = jnp.dtype(dtype)
        self.base = base
        self.stochastic = stochastic

    def _model_and_state(self):
        model = self._model_ref()
        if model is None:
            raise RuntimeError("The model used to build this sampler no longer exists.")
        graphdef, state = nnx.split(model)
        if graphdef != self._graphdef:
            raise RuntimeError(
                "The model structure changed after this sampler was built. "
                "Call build_sampler() again."
            )
        return model, state

    def from_noise(self, eps: Array, *, rng: RngKey | None = None):
        """Generate from initial noise with arbitrary leading batch dimensions."""
        eps = jnp.asarray(eps, dtype=self.dtype)
        event_ndim = len(self.event_shape)
        if event_ndim:
            if eps.ndim < event_ndim or eps.shape[-event_ndim:] != self.event_shape:
                raise ValueError(
                    "Expected trailing event shape "
                    f"{self.event_shape}, got {eps.shape}."
                )
            batch_shape = eps.shape[:-event_ndim]
        else:
            batch_shape = eps.shape

        flat_eps = eps.reshape((-1,) + self.event_shape)
        _, state = self._model_and_state()
        if self.stochastic:
            if rng is None:
                raise ValueError("rng is required for stochastic sampling.")
            keys = jax.random.split(rng, flat_eps.shape[0])
            output = self.exported.call(state, keys, flat_eps)
        else:
            output = self.exported.call(state, flat_eps)
        return jax.tree.map(
            lambda leaf: leaf.reshape(batch_shape + leaf.shape[1:]), output
        )

    def __call__(
        self,
        rng: RngKey,
        sample_shape: tuple[int, ...] = (),
    ):
        """Draw base noise and generate ``sample_shape`` samples."""
        model, _ = self._model_and_state()
        shape = tuple(sample_shape) + self.event_shape
        if self.stochastic:
            noise_key, sample_key = jax.random.split(rng)
        else:
            noise_key, sample_key = rng, None
        eps = jax.random.normal(noise_key, shape, dtype=self.dtype)
        if self.base == "flow":
            eps = eps * model.std0.get_value() + model.mu0.get_value()
        return self.from_noise(eps, rng=sample_key)


def export_sampler(
    state: nnx.State,
    event_shape: tuple[int, ...],
    dtype,
    sample_fn: Callable,
    *,
    stochastic: bool = False,
) -> jax_export.Exported:
    """Export ``sample_fn(state, eps)`` with a symbolic leading batch axis."""
    example_eps = jnp.zeros((1,) + event_shape, dtype=dtype)
    if stochastic:
        example_keys = jax.random.split(jax.random.key(0), 1)
        args = (state, example_keys, example_eps)
        shape_specs = (None, "b, ...", "b, ...")
    else:
        args = (state, example_eps)
        shape_specs = (None, "b, ...")
    specs = jax_export.symbolic_args_specs(
        args,
        shape_specs,
        constraints=("b >= 1",),
    )
    return jax_export.export(jax.jit(sample_fn))(*specs)


def cached_sampler(model: nnx.Module, key: tuple, build: Callable[[], BuiltSampler]):
    """Return the per-model sampler for ``key``, building it at most once."""
    model_cache = _SAMPLER_CACHE.setdefault(model, {})
    sampler = model_cache.get(key)
    if sampler is not None:
        try:
            sampler._model_and_state()
        except RuntimeError:
            sampler = None
    if sampler is None:
        sampler = build()
        model_cache[key] = sampler
    return sampler


def clear_sampler_cache(model: nnx.Module) -> None:
    """Invalidate every exported sampler associated with ``model``."""
    _SAMPLER_CACHE.pop(model, None)
