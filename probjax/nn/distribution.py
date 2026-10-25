"""Distribution views of generative models.

``model.as_dist(**options)`` binds an event spec and static sampling options
(``num_steps``, ``mode="sde"``, ...) and returns a
:class:`~probjax.stats.base.DistributionAPI`:

>>> dist = EDM(net, event_spec=(d,)).as_dist(num_steps=50)
>>> samples = dist.sample(key, (N,))
>>> dist.condition(context=c).sample(key, (N,))

Calls go through the model's jitted operations
(:meth:`~probjax.nn.generative.base.GenerativeModel._compiled`), which read the
model's current weights on every call; like any ``jax.jit`` function they
compile once per input shape. A view is an ordinary JAX function, so it can be
called inside ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

Named inputs follow one rule: inputs bound with :meth:`ModelDistribution.condition`
are single events shared by every sample, while inputs passed to a call carry
the call's batch (``sample_shape`` for sampling, the leading axes of ``x`` for
``logpdf``).
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from probjax.stats.base import DistributionAPI
from probjax.utils.typing import Array, PyTree, RngKey

__all__ = ["ModelDistribution"]


def _flatten_batch(value, batch_shape: tuple[int, ...], name: str):
    """Merge the leading ``batch_shape`` axes of every leaf into one."""
    size = math.prod(batch_shape)

    def flatten(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.shape[: len(batch_shape)] != batch_shape:
            raise ValueError(
                f"{name} must lead with the batch shape {batch_shape}; got a "
                f"leaf of shape {leaf.shape}. Bind inputs shared by every "
                "sample with condition() instead."
            )
        return leaf.reshape((size,) + leaf.shape[len(batch_shape) :])

    return jax.tree.map(flatten, value)


def _restore_batch(value, batch_shape: tuple[int, ...], *, time_first=False):
    """Split the merged batch axis back into ``batch_shape``."""
    lead = 1 if time_first else 0
    return jax.tree.map(
        lambda leaf: leaf.reshape(
            leaf.shape[:lead] + batch_shape + leaf.shape[lead + 1 :]
        ),
        value,
    )


class ModelDistribution(DistributionAPI):
    """A generative model viewed as a distribution over one event."""

    def __init__(self, model, event_spec, *, inputs=None, options=None) -> None:
        self._model = model
        self._event_spec = event_spec
        self._inputs = dict(inputs or {})
        self._options = dict(options or {})

    @property
    def model(self):
        return self._model

    @property
    def event_spec(self):
        return self._event_spec

    @property
    def event_shape(self):
        return jax.tree.map(lambda leaf: leaf.shape, self._event_spec)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    def condition(self, **inputs) -> ModelDistribution:
        """Bind named inputs shared by every sample (overriding bound ones)."""
        return ModelDistribution(
            self._model,
            self._event_spec,
            inputs={**self._inputs, **inputs},
            options=self._options,
        )

    def _inputs_for(self, inputs: dict, batch_shape: tuple[int, ...]) -> dict:
        """Call inputs flattened, bound inputs broadcast, to one batch axis."""
        size = math.prod(batch_shape)
        bound = {
            name: jax.tree.map(
                lambda leaf: jnp.broadcast_to(leaf, (size,) + jnp.shape(leaf)), value
            )
            for name, value in self._inputs.items()
            if name not in inputs
        }
        return {**bound, **_flatten_batch(inputs, batch_shape, "inputs")}

    def _draw(self, rng: RngKey, shape, inputs: dict, *, trace: bool):
        shape = tuple(shape)
        size = math.prod(shape)
        noise_key, sample_key = jax.random.split(rng)
        noise = self._model.sample_base(
            noise_key, (size,), self._event_spec, **self._options
        )
        # Only path-producing families take ``trace``; the others reject it.
        options = {**self._options, "trace": True} if trace else self._options
        fn, state = self._model._compiled("sample_from_noise", **options)
        out = fn(state, noise, sample_key, self._inputs_for(inputs, shape))
        return _restore_batch(out, shape, time_first=trace)

    def sample(
        self, rng: RngKey, shape: tuple[int, ...] = (), **inputs: PyTree[Array]
    ) -> PyTree[Array]:
        """Draw ``shape`` samples; inputs carry ``shape`` as leading axes."""
        return self._draw(rng, shape, inputs, trace=False)

    def rvs(
        self,
        rng: RngKey,
        shape: tuple[int, ...] = (),
        name: str | None = None,
        **inputs: Any,
    ) -> PyTree[Array]:
        del name
        return self.sample(rng, shape, **inputs)

    def sample_path(
        self, rng: RngKey, shape: tuple[int, ...] = (), **inputs: PyTree[Array]
    ) -> PyTree[Array]:
        """Draw samples with their solver path, time-first: ``(steps, *shape, ...)``."""
        return self._draw(rng, shape, inputs, trace=True)

    def logpdf(self, x: PyTree[Array], **inputs: PyTree[Array]) -> Array:
        """Log density of ``x``, whose leading axes are batch axes."""
        if not hasattr(self._model, "logpdf"):
            raise NotImplementedError(
                f"{type(self._model).__name__} has no tractable log-density."
            )
        spec_leaves, treedef = jax.tree.flatten(self._event_spec)
        try:
            x_leaves = treedef.flatten_up_to(x)
        except (TypeError, ValueError) as err:
            raise ValueError(f"x does not match the event structure {treedef}.") from err
        first = jnp.asarray(x_leaves[0])
        batch_shape = first.shape[: first.ndim - len(spec_leaves[0].shape)]
        x = treedef.unflatten([
            jnp.asarray(leaf, spec.dtype)
            for leaf, spec in zip(x_leaves, spec_leaves, strict=True)
        ])
        fn, state = self._model._compiled("logpdf")
        out = fn(
            state,
            _flatten_batch(x, batch_shape, "x"),
            self._inputs_for(inputs, batch_shape),
        )
        return out.reshape(batch_shape)

    def pdf(self, x: PyTree[Array], **inputs: PyTree[Array]) -> Array:
        return jnp.exp(self.logpdf(x, **inputs))

    def __repr__(self) -> str:
        return f"{type(self._model).__name__}(event_spec={self._event_spec!r})"
