"""Distribution views for learned generative models.

Most generative-model classes in :mod:`probjax.nn.generative` don't carry an
intrinsic ``event_shape`` (a flow matcher trained on R^d looks the same as
one trained on R^k), and their sampling pipelines have free parameters
(``num_steps``, ``mode="ode"`` vs ``"sde"``). A model distribution binds
those options and lazily compiles efficient sampling and density operations:

>>> ddpm = EDM(net, ...)              # already trained
>>> dist = ddpm.as_dist(event_spec=(d,), num_steps=50)
>>> samples = dist.sample(key, shape=(N,))

Normalizing-flow views additionally compile ``logpdf``; diffusion and flow
matching remain sample-only unless they implement a density operation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from probjax.nn.export import (
    NNXExportedFunction,
    export_symbolic_batch,
    flatten_broadcast_batch,
    flatten_spec_batch,
    normalize_spec,
    restore_batch,
    spec_from_value,
    zeros_from_spec,
)
from probjax.stats.base import DistributionAPI
from probjax.utils.typing import Array, PyTree, RngKey


class _ExportedLogPDF(NNXExportedFunction):
    def __init__(self, model, graphdef, exported, spec, *, context_spec=None):
        super().__init__(model, graphdef, exported)
        self.spec = spec
        self.context_spec = context_spec

    def __call__(self, value, *, context=None):
        _, state = self.model_and_state()
        return self.with_state(state, value, context=context)

    def with_state(self, state, value, *, context=None):
        """Evaluate the log density using an explicit model state."""
        batch_shape, flat_value = flatten_spec_batch(value, self.spec, name="value")
        flat_context = flatten_broadcast_batch(
            context, batch_shape, self.context_spec, name="context"
        )
        call_args = (state, flat_value)
        if flat_context is not None:
            call_args += (flat_context,)
        return restore_batch(self.exported.call(*call_args), batch_shape)


def _export_logpdf(state, spec, logpdf_fn, *, context_spec=None):
    args = (state, zeros_from_spec(spec))
    axis_specs = (None, "b, ...")
    if context_spec is not None:
        args += (zeros_from_spec(context_spec),)
        axis_specs += ("b, ...",)
    return export_symbolic_batch(logpdf_fn, args, axis_specs)


class _ModelDistribution(DistributionAPI):
    """Lazy, compiled distribution view of a generative model.

    Sampling and density exports are built on first use and then served from
    the model's export cache. Events and optional context may both be pytrees.
    """

    def __init__(
        self,
        model,
        event_spec,
        *,
        context_spec=None,
        context=None,
        sampler_options: dict[str, Any] | None = None,
        logpdf_options: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        options = dict(sampler_options or {})
        if "collect_trace" in options:
            raise TypeError(
                "collect_trace is not a distribution option; use sample_path() "
                "or path_from()."
            )
        dtype = options.get("dtype", jnp.float32)
        if context_spec is None and context is not None:
            context_spec = spec_from_value(context)

        self._model_instance = model
        self._event_spec = normalize_spec(event_spec, dtype)
        self._context_spec = (
            None if context_spec is None else normalize_spec(context_spec, dtype)
        )
        self._context = context
        self._sampler_options = options
        self._logpdf_options = dict(logpdf_options or {})
        self._name = name or type(model).__name__
        self._sampler = None
        self._path_sampler = None
        self._logpdf = None

    def _model(self):
        return self._model_instance

    @property
    def event_spec(self):
        return self._event_spec

    @property
    def context_spec(self):
        return self._context_spec

    @property
    def event_shape(self):
        return jax.tree.map(lambda leaf: leaf.shape, self._event_spec)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def has_logpdf(self) -> bool:
        return callable(getattr(self._model(), "_distribution_logpdf", None))

    def _resolve_context(self, context):
        return self._context if context is None else context

    def _get_sampler(self, *, path=False):
        attribute = "_path_sampler" if path else "_sampler"
        sampler = getattr(self, attribute)
        if sampler is not None and sampler.invalidated:
            sampler = None
        if sampler is None:
            options = dict(self._sampler_options)
            if path:
                options["collect_trace"] = True
            sampler = self._model()._distribution_sampler(
                self._event_spec,
                context_spec=self._context_spec,
                **options,
            )
            setattr(self, attribute, sampler)
        return sampler

    def _get_logpdf(self):
        if not self.has_logpdf:
            raise NotImplementedError(f"{self!r} does not expose a tractable logpdf.")
        if self._logpdf is not None and self._logpdf.invalidated:
            self._logpdf = None
        if self._logpdf is None:
            options = dict(self._logpdf_options)
            self._logpdf = self._model()._distribution_logpdf(
                self._event_spec,
                context_spec=self._context_spec,
                **options,
            )
        return self._logpdf

    def compile(self, *operations: str):
        """Eagerly build selected operations; defaults to every available one."""
        operations = operations or (
            ("sample", "logpdf") if self.has_logpdf else ("sample",)
        )
        for operation in operations:
            if operation == "sample":
                self._get_sampler()
            elif operation == "sample_path":
                self._get_sampler(path=True)
            elif operation == "logpdf":
                self._get_logpdf()
            else:
                raise ValueError(f"Unknown distribution operation {operation!r}.")
        return self

    def model_state(self):
        """Return the current model state for explicit-state evaluation."""
        operation = self._get_logpdf() if self.has_logpdf else self._get_sampler()
        _, state = operation.model_and_state()
        return jax.tree.map(lambda value: value.copy(), state)

    def sample_with_state(
        self,
        state,
        rng: RngKey,
        shape: tuple[int, ...] = (),
        *,
        context: PyTree[Array] | None = None,
    ) -> PyTree[Array]:
        """Draw samples using an explicit model state."""
        return self._get_sampler().sample_with_state(
            state,
            rng,
            tuple(shape),
            context=self._resolve_context(context),
        )

    def logpdf_with_state(
        self,
        state,
        value: PyTree[Array],
        *,
        context: PyTree[Array] | None = None,
    ) -> Array:
        """Evaluate the log density using an explicit model state."""
        return self._get_logpdf().with_state(
            state,
            value,
            context=self._resolve_context(context),
        )

    def sample(
        self,
        rng: RngKey,
        shape: tuple[int, ...] = (),
        *,
        context: PyTree[Array] | None = None,
    ) -> PyTree[Array]:
        return self._get_sampler()(
            rng,
            tuple(shape),
            context=self._resolve_context(context),
        )

    def sample_from(
        self,
        initial: PyTree[Array],
        *,
        rng: RngKey | None = None,
        context: PyTree[Array] | None = None,
    ) -> PyTree[Array]:
        return self._get_sampler().from_noise(
            initial,
            rng=rng,
            context=self._resolve_context(context),
        )

    def sample_path(
        self,
        rng: RngKey,
        shape: tuple[int, ...] = (),
        *,
        context: PyTree[Array] | None = None,
    ) -> PyTree[Array]:
        return self._get_sampler(path=True)(
            rng,
            tuple(shape),
            context=self._resolve_context(context),
        )

    def path_from(
        self,
        initial: PyTree[Array],
        *,
        rng: RngKey | None = None,
        context: PyTree[Array] | None = None,
    ) -> PyTree[Array]:
        return self._get_sampler(path=True).from_noise(
            initial,
            rng=rng,
            context=self._resolve_context(context),
        )

    def rvs(
        self,
        rng: RngKey,
        shape: tuple[int, ...] = (),
        name: str | None = None,
        *,
        context: PyTree[Array] | None = None,
        **kwargs: Any,
    ) -> PyTree[Array]:
        del name
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected sampling arguments: {names}.")
        return self.sample(rng, shape, context=context)

    def logpdf(
        self,
        value: PyTree[Array],
        *,
        context: PyTree[Array] | None = None,
    ) -> Array:
        return self._get_logpdf()(
            value,
            context=self._resolve_context(context),
        )

    def pdf(
        self,
        value: PyTree[Array],
        *,
        context: PyTree[Array] | None = None,
    ) -> Array:
        return jnp.exp(self.logpdf(value, context=context))

    def condition(self, context) -> "_ModelDistribution":
        return _ModelDistribution(
            self._model(),
            self._event_spec,
            context_spec=self._context_spec,
            context=context,
            sampler_options=self._sampler_options,
            logpdf_options=self._logpdf_options,
            name=self._name,
        )

    def __repr__(self) -> str:
        return f"{self._name}(event_spec={self._event_spec!r})"
