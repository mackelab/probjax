from __future__ import annotations

from dataclasses import is_dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from blackjax.smc import base as bj_smc_base

from probjax.inference.base import Kernel
from probjax.utils.guards import has_attrs, params_to_dict
from probjax.utils.jaxutils import (
    KernelAPI,
    as_configured,
    filter_kwargs,
)
from probjax.utils.typing import Array, ArrayLike, PyTree

SMCState = bj_smc_base.SMCState


SMCKernel = Kernel


class SMCKernelAPI(KernelAPI):
    """Base class for SMC kernels.

    Subclass it and set ``init``, ``init_params`` and ``build_step`` as static
    methods. Document the kernel in the subclass docstring:
    ``cls(logprior_fn, loglikelihood_fn, **options)`` forwards ``options`` to
    ``build_step`` (and the ones ``init`` accepts to ``init``) and returns an
    ``SMCKernel``. Use ``cls.configure(**options)`` to bind the options before
    the model's functions are known.
    """

    _kernel_type = SMCKernel

    @staticmethod
    def init(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def init_params(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("init_params method must be implemented")

    @staticmethod
    def build_step(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("build_step method must be implemented")

    def __new__(cls, logprior_fn: Callable, loglikelihood_fn: Callable, **kwargs: Any):
        init = partial(
            cls.init,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            **filter_kwargs(cls.init, kwargs, allow_kwargs=False),
        )
        init_params = partial(
            cls.init_params, logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_fn
        )
        step = cls.build_step(logprior_fn, loglikelihood_fn, **kwargs)
        return cls.kernel_type(init, step, init_params)


def _ensure_param_batch(params: Any, *, shared: bool) -> dict[str, Array]:
    params = params_to_dict(params)
    batched = {}
    for k, v in params.items():
        arr = jnp.asarray(v)
        if shared:
            if arr.ndim == 0 or arr.shape[0] != 1:
                arr = arr[None, ...]
            # Ensure shared params are at least 2-D so that
            # BlackJAX's unshared_parameters_and_step_fn (which does
            # v[0, ...] on shared params) preserves the inner array.
            if arr.ndim == 1:
                arr = arr[None, ...]
        else:
            if arr.ndim == 0:
                arr = arr[None]
        batched[k] = arr
    return batched


class _ParamsProxy:
    def __init__(self, params: dict[str, ArrayLike]):
        self._params = params
        for k, v in params.items():
            setattr(self, k, v)

    def _asdict(self) -> dict[str, ArrayLike]:
        return self._params


def _wrap_params(params: dict[str, ArrayLike]) -> Any:
    if params is None:
        return _ParamsProxy({})
    if has_attrs(params, "_asdict"):
        return params
    if is_dataclass(params):
        return _ParamsProxy(params_to_dict(params))
    return _ParamsProxy(params)


def make_mcmc_adapter(
    mcmc_kernel: Any, **kernel_kwargs: Any
) -> tuple[Callable, Callable]:
    """Adapt a probjax MCMC API into BlackJAX SMC mcmc_init_fn/mcmc_step_fn.

    ``mcmc_kernel`` is a MCMC API class (``hmc``) or a configured one
    (``hmc.configure(num_integration_steps=10)``). Options passed here are
    merged into the configured ones, which win.
    """
    configured = as_configured(mcmc_kernel, **kernel_kwargs)
    if not has_attrs(configured.api, "build_step", "init"):
        raise TypeError("mcmc_kernel must be a probjax MCMC API (not an instance).")

    def mcmc_init_fn(
        position: Any,
        logdensity_fn: Callable,
        rng = None,
    ) -> Any:
        return configured.init(position, logdensity_fn, rng=rng)

    def mcmc_step_fn(
        rng, state: Any, logdensity_fn: Callable, **step_parameters: Any
    ) -> Any:
        step = configured.build_step(logdensity_fn)
        return step(rng, state, _wrap_params(step_parameters))

    return mcmc_init_fn, mcmc_step_fn


def init_mcmc_params_from_logdensity(
    particles: PyTree,
    logdensity_fn: Callable,
    mcmc_kernel: Any,
    rng = None,
    mcmc_kernel_kwargs: dict[str, Any] | None = None,
    allow_kwargs: bool = False,
    **mcmc_param_kwargs: Any,
) -> dict[str, Array]:
    """Initialize MCMC params from the first particle under ``logdensity_fn``.

    Shared body of the ``init_params`` functions in ``adaptive``,
    ``persistent_smc``, ``adaptive_persistent_smc`` and ``path_smc``: take the
    first particle, init the MCMC kernel on it, filter ``mcmc_param_kwargs``
    down to what ``init_params`` accepts, and batch the result as shared
    parameters.
    """
    mcmc_kernel_kwargs = mcmc_kernel_kwargs or {}
    mcmc_init_fn, _ = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    particle0 = jax.tree_util.tree_map(lambda x: x[0], particles)
    mcmc_state = mcmc_init_fn(particle0, logdensity_fn, rng=rng)
    configured = as_configured(mcmc_kernel)
    init_param_kwargs = filter_kwargs(
        configured.api.init_params, mcmc_param_kwargs, allow_kwargs=allow_kwargs
    )
    params = configured.init_params(mcmc_state, **init_param_kwargs)
    return _ensure_param_batch(params_to_dict(params), shared=True)


def init_tempered_mcmc_params(
    particles: PyTree,
    *,
    logprior_fn: Callable | None,
    loglikelihood_fn: Callable | None,
    mcmc_kernel: Any,
    rng = None,
    lmbda: float = 0.0,
    mcmc_kernel_kwargs: dict[str, Any] | None = None,
    kind: str = "persistent SMC",
    **mcmc_param_kwargs: Any,
) -> dict[str, Array]:
    """Initialize MCMC params under the tempered log-posterior.

    The tempered density is ``logprior_fn(position) + lmbda *
    loglikelihood_fn(position)``. Shared body of the ``init_params``
    functions of ``persistent_smc`` and ``adaptive_persistent_smc``.
    """
    if logprior_fn is None or loglikelihood_fn is None:
        raise ValueError(f"{kind} requires both logprior_fn and loglikelihood_fn.")

    def tempered_logposterior_fn(position: Any) -> Array:
        return logprior_fn(position) + lmbda * loglikelihood_fn(position)

    return init_mcmc_params_from_logdensity(
        particles,
        tempered_logposterior_fn,
        mcmc_kernel,
        rng=rng,
        mcmc_kernel_kwargs=mcmc_kernel_kwargs,
        **mcmc_param_kwargs,
    )
