import inspect
from dataclasses import asdict, is_dataclass
from functools import partial
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
from blackjax.smc import base as bj_smc_base

from probjax.inference.base import Kernel
from probjax.utils.guards import has_attrs, params_to_dict
from probjax.utils.jaxutils import API
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey

SMCState = bj_smc_base.SMCState


SMCKernel = Kernel


class SMCKernelAPI(metaclass=API):
    """Base class for SMC kernels.

    Subclass it and set ``init``, ``init_params`` and ``build_step`` as static
    methods. Document the kernel in the subclass docstring:
    ``cls(logprior_fn, loglikelihood_fn, **options)`` forwards ``options`` to
    ``build_step`` (and the ones ``init`` accepts to ``init``) and returns an
    ``SMCKernel``.
    """

    @staticmethod
    def init(particles, rng_key: Optional[RngKey] = None, **kwargs) -> SMCState:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def init_params(particles, rng_key: Optional[RngKey] = None, **kwargs) -> Dict:
        raise NotImplementedError("init_params method must be implemented")

    @staticmethod
    def build_step(*args, **kwargs) -> Callable:
        raise NotImplementedError("build_step method must be implemented")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Built kernels are instances of a per-algorithm Kernel subclass, so
        # help(kernel) and repr(kernel) show this algorithm, not Kernel.
        # The subclass is display-only: dispatch on Kernel, not on it, and
        # do not pickle kernels (they close over jit-compiled steps).
        cls.kernel_type = type(
            cls.__name__,
            (SMCKernel,),
            {"__doc__": cls.__doc__, "__module__": cls.__module__, "__slots__": ()},
        )

    def __new__(cls, logprior_fn: Callable, loglikelihood_fn: Callable, **kwargs):
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


def filter_kwargs(
    fn: Callable, kwargs: Dict[str, ArrayLike], *, allow_kwargs: bool = True
) -> Dict[str, ArrayLike]:
    sig = inspect.signature(fn)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    if accepts_kwargs and allow_kwargs:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _ensure_param_batch(params, *, shared: bool) -> Dict[str, Array]:
    params = _params_to_dict(params)
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


def _params_to_dict(params) -> Dict[str, ArrayLike]:
    """Back-compat alias for :func:`params_to_dict`."""
    return params_to_dict(params)


class _ParamsProxy:
    def __init__(self, params: Dict[str, ArrayLike]):
        self._params = params
        for k, v in params.items():
            setattr(self, k, v)

    def _asdict(self) -> Dict[str, ArrayLike]:
        return self._params


def _wrap_params(params: Dict[str, ArrayLike]):
    if params is None:
        return _ParamsProxy({})
    if has_attrs(params, "_asdict"):
        return params
    if is_dataclass(params):
        return _ParamsProxy(asdict(params))
    return _ParamsProxy(params)


def make_mcmc_adapter(mcmc_kernel, **kernel_kwargs):
    """Adapt a probjax MCMC API into BlackJAX SMC mcmc_init_fn/mcmc_step_fn."""
    if not has_attrs(mcmc_kernel, "build_step", "init"):
        raise TypeError("mcmc_kernel must be a probjax MCMC API (not an instance).")

    def mcmc_init_fn(position, logdensity_fn, rng_key: Optional[RngKey] = None):
        init_kwargs = filter_kwargs(mcmc_kernel.init, kernel_kwargs, allow_kwargs=False)
        return mcmc_kernel.init(position, logdensity_fn, rng_key=rng_key, **init_kwargs)

    def mcmc_step_fn(rng_key, state, logdensity_fn, **step_parameters):
        build_kwargs = filter_kwargs(mcmc_kernel.build_step, kernel_kwargs)
        step = mcmc_kernel.build_step(logdensity_fn, **build_kwargs)
        return step(rng_key, state, _wrap_params(step_parameters))

    return mcmc_init_fn, mcmc_step_fn


def init_mcmc_params_from_logdensity(
    particles: PyTree,
    logdensity_fn: Callable,
    mcmc_kernel,
    rng_key: Optional[RngKey] = None,
    mcmc_kernel_kwargs: Optional[Dict[str, ArrayLike]] = None,
    allow_kwargs: bool = False,
    **mcmc_param_kwargs,
) -> Dict[str, Array]:
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
    mcmc_state = mcmc_init_fn(particle0, logdensity_fn, rng_key=rng_key)
    init_param_kwargs = filter_kwargs(
        mcmc_kernel.init_params, mcmc_param_kwargs, allow_kwargs=allow_kwargs
    )
    params = mcmc_kernel.init_params(mcmc_state, **init_param_kwargs)
    return _ensure_param_batch(_params_to_dict(params), shared=True)
