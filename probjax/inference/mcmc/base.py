import inspect
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from blackjax.base import State
from jax.flatten_util import ravel_pytree

from probjax.inference.base import Kernel, MCMCResult
from probjax.utils.jaxutils import API
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey


class Params(NamedTuple):
    """BlackJAX only distinguishes between State and Info, but we will add Params for
    convenience.

    This should contain all the parameters that can be optimized in the kernel.
    """

    pass


MarkovKernel = Kernel


class MarkovKernelAPI(metaclass=API):
    #: Optional factory returning this kernel's recommended adaptation
    #: policy (an Adaptor or a Warmup). Attached to built kernels;
    #: wrappers (e.g. pseudo-marginal) inherit it.
    default_adaptation_fn: Optional[Callable[..., object]] = None

    @staticmethod
    def init(position, rng_key: Optional[RngKey] = None, **kwargs) -> State:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def init_params(state, *args, **kwargs) -> Params:
        raise NotImplementedError("init_params method must be implemented")

    @staticmethod
    def build_step(*args, **kwargs) -> Callable:
        raise NotImplementedError("build_kernel method must be implemented")

    def __new__(cls, logdensity_fn: Callable, **kwargs) -> MarkovKernel:
        def init_fn(key, position=None, **init_kwargs):
            # Accept the old position-first form while internal SMC composition
            # is migrated to constructed kernel values.
            if position is None:
                position, key = key, init_kwargs.pop("rng_key", None)
            return cls.init(
                position, logdensity_fn=logdensity_fn, rng_key=key, **init_kwargs
            )

        raw_step = cls.build_step(logdensity_fn, **kwargs)

        def step(key, state, params, *args):
            return raw_step(key, state, params)

        return MarkovKernel(init_fn, step, cls.init_params, cls.default_adaptation_fn)


def make_kernel_api(
    *,
    name: str,
    init_fn: Callable,
    init_params_fn: Callable,
    build_step_fn: Callable,
    default_adaptation_fn: Optional[Callable[..., object]] = None,
):
    """Create a MarkovKernelAPI subclass with minimal boilerplate."""
    sig = inspect.signature(init_fn)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    accepts_rng_key = "rng_key" in sig.parameters or accepts_kwargs

    def _init_with_rng(*args, rng_key=None, **kwargs):
        if accepts_rng_key:
            return init_fn(*args, rng_key=rng_key, **kwargs)
        return init_fn(*args, **kwargs)

    attrs = {
        "init": staticmethod(_init_with_rng),
        "init_params": staticmethod(init_params_fn),
        "build_step": staticmethod(build_step_fn),
    }
    if default_adaptation_fn is not None:
        # staticmethod: the factory must not bind the API class as self.
        attrs["default_adaptation_fn"] = staticmethod(default_adaptation_fn)
    return type(name, (MarkovKernelAPI,), attrs)


def make_step_from_kernel(
    logdensity_fn: Callable,
    kernel_builder: Callable,
    *,
    call_defaults: Optional[dict] = None,
) -> Callable:
    """Build a step function from a BlackJAX-style kernel builder."""
    kernel = kernel_builder()

    def step(key: RngKey, state: State, params: Params, *args):
        params_kwargs = params._asdict() if hasattr(params, "_asdict") else {}
        call_kwargs = {}
        if call_defaults:
            call_kwargs.update(call_defaults)
        call_kwargs.update(params_kwargs)
        return kernel(key, state, logdensity_fn, **call_kwargs)

    return step


def _select(value, fields):
    return {field: getattr(value, field) for field in fields}


def init_hmc_like_params(state: PyTree, inverse_mass_matrix: Optional[Array]) -> Array:
    """Default inverse mass matrix: identity of matching dimension."""
    position = state.position if hasattr(state, "position") else state
    flat_position, _ = ravel_pytree(position)
    dim = flat_position.shape[0]
    if inverse_mass_matrix is None:
        return jnp.ones((dim,))
    if inverse_mass_matrix.shape[0] != dim:
        raise ValueError(
            "The dimension of the inverse mass matrix must match the dimension"
            " of the position."
        )
    return inverse_mass_matrix


def scale_step_size_by_grad(
    state: PyTree, step_size: float, inverse_mass_matrix: Optional[Array] = None
) -> ArrayLike:
    """Scale ``step_size`` by the gradient norm so first steps stay sane."""
    grad = getattr(state, "logdensity_grad", None)
    if grad is None:
        return step_size
    flat_grad, _ = ravel_pytree(grad)
    if inverse_mass_matrix is not None:
        inv = inverse_mass_matrix
        if inv.ndim == 0:
            grad_norm = jnp.sqrt(inv) * jnp.linalg.norm(flat_grad)
        elif inv.ndim == 1:
            grad_norm = jnp.sqrt(jnp.sum(inv * flat_grad**2))
        elif inv.ndim == 2:
            grad_norm = jnp.sqrt(flat_grad @ (inv @ flat_grad))
        else:
            raise ValueError("Invalid inverse_mass_matrix shape")
    else:
        grad_norm = jnp.linalg.norm(flat_grad)
    return step_size / jnp.where(grad_norm == 0, 1.0, grad_norm)


def _reshape_thin_args(args, num_samples, thin):
    """Reshape leading axes of ``args`` to ``(num_samples, thin, ...)``."""
    if args is None:
        return None
    return jax.tree.map(
        lambda value: value.reshape((num_samples, thin) + value.shape[1:]),
        args,
    )


def _mcmc_collect_body(kernel, params, step_args, collect_info, stats_fn=None):
    """Two-level scan body shared by silent and verbose ``sample``."""

    def transition(state, inner_xs):
        if step_args is None:
            step_key, current_args = inner_xs, ()
        else:
            step_key, current_args = inner_xs
        state, info = kernel(step_key, state, params, *current_args)
        return state, (info, _select(info, collect_info))

    def collect_one(state, xs):
        if step_args is None:
            sample_keys, sample_args = xs, None
        else:
            sample_keys, sample_args = xs
        inner_xs = sample_keys if sample_args is None else (sample_keys, sample_args)

        state, (step_info, selected_info) = jax.lax.scan(transition, state, inner_xs)
        diagnostics = jax.tree.map(lambda value: value[-1], selected_info)
        if stats_fn is None:
            return state, (state.position, diagnostics)
        last_info = jax.tree.map(lambda value: value[-1], step_info)
        return state, (stats_fn(state, last_info), state.position, diagnostics)

    return collect_one


@partial(
    jax.jit,
    static_argnames=(
        "kernel",
        "num_samples",
        "thin",
        "collect_info",
    ),
)
def sample_kernel(
    key: RngKey,
    kernel: MarkovKernel,
    state: State,
    num_samples: int,
    params: Params,
    args: Optional[Tuple] = None,
    *,
    thin: int = 1,
    collect_info: Tuple[str, ...] = (),
) -> MCMCResult:
    """Collect positions from a compiled kernel scan."""
    keys = jax.random.split(key, (num_samples, thin))
    step_args = _reshape_thin_args(args, num_samples, thin)
    xs = keys if step_args is None else (keys, step_args)

    collect_one = _mcmc_collect_body(kernel, params, step_args, collect_info)

    state, (samples, info) = jax.lax.scan(collect_one, state, xs)
    return MCMCResult(state, samples, info if collect_info else None)
