import concurrent.futures
import importlib
import inspect
import math
import sys
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import jax
import jax.experimental
import jax.numpy as jnp
from jax._src import linear_util as lu
from jax._src.api_util import debug_info
from jax._src.core import eval_jaxpr
from jax._src.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from probjax.utils.typing import ArrayLike


def _resolve_partial_eval_jaxpr_nounits() -> Callable:
    """Find the partial-evaluation helper; its name moved across JAX releases."""
    for module_name in (
        "jax.interpreters.partial_eval",
        "jax._src.interpreters.partial_eval",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        nounits = getattr(module, "partial_eval_jaxpr_nounits", None)
        if nounits is not None:
            return nounits
        legacy = getattr(module, "partial_eval_jaxpr", None)
        if legacy is not None:
            return lambda jaxpr, unknowns, instantiate: legacy(
                jaxpr, unknowns, instantiate
            )
    raise ImportError("JAX exposes no usable partial-evaluation helper.")


partial_eval_jaxpr_nounits = _resolve_partial_eval_jaxpr_nounits()


_progress_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="progress"
)


class API(type):
    """API class for algorithms.

    An API class is a namespace, not an instance: it holds the functions that
    build a kernel (``init``, ``init_params``, ``build_step``) and is turned
    into a kernel by calling it with the model's functions.
    """

    def __str__(self):
        return self.__doc__ or ""

    def __repr__(self):
        return self.__doc__ or self.__name__

    def configure(cls, **options: Any) -> "Configured":
        """Bind options now; the model's functions can be supplied later.

        ``hmc.configure(num_integration_steps=10)`` returns a
        :class:`Configured`; calling that with the log-density builds the
        kernel. The result is hashable and compares by value, so it can be a
        static ``jit`` argument without retracing.
        """
        return Configured(cls, tuple(sorted(options.items())))


@dataclass(frozen=True)
class Configured:
    """An API class with its options bound, waiting for the model.

    Produced by :meth:`API.configure`, and by calling an API class with
    options only where the model's function is unambiguous::

        kernel = hmc.configure(num_integration_steps=10)(logdensity)

    The bound options are routed to whichever of ``init``, ``init_params``
    and ``build_step`` accepts them, so one set of options can describe a
    kernel's whole configuration.
    """

    api: type
    options: Tuple[Tuple[str, Any], ...] = ()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Build the kernel, with the bound options that apply to building."""
        options = self._routed("build_step", allow_kwargs=True)
        options.update(kwargs)
        return self.api(*args, **options)

    @property
    def info(self) -> Dict[str, Any]:
        """The API's metadata (a solver's order and adaptivity, say)."""
        return getattr(self.api, "info", {})

    @property
    def default_adaptation_fn(self) -> Any:
        """The API's recommended adaptation policy, if it has one."""
        return getattr(self.api, "default_adaptation_fn", None)

    def _routed(self, name: str, *, allow_kwargs: bool) -> Dict[str, Any]:
        """The bound options that ``api.<name>`` accepts."""
        return filter_kwargs(
            getattr(self.api, name), dict(self.options), allow_kwargs=allow_kwargs
        )

    def init(self, *args: Any, **kwargs: Any) -> Any:
        """``api.init`` with the options it accepts."""
        options = self._routed("init", allow_kwargs=False)
        options.update(kwargs)
        return self.api.init(*args, **options)

    def init_params(self, *args: Any, **kwargs: Any) -> Any:
        """``api.init_params`` with the options it accepts."""
        options = self._routed("init_params", allow_kwargs=False)
        options.update(kwargs)
        return self.api.init_params(*args, **options)

    def build_step(self, *args: Any, **kwargs: Any) -> Any:
        """``api.build_step`` with the options it accepts."""
        options = self._routed("build_step", allow_kwargs=True)
        options.update(kwargs)
        return self.api.build_step(*args, **options)


def as_configured(api_or_configured: Any, **options: Any) -> Configured:
    """Normalize an API class or a :class:`Configured` into a ``Configured``.

    Extra options are merged in; the ones already bound to a ``Configured``
    win, since they were the more explicit binding.
    """
    if isinstance(api_or_configured, Configured):
        base = api_or_configured
    else:
        base = Configured(api_or_configured, ())
    if not options:
        return base
    merged = dict(base.options) | options
    return Configured(base.api, tuple(sorted(merged.items())))


def filter_kwargs(
    fn: Callable,
    kwargs: Mapping[str, Any],
    *,
    allow_kwargs: bool = True,
    exclude_positional_only: bool = False,
) -> Dict[str, Any]:
    """The subset of ``kwargs`` that ``fn`` accepts.

    A function that takes ``**kwargs`` takes everything, unless
    ``allow_kwargs`` is false -- then only the options it names are kept.

    ``exclude_positional_only`` drops parameters that can only be passed
    positionally, which cannot be forwarded as a keyword.
    """
    sig = inspect.signature(fn)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    if accepts_kwargs and allow_kwargs:
        return dict(kwargs)
    return {
        k: v
        for k, v in kwargs.items()
        if k in sig.parameters
        and not (
            exclude_positional_only
            and sig.parameters[k].kind == inspect.Parameter.POSITIONAL_ONLY
        )
    }


class KernelAPI(metaclass=API):
    """Base class for algorithms that build a callable kernel.

    A subclass sets ``_kernel_type`` -- the container describing a built
    kernel -- and implements ``init`` and ``build_step``. The built kernel
    gets a per-algorithm type, so ``repr(kernel)`` and ``help(kernel)`` show
    the algorithm rather than the generic container.
    """

    #: The container describing a built kernel; set by each subclass.
    _kernel_type: ClassVar[Optional[type]] = None
    #: The per-algorithm kernel type, created for each subclass. Typed
    #: loosely because it is a subclass of a container the subclass chooses.
    kernel_type: ClassVar[Any] = None

    @staticmethod
    def init(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def init_params(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("init_params method must be implemented")

    @staticmethod
    def build_step(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("build_step method must be implemented")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kernel_type = cls._kernel_type
        if kernel_type is None:
            return
        # Built kernels are instances of a per-algorithm type, so
        # help(kernel) and repr(kernel) show this algorithm. The subclass is
        # display-only: dispatch on the base type, and do not pickle kernels
        # (they close over jit-compiled steps).
        cls.kernel_type = type(
            cls.__name__,
            (cast(type, kernel_type),),
            {"__doc__": cls.__doc__, "__module__": cls.__module__, "__slots__": ()},
        )


_MISSING = object()


def _resolve_attr(obj, dotted: str):
    """Walk a dotted attribute path, returning ``_MISSING`` if any link is absent."""
    if obj is None:
        return _MISSING
    for part in dotted.split("."):
        obj = getattr(obj, part, _MISSING)
        if obj is _MISSING or obj is None:
            return _MISSING
    return obj


class WithProgressBarAPI:
    """Mixin providing a JIT-friendly progress bar driven by tracked diagnostics.

    Subclasses set ``_default_tracked_stats`` to a tuple of names. Each name is
    looked up — at trace time — first on the per-step **state**, then on
    **info**, then in the class-level ``_computed_stats`` registry, falling back
    to ``NaN``. Subclasses with extra inspection sources (e.g. SMC's nested
    ``info.update_info``) override :meth:`_stat_objects`. Names may be dotted
    paths (``"update_info.acceptance_rate"``); array values are reduced to a
    scalar via mean.
    """

    _print_rate: int = 1000
    _print_length: int = 50
    _ema_gamma: float = 0.9
    _default_tracked_stats: Tuple[str, ...] = ()
    _computed_stats: Dict[str, Callable] = {}
    #: Names of the diagnostics to display; set per instance by the runners.
    tracked_stats: Tuple[str, ...] = ()

    @staticmethod
    def _write_progress(owner, iteration, total, stats, stat_names=()):
        try:
            i, t = int(iteration), int(total)
        except Exception:
            return

        print_len = owner._print_length
        pct = min(100.0, 100 * (i + 1) / t)
        filled = int(print_len * (i + 1) // t)
        bar = "█" * filled + "-" * (print_len - filled)
        s = f"Progress: |{bar}| {pct:.2f}%"
        for name, val in zip(stat_names, stats, strict=False):
            s += f" {name}: {float(val):.2f}"

        _progress_executor.submit(sys.stdout.write, f"\r{s}")

    # ------------------------------------------------------------------
    # Stat extraction — resolved at JAX trace time, no probe call.
    # ------------------------------------------------------------------

    def _stat_objects(self, state: Any, info: Any) -> tuple[Any, ...]:
        """Return the tuple of objects to search, in order, for tracked stats."""
        return (state, info)

    def _resolve_stat(self, name: str, state, info):
        """Resolve a single tracked-stat name to a scalar float32 value."""
        for obj in self._stat_objects(state, info):
            val = _resolve_attr(obj, name)
            if val is not _MISSING:
                arr = jnp.asarray(val, dtype=jnp.float32)
                return jnp.mean(arr) if arr.ndim > 0 else arr
        fn = type(self)._computed_stats.get(name)
        if fn is not None:
            val = fn(state, info)
            if val is not None:
                arr = jnp.asarray(val, dtype=jnp.float32)
                return jnp.mean(arr) if arr.ndim > 0 else arr
        return jnp.float32(jnp.nan)

    def _extract_stats(self, state, info) -> Tuple:
        return tuple(self._resolve_stat(n, state, info) for n in self.tracked_stats)

    def _make_verbose_fns(
        self,
        num_steps: int,
        stats_fn: Optional[Callable] = None,
    ):
        """Build (update_stats, print_fn, init_stats, print_rate) for ``print_scan``.

        ``stats_fn(carry, y) -> tuple`` extracts the per-step stats tuple from
        the scan output. Defaults to ``lambda _, y: y`` (the scan's ``y``
        already is the stats tuple, as in :class:`MCMC`/:class:`SMC`).
        """
        gamma = self._ema_gamma
        names = self.tracked_stats
        n = len(names)
        if stats_fn is None:
            stats_fn = lambda _carry, y: y

        def update_stats(stats, carry, y):
            step_stats = stats_fn(carry, y)
            return tuple(
                gamma * stats[i] + (1 - gamma) * step_stats[i] for i in range(n)
            )

        def print_fn(i, total, stats):
            type(self)._write_progress(type(self), i, total, stats, names)

        init_stats = tuple(0.0 for _ in names)
        print_rate = num_steps // self._print_rate + 1
        return update_stats, print_fn, init_stats, print_rate


def where(condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> Array:
    """``jnp.where`` with a precise return type.

    JAX types ``jnp.where`` as ``Array | tuple[Array, ...]`` because it also
    has a pytree form; the numerical helpers here always mean the array
    form, and the union otherwise poisons every downstream expression.
    """
    return jnp.where(condition, x, y)


def prepend_to_trace(initial: PyTree, trace: PyTree) -> PyTree:
    """Prepend ``initial`` to a per-step ``trace`` along the leading axis.

    ``initial`` and ``trace`` must have the same tree structure, and each
    corresponding leaf must have the same shape apart from the leading axis.
    """
    return jax.tree_util.tree_map(
        lambda init_leaf, trace_leaf: jnp.concatenate(
            [jnp.asarray(init_leaf)[None], jnp.asarray(trace_leaf)], axis=0
        ),
        initial,
        trace,
    )


@lu.transformation
def ravel_arg_(unravel, index, *args):
    flat_arg_i = args[index]
    arg_i = unravel(flat_arg_i)
    args = args[:index] + (arg_i,) + args[index + 1 :]
    ans = yield args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


def precompute(func: Callable, arg_list: list, known_argnums: list) -> Callable:
    """Precomputes all computations that can be done with all known arguments.

    Args:
        func (Callable): Function to be precomputed
        arg_list (list): List of arguments to be precomputed (with dummies for unknowns)
        known_argnums (list): List of indices of known arguments

    Returns:
        Callable: Function that inputs all unknown arguments and returns the result of
        the function
    """
    jaxpr = jax.make_jaxpr(func)(*arg_list)
    unknowns = [k not in known_argnums for k in range(len(arg_list))]
    instantiate = False

    (known_jaxpr, unknown_jaxpr, _, _) = partial_eval_jaxpr_nounits(
        jaxpr, unknowns, instantiate
    )

    known_values = [arg_k for (k, arg_k) in enumerate(arg_list) if k in known_argnums]
    precomputed_values = eval_jaxpr(
        known_jaxpr.jaxpr, known_jaxpr.consts, *known_values
    )

    def inner(*args):
        values = eval_jaxpr(
            unknown_jaxpr.jaxpr,
            unknown_jaxpr.consts,
            *precomputed_values,
            *args,
            propagate_source_info=False,
        )
        return values if len(values) > 1 else values[0]

    return inner


def ravel_args(in_vals: PyTree) -> Tuple[Array, Callable]:
    """_summary_

    Args:
        in_vals (PyTree): _description_

    Returns:
        Tuple[Array, Callable]: _description_
    """
    flat_vals, unflatten = ravel_pytree(in_vals)
    return flat_vals, unflatten


def ravel_arg_fun(fun: Callable, unravel, index: int) -> Callable:
    info = debug_info("Raveled arg function", fun, (), {})
    return ravel_arg_(lu.wrap_init(fun, debug_info=info), unravel, index).call_wrapped


def nested_checkpoint_scan(
    f,
    init,
    xs,
    length: Optional[int] = None,
    *,
    nested_lengths: Sequence[int],
    scan_fn: Callable = jax.lax.scan,
    checkpoint_fn: Callable = jax.checkpoint,  # Corrected type hint
    unroll: int = 1,
):
    """A version of lax.scan that supports recursive gradient checkpointing.

    Code taken from: https://github.com/google/jax/issues/2139

    The interface of `nested_checkpoint_scan` exactly matches lax.scan, except for
    the required `nested_lengths` argument.

    The key feature of `nested_checkpoint_scan` is that gradient calculations
    require O(max(nested_lengths)) memory, vs O(prod(nested_lengths)) for unnested
    scans, which it achieves by re-evaluating the forward pass
    `len(nested_lengths) - 1` times.

    `nested_checkpoint_scan` reduces to `lax.scan` when `nested_lengths` has a
    single element.

    Args:
        f: function to scan over.
        init: initial value.
        xs: scanned over values.
        length: leading length of all dimensions
        nested_lengths: required list of lengths to scan over for each level of
            checkpointing. The product of nested_lengths must match length (if
            provided) and the size of the leading axis for all arrays in ``xs``.
        scan_fn: function matching the API of lax.scan
        checkpoint_fn: function matching the API of jax.checkpoint.
    """

    if xs is not None:
        length = xs.shape[0]
    if length is not None and length != math.prod(nested_lengths):
        raise ValueError(f"inconsistent {length=} and {nested_lengths=}")

    def nested_reshape(x):
        x = jnp.asarray(x)
        new_shape = tuple(nested_lengths) + x.shape[1:]
        return x.reshape(new_shape)

    _scan_fn = partial(scan_fn, unroll=unroll)

    sub_xs = jax.tree_util.tree_map(nested_reshape, xs)
    return _inner_nested_scan(f, init, sub_xs, nested_lengths, _scan_fn, checkpoint_fn)


def _inner_nested_scan(f, init, xs, lengths, scan_fn, checkpoint_fn):
    """Recursively applied scan function."""
    if len(lengths) == 1:
        return scan_fn(f, init, xs, lengths[0])

    @checkpoint_fn
    def sub_scans(carry, xs):
        return _inner_nested_scan(f, carry, xs, lengths[1:], scan_fn, checkpoint_fn)

    carry, out = scan_fn(sub_scans, init, xs, lengths[0])
    stacked_out = jax.tree_util.tree_map(jnp.concatenate, out)
    return carry, stacked_out


def print_scan(
    f,
    init,
    init_stats,
    xs: Any = None,
    length: Optional[int] = None,
    *,
    unroll: int = 1,
    update_stats: Callable,
    print_rate: Optional[int] = None,
    print_fn: Callable = print,
):
    """A version of lax.scan that supports printing the progress of the scan.

    The interface of `print_scan` exactly matches lax.scan, except for
    the print_rate argument.

    """
    if length is None:
        # xs may be a pytree (e.g. tuple of arrays); grab the first leaf.
        leaves = jax.tree_util.tree_leaves(xs)
        length = int(leaves[0].shape[0])

    if print_rate is None:
        print_rate = length // 50 + 1

    num_stats = len(init_stats)

    def scan_fn(carry, x):
        i, *carry = carry
        stats, carry = carry[:num_stats], carry[num_stats:]
        carry, y = f(carry, x)
        stats = update_stats(stats, carry, y)

        jax.lax.cond(
            i % print_rate == 0,
            lambda: jax.debug.callback(print_fn, i, length, stats),
            lambda: None,
        )
        i += 1
        return (i,) + stats + carry, y

    carry, y = jax.lax.scan(
        scan_fn, (0,) + init_stats + init, xs, length, unroll=unroll
    )
    carry = carry[1 + num_stats :]
    return carry, y


@lru_cache(maxsize=1)
def def_partition_supports_need_replication_factors(def_partition) -> bool:
    """Whether ``custom_partitioning.def_partition`` accepts the kwarg."""
    try:
        return "need_replication_factors" in inspect.signature(def_partition).parameters
    except (TypeError, ValueError):
        return False


def def_partition_compat(def_partition, /, **kwargs):
    """Call ``def_partition`` across JAX versions.

    Older JAX builds do not accept ``need_replication_factors``.
    """
    if (
        "need_replication_factors" in kwargs
        and not def_partition_supports_need_replication_factors(def_partition)
    ):
        kwargs.pop("need_replication_factors")
    return def_partition(**kwargs)
