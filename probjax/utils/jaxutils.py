import math
from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.experimental
import jax.numpy as jnp
from jax._src import linear_util as lu
from jax._src.flatten_util import ravel_pytree
from jax.core import eval_jaxpr
from jax.interpreters.partial_eval import partial_eval_jaxpr_nounits
from jaxtyping import Array, PyTree


class API(type):
    """API class for algorithms"""

    def __str__(self):
        return self.__doc__

    def __repr__(self):
        text = self.__doc__
        return text


class WithProgressBarAPI:
    _print_rate: int = 1000
    _print_length: int = 50
    _running_stats = ()

    @staticmethod
    def _print_progress(cls, iteration, total, stats):
        print_rate = total // cls._print_rate + 1
        percent = 100 * ((iteration + print_rate) / float(total))

        percent = min(percent, 100)

        percent = ("{0:." + str(2) + "f}").format(percent)

        filled_length = int(cls._print_length * iteration // total + 1)
        bar = '█' * filled_length + '-' * (cls._print_length - filled_length)

        progress_bar = f'\rProgress: |{bar}| {percent}%'

        progress_bar += " ".join(
            f" {name}: {stat:.2f}" for name, stat in zip(cls._running_stats, stats)
        )

        print(progress_bar, end="\r")

        if iteration == total:
            print()


@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


@lu.transformation
def ravel_arg_(unravel, index, *args):
    flat_arg_i = args[index]
    arg_i = unravel(flat_arg_i)
    args = args[:index] + (arg_i,) + args[index + 1 :]
    ans = yield args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


@lu.transformation
def ravel_args_(unravel, args_flat):
    args = unravel(args_flat)
    ans = yield args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


@lu.transformation_with_aux
def flatten_args_(in_tree, *flat_args):
    args = jax.tree_util.tree_unflatten(in_tree, flat_args)
    ans = yield (args,), {}
    ans_flat = jax.tree_util.tree_flatten(ans)
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


def flatten_fun(fun: Callable, in_tree: PyTree) -> Callable:
    """Flattens the input arguments of a function. Meaning than all abstract inputs are
    flattened into a list of arrays.

    Args:
        fun (Callable): Function to be flattened
        in_tree (PyTree): In tree of the functions input arguments

    Returns:
        Tuple[Callable]: The flattened function
    """

    def fun_new(*args):
        f_flat, out_tree = flatten_args_(lu.wrap_init(fun), in_tree)
        out = f_flat.call_wrapped(*args)
        return jax.tree_util.tree_unflatten(out_tree(), out)

    return fun_new


def ravel_args(in_vals: PyTree) -> Tuple[Array, Callable]:
    """_summary_

    Args:
        in_vals (PyTree): _description_

    Returns:
        Tuple[Array, Callable]: _description_
    """
    flat_vals, unflatten = ravel_pytree(in_vals)
    return flat_vals, unflatten


def ravel_fun(fun: Callable, unravel) -> Callable:
    return ravel_args_(lu.wrap_init(fun), unravel).call_wrapped


def ravel_arg_fun(fun: Callable, unravel, index: int) -> Callable:
    return ravel_arg_(lu.wrap_init(fun), unravel, index).call_wrapped


def ravel_first_arg_fun(fun: Callable, unravel) -> Callable:
    return ravel_first_arg_(lu.wrap_init(fun), unravel).call_wrapped


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

    sub_xs = jax.tree_map(nested_reshape, xs)
    return _inner_nested_scan(f, init, sub_xs, nested_lengths, _scan_fn, checkpoint_fn)


def _inner_nested_scan(f, init, xs, lengths, scan_fn, checkpoint_fn):
    """Recursively applied scan function."""
    if len(lengths) == 1:
        return scan_fn(f, init, xs, lengths[0])

    @checkpoint_fn
    def sub_scans(carry, xs):
        return _inner_nested_scan(f, carry, xs, lengths[1:], scan_fn, checkpoint_fn)

    carry, out = scan_fn(sub_scans, init, xs, lengths[0])
    stacked_out = jax.tree_map(jnp.concatenate, out)
    return carry, stacked_out


def print_scan(
    f,
    init,
    init_stats,
    xs: Optional[Array] = None,
    length: Optional[int] = None,
    *,
    unroll: int = 1,
    update_stats: Callable = None,
    print_rate: Optional[int] = None,
    print_fn: Callable = print,
):
    """A version of lax.scan that supports printing the progress of the scan.

    The interface of `print_scan` exactly matches lax.scan, except for
    the print_rate argument.

    """
    if length is None:
        length = xs.shape[0]

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
            lambda: jax.experimental.io_callback(print_fn, None, i, length, stats),
            lambda: None,
        )
        i += 1
        return (i,) + stats + carry, y

    carry, y = jax.lax.scan(
        scan_fn, (0,) + init_stats + init, xs, length, unroll=unroll
    )
    carry = carry[1 + num_stats :]
    return carry, y
