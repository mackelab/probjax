from jax._src import core as jax_core
from jax.tree_util import tree_leaves


def has_tracer(tree) -> bool:
    return any(isinstance(x, jax_core.Tracer) for x in tree_leaves(tree))


def ensure_hashable(x, where: str):
    try:
        hash(x)
    except TypeError as e:
        raise TypeError(f"{where} must be hashable; got {type(x)}") from e
    return x


def fail_on_tracer_constants(consts, where: str):
    if has_tracer(consts):
        raise TypeError(
            f"{where} closed over traced JAX values. "
            "Pass such data as dynamic arguments or mark them static before "
            "registering the custom primitive."
        )
