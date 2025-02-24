from functools import update_wrapper
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import core
from jax._src import ad_util
from jax._src import linear_util as lu
from jax._src.api_util import argnums_partial, flatten_fun_nokwargs
from jax._src.core import shaped_abstractify
from jax._src.util import cache, safe_map, safe_zip
from jax.extend.core import ClosedJaxpr, Primitive
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_flatten, tree_unflatten

# Use safe_map and safe_zip.
map = safe_map
zip = safe_zip

jax.numpy.set_printoptions(precision=3, suppress=True)

from probjax.core.custom_primitives.custom_inverse import custom_inverse


@custom_inverse
def f(x):
    return x**2


def _f_inv(x):
    return jnp.sqrt(x)


f.definv(_f_inv)

print(f(2.0))  # 4.0
print(jax.grad(f)(3.0))  # 4.0
print(jax.vmap(f)(jnp.array([2.0, 3.0])))  # [4.0, 9.0]
print(jax.make_jaxpr(f)(2.0))

from probjax.core import inverse

print(inverse(f)(4.0))  # 2.0
