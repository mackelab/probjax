

import jax
from jax.core import CallPrimitive, new_sublevel
from jax import linear_util as lu


def call_impl(f: lu.WrappedFun, *args, **params):
  del params  # params parameterize the call primitive, not the function
  with new_sublevel():
    return f.call_wrapped(*args)

rqs_p = CallPrimitive("rational_quadratic_spline")






