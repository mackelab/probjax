import jax 
import jax.numpy as jnp
import jax.random as jrandom

from jax import lax
from jax import core
from jax.tree_util import tree_leaves

from functools import partial 
from jaxtyping import Array, Float, PyTree, Int
from typing import Callable, Optional
from jax.random import PRNGKeyArray

@partial(jax.jit, static_argnums=(1, 2))
def _sdeint_on_grid(key: PRNGKeyArray, drift: Callable, diffusion: Callable, y0: Array, ts: Array, *args) -> Array:
    """ Solve a stochastic differential equation on a grid.

    Args:
        drift (Callable): Drift function.
        diffusion (Callable): Diffusion function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        *args: Other arguments.

    Returns:
        Array: Solution of the SDE.
    """
    _f = lambda t,y: drift(t, y, *args)
    _g = lambda t,y: diffusion(t,y, *args)

    dts = ts[1:] - ts[:-1]
    B0 = jnp.asarray(_g(ts[0], y0))
    if B0.ndim == 0:
        dim = 1
    else:
        dim = B0.shape[1]
    wdts = jnp.sqrt(dts)[:, None] * jrandom.normal(key, (len(dts),dim))

    def scan_fun(carry, data):
        y0, t0 = carry
        t1, dt, wdt = data
        y1 = y0 + _f(t0, y0) * dt + jnp.dot(_g(t0, y0), wdt)
        return (y1, t1), y1

    init_carry = (y0, ts[0])
    _, ys = lax.scan(scan_fun, init_carry, (ts[1:], dts, wdts))
    return jnp.concatenate((y0[None], ys))




def sdeint(key: PRNGKeyArray, drift: Callable, diffusion: Callable, y0: Array, ts: Array, *args, method: str ="euler_maruyama", dt: Optional[Float] = None, rtol:Float=1e-6, atol:Float=1e-6, mxstep:Int=jnp.inf, dtmin: Float=0., dtmax: Float=jnp.inf) -> Array:
    """ Solve a stochastic differential equation.

    Args:
        drift (Callable): Drift function.
        diffusion (Callable): Diffusion function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        method (str, optional): Methods to use. Defaults to "euler_maruyama".
        dt (Optional[Float], optional): Fixed step size (optionally infered from ts). Defaults to None.
        rtol (Float, optional): Relative tolerance. Defaults to 1e-6.
        atol (Float, optional): Absolute tolerance. Defaults to 1e-6.
        mxstep (Int, optional): Maximum number of steps used by solver. Defaults to jnp.inf.
        dtmin (Float, optional): Minimal step size used by solver. Defaults to 0..
        dtmax (Float, optional): Maximal step size used by solver. Defaults to jnp.inf.

    Raises:
        TypeError: The arguments passed not jax types.
        TypeError: The arguments passed not jax types.

    Returns:
        _type_: _description_
    """
    for arg in tree_leaves(args):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            raise TypeError(
                f"The contents of sdeint *args must be arrays or scalars, but got {arg}.")
        if not jnp.issubdtype(ts.dtype, jnp.floating):
            raise TypeError(f"t must be an array of floats, but got {t}.")
        
    return _sdeint_on_grid(key,drift, diffusion, y0, ts, *args)