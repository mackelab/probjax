from typing import Callable, Sequence
import jax
import jax.numpy as jnp

from jax.typing import ArrayLike


def periodic_boundary_conditions(x: ArrayLike):
    """
    Periodic boundary conditions.

    Args:
        x: ArrayLike, the spatial grid
    """
    # For an x of N dimensions, we have to apply to the first and last elements of x
    # in each dimension

    ndim = x.ndim
    for i in range(ndim):
        numpy_index = [slice(None)] * ndim
        numpy_index[i] = 0
        x = x.at[tuple(numpy_index)].set(
            x[tuple(numpy_index)] + x[tuple(numpy_index)][-1]
        )
        numpy_index[i] = -1
        x = x.at[tuple(numpy_index)].set(
            x[tuple(numpy_index)] + x[tuple(numpy_index)][0]
        )
    return x


def dirichlet_boundary_conditions(x: ArrayLike, value: float = 0.0):
    """
    Dirichlet boundary conditions.

    Args:
        x: ArrayLike, the spatial grid
    """
    # For an x of N dimensions, we have to apply to the first and last elements of x
    # in each dimension

    ndim = x.ndim
    for i in range(ndim):
        numpy_index = [slice(None)] * ndim
        numpy_index[i] = 0
        x = x.at[tuple(numpy_index)].set(value)
        numpy_index[i] = -1
        x = x.at[tuple(numpy_index)].set(value)
    return x


def neumann_boundary_conditions(x: ArrayLike, value: float = 0.0):
    """
    Neumann boundary conditions.

    Args:
        x: ArrayLike, the spatial grid
    """
    # For an x of N dimensions, we have to apply to the first and last elements of x
    # in each dimension

    ndim = x.ndim
    for i in range(ndim):
        numpy_index = [slice(None)] * ndim
        numpy_index[i] = 0
        x = x.at[tuple(numpy_index)].set(x[tuple(numpy_index)][1] + value)
        numpy_index[i] = -1
        x = x.at[tuple(numpy_index)].set(x[tuple(numpy_index)][-2] + value)
    return x


def uniform_grid(
    start: float | Sequence[float] | ArrayLike,
    stop: float | Sequence[float] | ArrayLike,
    num_points: int | Sequence[int] | ArrayLike,
):
    """
    Creates a uniform ND grid

    Args:
        start: float | Sequence[float] | ArrayLike, the start of the grid
        stop: float | Sequence[float] | ArrayLike, the end of the grid
        num_points: int | Sequence[int] | ArrayLike, the number of points in the grid
    """
    if isinstance(start, (int, float)):
        start = jnp.array([start])
    if isinstance(stop, (int, float)):
        stop = jnp.array([stop])
    if isinstance(num_points, int):
        num_points = jnp.array([num_points])

    ndim = max(start.ndim, stop.ndim, num_points.ndim)
    if start.ndim == 0:
        start = jnp.repeat(start, ndim)
    if stop.ndim == 0:
        stop = jnp.repeat(stop, ndim)
    if num_points.ndim == 0:
        num_points = jnp.repeat(num_points, ndim)

    grid = []
    for s, e, n in zip(start, stop, num_points):
        grid.append(jnp.linspace(s, e, n))
    return tuple(grid)


def pdeint(
    f: Callable,
    ts: ArrayLike,
    x0: ArrayLike,
    spatial_grid: Sequence[ArrayLike],
    boundary_conditions: Callable,
    *args,
    **kwargs
):
    """
    Solves a PDE using the Method of Lines.

    Args:
        f: Callable, the PDE function
        ts: ArrayLike, the time points to solve the PDE at
        x0: ArrayLike, the initial condition
        spatial_grid: Sequence[ArrayLike], the spatial grid to solve the PDE on
        boundary_conditions: Callable, the boundary conditions
        *args: additional arguments to pass to the PDE function
        **kwargs: additional keyword arguments to pass to the PDE function
    """
    
    x0_flat, unflatten = jax.flatten_util.ravel_pytree(x0)
    x0_flat = jnp.array(x0_flat)
    
    def ode_fn(x, t, *args):
        x = unflatten(x)
        x = boundary_conditions(x)
        dxdt = f(x, t, *args)
        return jax.flatten_util.ravel_pytree(dxdt)[0]
    
    
    
