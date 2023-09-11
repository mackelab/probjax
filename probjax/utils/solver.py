import jax 
import jax.numpy as jnp


def newton_raphson(f, x0, tol=1e-6, max_iter=50):
    """
    Newton-Raphson root-finding algorithm for a vector-valued function.
    
    Args:
    - f: A function that takes a vector x and returns a vector of the same shape.
    - x0: Initial guess for the root.
    - tol: Tolerance for stopping criterion (default: 1e-6).
    - max_iter: Maximum number of iterations (default: 100).
    
    Returns:
    - x: The estimated root of the function.
    """
    x = x0
    shape = x.shape
    # Flatten
    def _f(x):
        y = f(x.reshape(shape))
        return y.reshape(-1)
    f_jax = jax.jacobian(_f)

    x = x.reshape(-1)

    def scan_fn(carry, i):
        tol_reached, x = carry

        def true_fn(x):
            return x, jnp.inf

        def false_fn(x):
            y = _f(x)
            J = f_jax(x)
            delta_x = jax.scipy.linalg.solve(J, -y)
            x = x + delta_x
            return x, jnp.linalg.norm(delta_x)
        
        
        x, delta_x = jax.lax.cond(tol_reached, true_fn, false_fn, x)
        tol_reached = delta_x < tol
        return (tol_reached, x), x
      
    
    tol_reached = False
    _, x = jax.lax.scan(scan_fn, (tol_reached, x), jnp.arange(max_iter))
    return x[-1].reshape(shape)



def root(fun, x0, args=(), method='newton-raphson', tol=1e-3, max_iter=100):
    """Find a root of a function, using a fixed point iteration.

    Args:
        fun (Callable): Function to find root of.
        x0 (Array): Initial value.
        args (tuple, optional): Extra arguments to pass to function. Defaults to ().
        method (str, optional): Method to use. Defaults to 'fixpoint'.
        tol (float, optional): Tolerance. Defaults to 1e-3.

    Returns:
        Array: Root of function.
    """

    # Dtype constraints on tolerance
    dtype = x0.dtype
    precission = jnp.finfo(dtype).precision
    tol = max(tol, precission)

    _f = lambda x: fun(x, *args)

    if method == 'newton-raphson':
        return newton_raphson(_f, x0, tol=tol, max_iter=max_iter)
    else:
        raise NotImplementedError(f"Method {method} not implemented.")

