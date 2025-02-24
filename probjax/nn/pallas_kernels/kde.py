from typing import Callable
import jax.numpy as jnp
from jax import Array, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


@pl.pallas_call
def _kde_kernel_impl(
    X_train_slice, X_test_slice, out_block, kernel_fn_ptr: pl.Function,
    BLOCK_M: pl.Axis, BLOCK_N: pl.Axis
):
    # Compute the kernel for this slice and accumulate
    m = pl.program_id(axis=BLOCK_M)
    n = pl.program_id(axis=BLOCK_N)
    val = kernel_fn_ptr(X_test_slice[m], X_train_slice[n])
    out_block[m, n] += val

def kde_kernel(
    X_train_ref,
    X_test_ref,
    kernel_fn: Callable,
    block_size: int = 128,
):
    """
    Kernel density estimation using a given kernel function.
    """
    # Convert inputs to device arrays
    X_train = jnp.asarray(X_train_ref)
    X_test = jnp.asarray(X_test_ref)

    # Create output buffer
    out = jnp.zeros((X_test.shape[0], X_train.shape[0]), dtype=X_test.dtype)

    # Launch pallas kernel
    out = _kde_kernel_impl(
        X_train, X_test, out, kernel_fn,
        BLOCK_M=block_size, BLOCK_N=block_size
    )

    # Sum across the training dimension to produce a single density value per X_test entry
    densities = out.sum(axis=1)

    # Return the aggregated result
    return densities
