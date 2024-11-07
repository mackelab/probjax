from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Array, PyTree

# Flow matching objectives


@partial(jax.jit, static_argnames=("model_fn", "mean_fn", "std_fn"))
def conditional_flow_and_score_matching_loss(
    params: PyTree,
    key: PRNGKey,
    times: Array,
    xs_source: Array,
    xs_target: Array,
    model_fn: Callable,
    mean_fn: Callable,
    std_fn: Callable,
    *args,
    estimate_score: bool = False,
):
    """This function computes the conditional flow matching loss and score matching loss. By setting estimate_score to False, only the conditional flow matching loss is computed. By setting estimate_score to True, both the conditional flow matching loss and score matching loss are computed.


    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        key (PRNGKey): Random key.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_source (Array): Marginal distribution at time t=0, refered to as source distribution.
        xs_target (Array): Marginal distribution at time t, refered to as target distribution.
        model_fn (Callable): Model_fn that takes parameters, times, and samples as input and returns the vector field and optionally the marginal score. Should be a function of the form model_fn(params, times, xs_t) -> v_t(, s_t).
        mean_fn (Callable): The mean function of the Gaussian probability path, should satisfy the following:
                                - mean_fn(xs_source, xs_target, 0) -> xs_source
                                - mean_fn(xs_source, xs_target, 1) -> xs_target
                                - Lipschitz continuous in time
        std_fn (Callable): The standard deviation function of the Gaussian probability path, should satisfy the following:
                                - std_fn(xs_source, xs_target, 0) -> 0
                                - std_fn(xs_source, xs_target, 1) -> 0
                                - std_fn(xs_source, xs_target, t) > 0 for all t in [0, 1]
                                - Two times continuously differentiable in time.
        estimate_score (bool, optional): If set to true, both flow and score matching objectives are computed. Defaults to False.

    Returns:
        (loss_flow, Optional[loss_score]): Respective loss functions
    """
    # Sample x_t
    eps = jax.random.normal(key, shape=xs_source.shape)
    xs_t = (
        mean_fn(xs_source, xs_target, times) + std_fn(xs_source, xs_target, times) * eps
    )

    # Compute u_t -> For flow matching
    # This is valid for Gaussian probability paths, which is currented here.
    t = jnp.broadcast_to(
        times, xs_target.shape
    )  # Pad to x shape for jax.grad -> x.shape
    std_fn_grad = jax.grad(lambda x_s, x_t, t: std_fn(x_s, x_t, t).sum(), argnums=2)
    mean_fn_grad = jax.grad(lambda x_s, x_t, t: mean_fn(x_s, x_t, t).sum(), argnums=2)
    u_t = std_fn_grad(xs_source, xs_target, t) * eps + mean_fn_grad(
        xs_source, xs_target, t
    )

    # Compute loss
    if not estimate_score:
        # Compute vector field -> Flow matching loss
        v_t = model_fn(params, times, xs_t, *args)

        # Compute loss
        loss = jnp.mean(jnp.sum((v_t - u_t) ** 2, axis=-1))

        return loss
    else:
        # Compute vector field and marginal score -> Flow matching loss + Score matching loss
        v_t, s_t = model_fn(params, times, xs_t, *args)

        # Compute loss
        loss = jnp.mean(jnp.sum((v_t - u_t) ** 2, axis=-1))
        loss_score = jnp.mean(
            jnp.sum((s_t + 1 / std_fn(xs_source, xs_target, times) * eps) ** 2, axis=-1)
        )

        return loss, loss_score
