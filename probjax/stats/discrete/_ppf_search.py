"""Shared inverse-CDF search for discrete distributions."""

import jax
import jax.numpy as jnp


def ppf_by_cdf_search(q, cdf_fn, *cdf_params, hi=None):
    """Smallest ``k`` with ``cdf_fn(k, *cdf_params) >= q`` via linear scan.

    Args:
        q: Quantiles in [0, 1] (clipped).
        cdf_fn: Callable ``(k, *cdf_params) -> cdf value``.
        *cdf_params: Per-element CDF parameters, broadcast with ``q``.
        hi: Optional per-element search cap (e.g. binomial ``n``); the
            result is clamped to ``hi`` and the scan stops past it.

    Returns:
        Array of quantiles broadcast to the inputs' shape.
    """
    q = jnp.clip(jnp.asarray(q), 0, 1)
    operands = (q, *cdf_params) if hi is None else (q, *cdf_params, hi)
    flat = jnp.broadcast_arrays(*operands)
    has_hi = hi is not None

    def ppf_single(q_single, *rest):
        if has_hi:
            *params_single, hi_single = rest
        else:
            params_single, hi_single = rest, None

        def body_fun(state):
            k, found = state
            found = found | (cdf_fn(k, *params_single) >= q_single)
            return (k + 1, found)

        def cond_fun(state):
            k, found = state
            if hi_single is None:
                return ~found
            # Stop past hi: guards q close to 1 when fp error keeps
            # the CDF from quite reaching q.
            return ~found & (k <= hi_single)

        # Subtract 1: the loop increments once past the found k.
        final_k, _ = jax.lax.while_loop(cond_fun, body_fun, (0, False))
        if hi_single is None:
            return final_k - 1
        return jnp.where(final_k > hi_single, hi_single, final_k - 1)

    for _ in range(flat[0].ndim):
        ppf_single = jax.vmap(ppf_single)
    return ppf_single(*flat)
