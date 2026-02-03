from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from probjax.utils.typing import Array, PyTree, RngKey

from probjax.inference.mcmc.adaptation import step_size_adaption
from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel


class LatentSliceParams(NamedTuple):
    step_size: float


class LatentSliceState(NamedTuple):
    position: PyTree
    logdensity: float
    latent_bracket_width: PyTree


class LatentSliceInfo(NamedTuple):
    num_evals: int
    proposal: LatentSliceState


def init_params(state: PyTree, step_size: float = 0.5) -> LatentSliceParams:
    return LatentSliceParams(step_size=step_size)


def init(
    position: PyTree, logdensity_fn: Callable, rng_key=None
) -> LatentSliceState:
    log_density = logdensity_fn(position)
    latent_bracket_width = jnp.zeros_like(position)
    return LatentSliceState(position, log_density, latent_bracket_width)


def build_kernel():
    def kernel(
        rng_key: RngKey,
        state: LatentSliceState,
        logdensity_fn: Callable,
        step_size: float = 0.5,
        max_evals: int = 100,
    ):
        rng_key, rng_slice, rng_bound, rng_update = jax.random.split(rng_key, 4)
        position = state.position
        logdensity = state.logdensity
        width = state.latent_bracket_width
        u = jax.random.uniform(rng_slice, shape=())
        y = jnp.log(1.0 - u) + logdensity

        # Calculate midpoint
        u_bound = jax.random.uniform(rng_bound, shape=position.shape)
        midpoint = width * u_bound + (position - width / 2)
        diff = jnp.abs(position - midpoint) * 2

        # Update bracket width
        u_update = jax.random.uniform(rng_update, shape=position.shape)
        width_update = diff - jnp.log(1.0 - u_update) / step_size

        # Update left and right bounds
        left = midpoint - width_update / 2
        right = midpoint + width_update / 2

        # Sample new position
        x_new, log_density, evals = accept_reject_proposal(
            logdensity_fn, rng_key, left, right, position, y, max_evals
        )

        new_state = LatentSliceState(x_new, log_density, width_update)
        info = LatentSliceInfo(evals, state)

        return new_state, info

    return kernel


def build_step(logdensity_fn: Callable, max_evals: int = 100) -> Callable:
    kernel_builder = build_kernel
    return make_step_from_kernel(
        logdensity_fn,
        kernel_builder,
        call_defaults={"max_evals": max_evals},
    )


def build_adaptation(
    logdensity_fn: Callable,
    max_evals: int = 100,
) -> Callable:
    def fit_params(
        key: RngKey,
        state: PyTree,
        params: LatentSliceParams,
        num_steps: int = 100,
        target_num_evals: int = 5,
        method: str = "step_size",
        t0: int = 10,
        gamma: float = 0.05,
        kappa: float = 0.75,
        **_,
    ) -> Tuple[LatentSliceState, LatentSliceParams]:
        position = state.position if hasattr(state, "position") else state
        if method == "step_size":
            adaption_alg = step_size_adaption(
                latent_slice,
                logdensity_fn,
                params,
                target=float(target_num_evals) / max_evals,
                target_from_info_fn=lambda info: info.num_evals / max_evals,
                t0=t0,
                gamma=gamma,
                kappa=kappa,
                algorithm_kwargs={
                    "max_evals": max_evals,
                },
            )
        else:
            raise ValueError("Invalid method")

        out, _ = adaption_alg.run(key, position, num_steps)
        return out.state, out.parameters

    return fit_params


latent_slice = make_kernel_api(
    name="latent_slice",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=build_step,
    build_adaptation_fn=build_adaptation,
)


def accept_reject_proposal(
    logdensity_fn: Callable,
    key: RngKey,
    lower: Array,
    upper: Array,
    state: Array,
    y: float,
    max_evals: int,
):
    def cond_fn(carry):
        i, is_accepted, *_ = carry
        return ~is_accepted & (i < max_evals)

    def body_fn(carry):
        i, is_accepted, rng, lower, upper, x, log_density = carry
        rng, key = jax.random.split(rng)
        u = jax.random.uniform(key, shape=x.shape)
        x_new = lower + u * (upper - lower)
        log_density = logdensity_fn(x_new)
        is_accepted = y < log_density

        def accept(x, x_new, lower, upper):
            return x_new, lower, upper

        def reject(x, x_new, lower, upper):
            mask_smaller = x_new < x
            mask_larger = x_new > x
            lower = jnp.where(mask_smaller, x_new, lower)
            upper = jnp.where(mask_larger, x_new, upper)
            return x, lower, upper

        x_new, lower, upper = jax.lax.cond(
            is_accepted,
            accept,
            reject,
            x,
            x_new,
            lower,
            upper,
        )

        return i + 1, is_accepted, rng, lower, upper, x_new, log_density

    init_carry = (0, jnp.array(False), key, lower, upper, state, y)
    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    i, is_accepted, _, _, _, x, logdensity = final_carry

    return x, logdensity, i
