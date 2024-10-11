from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Array, ArrayLike


class ARSState(NamedTuple):
    x: Array  # Evaluation points used to define the piecewise linear function
    h: Array  # Function values at the evaluation points
    hprime: Array  # Derivative values at the evaluation points
    z: Array  # Intersection points of the piecewise linear function (tangent lines)
    u: Array  # Proposal log density values at the intersection points
    s: Array  # Cumulative distribution function values at the intersection points
    num_points: int  # Number of points used to define the piecewise linear function


def init_ars_state(
    log_density_fn: Callable,
    xi: Array,
    lb: ArrayLike = -jnp.inf,
    ub: ArrayLike = jnp.inf,
    max_points: int = 50,
):
    num_initial_points = len(xi)

    x = jnp.sort(xi)
    h, hprime = jax.vmap(jax.value_and_grad(log_density_fn))(x)

    z = (jnp.diff(h) + x[:-1] * hprime[:-1] - x[1:] * hprime[1:]) / -(
        jnp.diff(hprime) + 1e-8
    )
    z = jnp.concatenate([jnp.array([lb]), z, jnp.array([ub])])

    u = jnp.empty_like(z)
    # Log density values -> tangent lines
    u = u.at[0].set(hprime[0] * (z[0] - x[0]) + h[0])
    u = u.at[1:].set(hprime * (z[1:] - x) + h)

    s = jnp.zeros_like(u)
    # Integral of piecewise linear functions
    s_sub = jnp.where(
        jnp.abs(hprime) > 1e-5, jnp.diff(jnp.exp(u)) / hprime, z[1:] - z[:-1]
    )
    # Cumulative sum of the integrals
    s = s.at[1:].set(jnp.cumsum(s_sub))
    # Can be infinite for bad initial points at the end
    s = s.at[-1].set(jnp.clip(s[-1], 0, 1e8))

    # Add infinities to the end of the arrays
    x = jnp.concatenate([x, jnp.full(max_points - num_initial_points, jnp.inf)])
    h = jnp.concatenate([h, jnp.full(max_points - num_initial_points, jnp.inf)])
    hprime = jnp.concatenate([
        hprime,
        jnp.full(max_points - num_initial_points, jnp.inf),
    ])

    z = jnp.concatenate([z, jnp.full(max_points - num_initial_points, jnp.inf)])
    u = jnp.concatenate([u, jnp.full(max_points - num_initial_points, jnp.inf)])
    s = jnp.concatenate([s, jnp.full(max_points - num_initial_points, jnp.inf)])

    return ARSState(x, h, hprime, z, u, s, num_initial_points)


def update_ars_state(state: ARSState, x: Array, h: Array, hprime: Array):
    # TODO improve this
    new_index = state.num_points

    def update(state, x, h, hprime):
        xs = state.x.at[new_index].set(x)
        h = state.h.at[new_index].set(h)
        hprime = state.hprime.at[new_index].set(hprime)
        idx = jnp.argsort(xs)
        xs = xs[idx]
        h = h[idx]
        hprime = hprime[idx]
        z = (jnp.diff(h) + xs[:-1] * hprime[:-1] - xs[1:] * hprime[1:]) / -(
            jnp.diff(hprime) + 1e-8
        )
        z = jnp.concatenate([jnp.array([-jnp.inf]), z, jnp.array([jnp.inf])])
        z = z.at[new_index + 1].set(jnp.inf)

        u = jnp.empty_like(z)
        u = u.at[0].set(hprime[0] * (z[0] - xs[0]) + h[0])
        u = u.at[1:].set(hprime * (z[1:] - xs) + h)

        s = jnp.zeros_like(u)
        s_sub = jnp.where(
            jnp.abs(hprime) > 1e-5, jnp.diff(jnp.exp(u)) / hprime, z[1:] - z[:-1]
        )
        s_sub = s_sub.at[new_index].set(jnp.clip(s_sub[new_index], 0, 1e8))

        s = s.at[1:].set(jnp.cumsum(s_sub))

        return ARSState(xs, h, hprime, z, u, s, state.num_points + 1)

    def do_nothing(state, x, h, hprime):
        return ARSState(
            state.x, state.h, state.hprime, state.z, state.u, state.s, state.num_points
        )

    max_points = len(state.x)
    state = jax.lax.cond(
        new_index < max_points, update, do_nothing, state, x, h, hprime
    )

    return state


def eval_upper(state: ARSState, x: Array, i: int):
    # Evaluate the upper piecewise linear function at x using the i-th segment.
    # NOTE: This will upper bound the log density function (if it is concave).
    return state.h[i] + state.hprime[i] * (x - state.x[i])


def eval_lower(state: ARSState, x: Array):
    # Evaluate a lower piecewise linear function at x.
    # NOTE: This will lower bound the log density function (if it is concave).
    i = jnp.searchsorted(state.x, x) - 1
    out = ((state.x[i + 1] - x) * state.h[i] + (x - state.x[i]) * state.h[i + 1]) / (
        state.x[i + 1] - state.x[i]
    )
    return jax.lax.cond(
        (i < 0) | (i == len(state.x) - 1), lambda x: -jnp.inf, lambda x: out, x
    )


def sample_upper(state: ARSState, key: PRNGKey):
    u = jax.random.uniform(key)
    max_index = state.num_points
    # Choose a bin with probability proportional to its contained probability mass.
    s_max = state.s[max_index]

    i = jnp.searchsorted(state.s / s_max, u) - 1

    def sample_nonzero_grad(state, u):
        # Piecewise linear function with non-zero gradient
        # -> Piecewise exponential distribution
        # -> Inverse CDF is a piecewise linear function
        xt = (
            state.x[i]
            + (
                -state.h[i]
                + jnp.log(
                    state.hprime[i] * (s_max * u - state.s[i]) + jnp.exp(state.u[i])
                )
            )
            / state.hprime[i]
        )
        return xt

    def sample_zero_grad(state, u):
        # Piecewise linear function with zero gradient
        # -> Piecewise uniform distribution
        # -> Inverse CDF is a piecewise linear function
        lower = state.z[i]
        upper = state.z[i + 1]
        lower_s = state.s[i] / s_max
        upper_s = state.s[i + 1] / s_max
        eps = (u - lower_s) / (upper_s - lower_s)

        return lower + (upper - lower) * eps

    xt = jax.lax.cond(
        jnp.abs(state.hprime[i]) > 1e-6, sample_nonzero_grad, sample_zero_grad, state, u
    )

    return xt, i


@partial(jax.jit, static_argnums=(2, 3))
def ars(state: ARSState, key: PRNGKey, num_samples: int, log_density_fn: Callable):
    samples = jnp.zeros(num_samples)
    n = 0

    def cond_fn(carry):
        _, n, _, _, _ = carry
        return n < num_samples

    def body_fn(carry):
        key, n, state, samples, iteration = carry
        key, sample_key, rejection_key = jax.random.split(key, 3)

        [xt, i] = sample_upper(state, sample_key)

        lh = eval_lower(state, xt)
        uh = eval_upper(state, xt, i)

        u = jax.random.uniform(rejection_key)

        def accept(xt, n, state, samples):
            samples = samples.at[n].set(xt)
            n += 1
            return samples, n, state

        def reject(xt, n, state, samples):
            return samples, n, state

        def reject_maybe_accept_update(xt, n, state, samples):
            h = log_density_fn(xt)
            hprime = jax.grad(log_density_fn)(xt)

            samples, n, state = jax.lax.cond(
                u <= jnp.exp(h - uh), accept, reject, xt, n, state, samples
            )
            state = update_ars_state(state, xt, h, hprime)
            return samples, n, state

        samples, n, state = jax.lax.cond(
            (u <= jnp.exp(lh - uh)) & (lh <= uh),
            accept,
            reject_maybe_accept_update,
            xt,
            n,
            state,
            samples,
        )

        return (key, n, state, samples, iteration + 1)

    carry = (key, n, state, samples, 0)
    _, _, state, samples, iterations = jax.lax.while_loop(cond_fn, body_fn, carry)

    return samples, state, num_samples / iterations


# TODO Add A2RMS
