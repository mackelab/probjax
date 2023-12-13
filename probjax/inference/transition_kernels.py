from typing import Any, Callable
from jax.random import PRNGKey
from jaxtyping import PyTree, Array

import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import KeyArray

from abc import abstractmethod
from functools import partial


# This code is used to represent the state of the MCMC chain.
# It is used by the MCMC kernel to keep track of the current state of the chain.
# The state consists of a random number generator key, a value x, and a dictionary of statistics and paramaters.


@register_pytree_node_class
class MCMCState:
    """MCMC state object."""

    def __init__(
        self, key: KeyArray, x: Array, stats: dict = {}, params: dict = {}
    ) -> None:
        """This class represents the state of the MCMC chain.

        Args:
            key (KeyArray): A random number generator key.
            x (Array): Current value of the chain.
            stats (dict, optional): Statistics that are tracked for the chain. Defaults to {}.
            params (dict, optional): Adaptive parameters for the MCMCKernel. Defaults to {}.
        """
        self.key = key
        self.x = x
        self.params = params
        self.stats = stats

    def set_x(self, x: Array):
        self.x = x
        return self

    def __repr__(self) -> str:
        return f"MCMCState(key={self.key}, x={self.x})"

    # Jax stuff
    def tree_flatten(self):
        return (self.key, self.x, self.stats, self.params), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def unzip_vals(states: PyTree[MCMCState] | MCMCState) -> PyTree[Array] | Array:
    """Unzips the states into a tuple of (x, key)"""
    x = jax.tree_map(lambda x: x.x, states, is_leaf=lambda x: isinstance(x, MCMCState))
    return x


class MCMCKernel:
    requires_mh: bool = True
    requires_potential: bool = False
    symmetric: bool = True

    def __call__(
        self, state: PyTree[MCMCState] | MCMCState
    ) -> PyTree[MCMCState] | MCMCState:
        new_state = jax.tree_map(
            self._update_mcmc_state, state, is_leaf=lambda x: isinstance(x, MCMCState)
        )
        return new_state

    def _update_mcmc_state(self, state: MCMCState) -> MCMCState:
        key1, key2 = jrandom.split(state.key)
        x_new = self._sample(key1, state.x, **state.params)
        stats = self.update_stats(state.x, x_new, state.stats, state.params)
        params = self.update_params(state.x, x_new, state.stats, state.params)
        new_state = MCMCState(key2, x_new, stats, params)
        return new_state

    @abstractmethod
    def _sample(self, key, x, **params) -> Array:
        pass

    # Only required if not symmetric
    def log_potential(self, x, x_new, **params):
        pass

    def init_params(self) -> dict:
        return {}

    def init_stats(self) -> dict:
        return {}

    def update_params(self, x, x_new, stats, params) -> dict:
        return params

    def update_stats(self, x, x_new, stats, params) -> dict:
        return stats


class PotentialBasedMCMCKernel(MCMCKernel):
    requires_potential: bool = True
    _potential_fn: Callable

    @property
    def potential_fn(self) -> Callable:
        return self._potential_fn

    @potential_fn.setter
    def potential_fn(self, potential_fn: Callable):
        self.set_potential_fn(potential_fn)

    def set_potential_fn(self, potential_fn: Callable):
        self._potential_fn = potential_fn
        return self


class GradientBasedMCMCKernel(PotentialBasedMCMCKernel):
    def set_potential_fn(self, potential_fn: Callable[..., Any]):
        def _potential_fn(x):
            return jnp.sum(potential_fn(x))

        self._potential_value_and_grad_fn = jax.value_and_grad(_potential_fn)
        return super().set_potential_fn(potential_fn)


class MetropolisHastingsKernel(PotentialBasedMCMCKernel):
    requires_potential: bool = True
    requires_mh: bool = False
    symmetric: bool = True

    def __init__(self, inner_kernel: PyTree[MCMCKernel] | MCMCKernel) -> None:
        super().__init__()
        self.inner_kernel = inner_kernel

    def __call__(
        self, state: PyTree[MCMCState] | MCMCState
    ) -> PyTree[MCMCState] | MCMCState:
        new_state = self.inner_kernel(state)
        key, key_accept = jnp.split(new_state.key)
        val_old, val_new = unzip_vals((state, new_state))
        # First also check if we cached evaluations in the state!
        logratio = self._mh_hastings_logratio(val_old, val_new)
        accept = jnp.log(jrandom.uniform(key, logratio.shape)) < logratio
        # Update the state
        val = jax.tree_map(
            lambda v_new, v_old: jnp.where(accept, v_new, v_old),
            val_new,
            val_old,
        )
        new_state = jax.tree_map(
            lambda s, v: s.set_x(v),
            new_state,
            val,
            is_leaf=lambda x: isinstance(x, MCMCState),
        )

        return new_state

    def _mh_hastings_logratio(
        self,
        val_old: PyTree[Array] | Array,
        val_new: PyTree[Array] | Array,
    ) -> Array:
        logratio = self.potential_fn(*val_new) - self.potential_fn(*val_old)

        # Proposal ratio
        def f(k, x_old, x_new):
            if k.symmetric:
                return 0.0
            else:
                return k.log_potential(x_old, x_new) - k.log_potential(x_new, x_old)

        logratio += jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_util.tree_map(f, self.inner_kernel, val_old, val_new),
        )

        return jnp.clip(logratio, a_max=0)


class GaussianKernel(MCMCKernel):
    def __init__(self, step_size: float = 0.1) -> None:
        self.step_size = step_size

    def _sample(self, key, x, step_size=0.1):
        return x + jrandom.normal(key, shape=x.shape) * self.step_size

    def log_potential(self, x, x_new, **params):
        return -0.5 / self.step_size**2 * jnp.sum(
            (x_new - x) ** 2, axis=-1
        ) + jnp.log(self.step_size)


class UniformKernel(MCMCKernel):
    def __init__(self, step_size: float = 0.1) -> None:
        self.step_size = step_size

    def _sample(self, key, x, step_size=0.1):
        return (
            x
            + jrandom.uniform(key, shape=x.shape) * 2 * self.step_size
            - self.step_size
        )


class LangevinDynKernel(GradientBasedMCMCKernel):
    requires_mh: bool = True
    symmetric: bool = False
    _potential_fn: Callable
    _potential_value_and_grad_fn: Callable

    def __init__(self, step_size=0.1) -> None:
        super().__init__()
        self.step_size = step_size

    def _sample(self, key, x):
        _, grad = self._potential_value_and_grad_fn(x)
        return (
            x
            + 0.5 * self.step_size * grad
            + jrandom.normal(key, shape=x.shape) * self.step_size
        )



class HMCKernel(GradientBasedMCMCKernel):
    requires_mh: bool = True

    def __init__(self, step_size=0.1, num_steps=10) -> None:
        super().__init__()
        self.step_size = step_size
        self.num_steps = num_steps
        self.energies = (0.5, 0.5)
        self.x_cache = None

    def _sample(self, key, x):
        # Sample random momentum
        momentum = jrandom.normal(key, shape=x.shape)
        kinetic_energey = 0.5 * jnp.sum(momentum**2)
        self.x_cache = x
        def body_fn(i, carry):
            x, momentum = carry
            _, grad = self._potential_value_and_grad_fn(x)
            momentum += 0.5 * self.step_size * grad
            x += self.step_size * momentum
            _, grad = self._potential_value_and_grad_fn(x)
            momentum += 0.5 * self.step_size * grad
            return (x, momentum)

        (x_new, momentum) = jax.lax.fori_loop(0, self.num_steps, body_fn, (x, momentum))

        new_kinetic_energy = 0.5 * jnp.sum(momentum**2)
        self.energies = (kinetic_energey, new_kinetic_energy)
        return x_new
    
    def log_potential(self, x, x_new, **params):
        mask1 = jnp.allclose(x, self.x_cache)
        mask2 = jnp.allclose(x_new, self.x_cache)

        return -mask1 * self.energies[0] - mask2 * self.energies[1]


class SliceKernel(PotentialBasedMCMCKernel):
    def __init__(self, bracket_step_size=0.1, slice_direction="axis") -> None:
        super().__init__()
        self.bracket_step_size = bracket_step_size
        self.slice_direction = slice_direction

    def _sample_slice_direction(self, key, x):
        if self.slice_direction == "axis":
            if x.ndim == 1:
                direction = jnp.ones_like(x)
            else:
                axis = jrandom.randint(key, shape=x.shape[-1], minval=0, maxval=x.ndim)
                direction = jnp.zeros_like(x)
                direction = direction.at[..., axis].set(1.0)
        elif self.slice_direction == "random":
            direction = jrandom.normal(key, shape=x.shape)
            direction = direction / jnp.linalg.norm(direction)
        else:
            raise ValueError("Invalid slice direction")
        return direction

    def _sample(self, key, x):
        key_direction, key_bracket, key_shrinkage = jrandom.split(key, 3)
        direction = self._sample_slice_direction(key_direction, x)
        u = jrandom.uniform(key_bracket, shape=x.shape)
        potential = self.potential_fn(x)
        y = jnp.log(u) + potential

        # Bracket expansion phase
        def cond_fn(carry):
            x, y, direction, mask = carry
            return jnp.any(mask)

        def body_fn(carry):
            x, y, direction, mask = carry

            # Only evaluate the potential if we need to
            def update_x(x, y, direction):
                x += self.bracket_step_size * direction
                potential = self.potential_fn(x)
                mask = potential > y
                return x, mask

            # If we don't need to evaluate the potential, just return the current x
            def finished_x(x, y, direction):
                return x, False

            x, mask = jax.vmap(jax.lax.cond, in_axes=(0, None, None, 0, 0, 0))(
                mask, update_x, finished_x, x, y, direction
            )

            return (x, y, direction, mask)

        x_upper, _, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (x, y, direction, jnp.ones_like(y, dtype=bool))
        )
        x_lower, _, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (x, y, -direction, jnp.ones_like(y, dtype=bool))
        )

        # Shrinkage phase requires rejection!
        key_shrinkage, key_rejections = jrandom.split(key_shrinkage)
        x_new = (
            jrandom.uniform(key_shrinkage, shape=x.shape) * (x_upper - x_lower)
            + x_lower
        )
        # potential_new = self.potential_fn(x_new)
        # rejection_mask = potential_new < y

        # def cond_fn2(carry):
        #     x_new,x_lower, x_upper, y, mask = carry
        #     return jnp.any(mask)

        # def body_fn2(carry):
        #     x_new,x_lower, x_upper, y, mask = carry

        #     def update_x(x_lower, x_upper, y):
        #         x_lower
        return x_new
