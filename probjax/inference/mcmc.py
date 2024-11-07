from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Key

from probjax.inference.kernels.base import MarkovKernel, Params, State
from probjax.utils.jaxutils import WithProgressBarAPI, print_scan


class MCMC(WithProgressBarAPI):
    _running_stats = ("acceptance_rate",)
    _state_gamma = 0.9

    def __init__(
        self,
        kernel: MarkovKernel,
        verbose: bool = False,
    ) -> None:
        self.kernel = kernel
        self.verbose = verbose

    @partial(jax.jit, static_argnums=(0, 3))
    def run(
        self,
        key: Key,
        state: State,
        num_steps: int,
        params: Optional[Params] = None,
    ):
        def scan_fn(carry, _):
            key, state = carry
            key, new_key = jax.random.split(key)
            new_state, info = self.kernel(key, state, params)
            return (new_key, new_state), info_filter(info)

        if params is None:
            params = self.kernel.init_params(state.position)

        carry = (key, state)

        if not self.verbose:
            # We don't need the info, so we can just run the steps
            info_filter = lambda x: None
            (_, out_state), info = jax.lax.scan(scan_fn, carry, length=num_steps)
            return out_state
        else:
            # We need the info, so we need to keep track of the stats
            info_filter = lambda x: tuple([
                getattr(x, stat) for stat in self._running_stats
            ])
            update_stats = lambda stats, _, y: tuple([
                self._state_gamma * stats[i] + (1 - self._state_gamma) * y[i]
                for i in range(len(stats))
            ])
            print_fn = lambda i, total, state: self._print_progress(
                type(self), i, total, state
            )
            init_stats = tuple([0.0 for _ in self._running_stats])
            (_, out_state), _ = print_scan(
                scan_fn,
                carry,
                init_stats,
                length=num_steps,
                update_stats=update_stats,
                print_fn=print_fn,
                print_rate=num_steps // self._print_rate + 1,
            )
        return out_state

    def sample(
        self,
        key: Key,
        state: State,
        num_samples: int,
        params: Optional[Params] = None,
        thin: int = 1,
    ):
        samples = jax.tree_map(
            lambda x: jnp.empty((num_samples,) + x.shape), state.position
        )

        def scan_fn(carry, i):
            samples, key, state = carry
            key, new_key = jax.random.split(key)

            def inner_scan_fn(carry, _):
                key, state = carry
                key, new_key = jax.random.split(key)
                new_state, info = self.kernel(key, state, params)
                return (new_key, new_state), info_filter(info)

            (_, new_state), info = jax.lax.scan(
                inner_scan_fn, (new_key, state), length=thin
            )

            samples = jax.tree_util.tree_map(
                lambda s, s_new: s.at[i].set(s_new), samples, new_state.position
            )
            if info is not None:
                info = jax.tree_util.tree_map(
                    jnp.mean, info
                )  # Average the info over the thinning

            return (samples, new_key, new_state), info

        if params is None:
            params = self.kernel.init_params(state.position)

        if not self.verbose:
            info_filter = lambda x: None
            carry = (samples, key, state)
            (samples, _, state), _ = jax.lax.scan(
                scan_fn, carry, jnp.arange(num_samples), length=num_samples
            )
        else:
            info_filter = lambda x: tuple(
                getattr(x, stat) for stat in self._running_stats
            )
            update_stats = lambda stats, _, y: (0.6 * stats[0] + 0.4 * y[0],)
            print_fn = lambda i, total, state: self._print_progress(
                type(self), i, total, state
            )
            init_stats = (0.0,)
            carry = (samples, key, state)
            (samples, _, state), _ = print_scan(
                scan_fn,
                carry,
                init_stats,
                jnp.arange(num_samples),
                length=num_samples,
                update_stats=update_stats,
                print_fn=print_fn,
                print_rate=num_samples // self._print_rate + 1,
            )

        return samples, state
