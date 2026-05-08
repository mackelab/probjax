from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from probjax.utils.typing import RngKey

from probjax.inference.mcmc.base import MarkovKernel, Params, State
from probjax.utils.jaxutils import WithProgressBarAPI, print_scan


class MCMC(WithProgressBarAPI):
    """Run an MCMC kernel, optionally displaying a progress bar.

    Args:
        kernel: A :class:`MarkovKernel`.
        verbose: If ``True``, display a progress bar with running stats.
        tracked_stats: Names of scalars to display in the progress bar.
            Each name is looked up first on the **state**, then on the
            **info** returned by the kernel.  If a name is not found on
            either, ``NaN`` is shown.  Defaults to
            ``("logdensity", "acceptance_rate")``.
    """

    _default_tracked_stats = ("logdensity", "acceptance_rate")

    def __init__(
        self,
        kernel: MarkovKernel,
        verbose: bool = False,
        tracked_stats: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.kernel = kernel
        self.verbose = verbose
        self.tracked_stats = tracked_stats or self._default_tracked_stats

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0, 3))
    def run(
        self,
        key: RngKey,
        state: State,
        num_steps: int,
        params: Optional[Params] = None,
        args: Optional[Tuple] = None,
    ):
        """Run the MCMC kernel for ``num_steps`` steps.

        Args:
            key: PRNG key.
            state: Initial MCMC state.
            num_steps: Number of steps to run.
            params: Kernel parameters (defaults via ``kernel.init_params``).
            args: Optional tuple of arrays whose leading axis is
                ``num_steps``.  Each step receives one slice, forwarded as
                extra positional arguments to the kernel (and ultimately to
                ``logdensity_fn``).
        """
        if params is None:
            params = self.kernel.init_params(state)

        def scan_fn(carry, xs):
            key, state = carry
            key, new_key = jax.random.split(key)
            if xs is None:
                new_state, info = self.kernel(key, state, params)
            else:
                new_state, info = self.kernel(key, state, params, *xs)
            stats = self._extract_stats(new_state, info)
            return (new_key, new_state), stats

        carry = (key, state)
        scan_length = num_steps if args is None else None

        if not self.verbose:
            (_, out_state), _ = jax.lax.scan(
                scan_fn, carry, xs=args, length=scan_length
            )
        else:
            update_stats, print_fn, init_stats, print_rate = self._make_verbose_fns(
                num_steps
            )
            (_, out_state), _ = print_scan(
                scan_fn,
                carry,
                init_stats,
                xs=args,
                length=scan_length,
                update_stats=update_stats,
                print_fn=print_fn,
                print_rate=print_rate,
            )
        return out_state

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------

    def sample(
        self,
        key: RngKey,
        state: State,
        num_samples: int,
        params: Optional[Params] = None,
        thin: int = 1,
        args: Optional[Tuple] = None,
    ):
        """Draw ``num_samples`` from the chain, thinning by ``thin`` steps.

        Args:
            key: PRNG key.
            state: Initial MCMC state.
            num_samples: Number of samples to collect.
            params: Kernel parameters (defaults via ``kernel.init_params``).
            thin: Number of kernel steps between collected samples.
            args: Optional tuple of arrays whose leading axis is
                ``num_samples * thin``.  Sliced so each thinning step
                receives one element, forwarded as extra positional
                arguments to the kernel.
        """
        if params is None:
            params = self.kernel.init_params(state)

        samples = jax.tree_util.tree_map(
            lambda x: jnp.empty((num_samples,) + x.shape), state.position
        )

        # Reshape args: (num_samples * thin, ...) -> (num_samples, thin, ...)
        if args is not None:
            outer_args = jax.tree_util.tree_map(
                lambda x: x.reshape((num_samples, thin) + x.shape[1:]), args
            )
        else:
            outer_args = None

        def scan_fn(carry, xs):
            if outer_args is None:
                i = xs
                step_args_chunk = None
            else:
                i, step_args_chunk = xs

            samples, key, state = carry
            key, new_key = jax.random.split(key)

            def inner_scan_fn(carry, inner_xs):
                key, state = carry
                key, new_key = jax.random.split(key)
                if inner_xs is None:
                    new_state, info = self.kernel(key, state, params)
                else:
                    new_state, info = self.kernel(key, state, params, *inner_xs)
                stats = self._extract_stats(new_state, info)
                return (new_key, new_state), stats

            (_, new_state), step_stats = jax.lax.scan(
                inner_scan_fn,
                (new_key, state),
                xs=step_args_chunk,
                length=thin if step_args_chunk is None else None,
            )

            samples = jax.tree_util.tree_map(
                lambda s, s_new: s.at[i].set(s_new), samples, new_state.position
            )
            # Average the per-thinning-step stats for the progress bar
            avg_stats = jax.tree_util.tree_map(jnp.mean, step_stats)
            return (samples, new_key, new_state), avg_stats

        # Outer scan xs: always includes indices, optionally args chunks
        indices = jnp.arange(num_samples)
        outer_xs = (indices, outer_args) if outer_args is not None else indices

        if not self.verbose:
            carry = (samples, key, state)
            (samples, _, state), _ = jax.lax.scan(
                scan_fn, carry, outer_xs, length=num_samples
            )
        else:
            update_stats, print_fn, init_stats, print_rate = self._make_verbose_fns(
                num_samples
            )
            carry = (samples, key, state)
            (samples, _, state), _ = print_scan(
                scan_fn,
                carry,
                init_stats,
                outer_xs,
                length=num_samples,
                update_stats=update_stats,
                print_fn=print_fn,
                print_rate=print_rate,
            )

        return samples, state
