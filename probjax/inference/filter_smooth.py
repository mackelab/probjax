from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.typing import ArrayLike

from probjax.inference.filtering.base import FilterInfo, FilterKernel, FilterState
from probjax.inference.filtering.smoothing import (
    particle_smoother,
    smooth as smooth_gaussian,
)
from probjax.utils.jaxutils import (
    WithProgressBarAPI,
    nested_checkpoint_scan,
)


class FilteringTrace(NamedTuple):
    ts: ArrayLike
    initial_state: FilterState
    states: Any
    infos: Any
    outputs: Any


def _extract_covariance(states):
    if hasattr(states, "cov"):
        return states.cov
    elif hasattr(states, "std"):
        return jax.vmap(lambda s: s @ s.T)(states.std)
    elif hasattr(states, "cov_factor") and hasattr(states, "cov_core"):
        return jax.vmap(lambda u, s: u @ s @ u.T)(states.cov_factor, states.cov_core)
    raise ValueError(
        "Filtering states must have either `cov`, `std`, or (`cov_factor`, `cov_core`)."
    )


def _extract_pred_covariance(infos):
    if hasattr(infos, "cov_pred"):
        return infos.cov_pred
    elif hasattr(infos, "std_pred"):
        return jax.vmap(lambda s: s @ s.T)(infos.std_pred)
    elif hasattr(infos, "cov_factor_pred") and hasattr(infos, "cov_core_pred"):
        return jax.vmap(lambda u, s: u @ s @ u.T)(
            infos.cov_factor_pred, infos.cov_core_pred
        )
    raise ValueError(
        "Filtering infos must have either `cov_pred`, `std_pred`, "
        "or (`cov_factor_pred`, `cov_core_pred`)."
    )


def _trace_ts(trace: FilteringTrace):
    num_states = jax.tree_util.tree_leaves(trace.states)[0].shape[0]
    if trace.ts.shape[0] == num_states + 1:
        return trace.ts[1:]
    return trace.ts


def unpack_log_likelihood(state: FilterState, info: FilterInfo):
    if info is not None and hasattr(info, "log_likelihood"):
        return info.log_likelihood
    else:
        return jnp.array(0.0)


class Filter(WithProgressBarAPI):
    """Run a filter kernel, optionally displaying a progress bar.

    Args:
        kernel: A :class:`FilterKernel`.
        verbose: If ``True``, display a progress bar with log-likelihood.
    """

    _running_stats = ("log_likelihood",)
    _ema_gamma = 0.9

    def __init__(self, kernel: FilterKernel, verbose: bool = False) -> None:
        self.kernel = kernel
        self.verbose = verbose

    def _extract_stats(self, info):
        if info is not None and hasattr(info, "log_likelihood"):
            return (jnp.float32(info.log_likelihood),)
        return (jnp.float32(jnp.nan),)

    def filter(
        self,
        key: PRNGKey,
        ts: ArrayLike,
        t_o: Optional[ArrayLike],
        x_o: Optional[ArrayLike],
        *args,
        unpack_fn: Optional[Callable] = None,
        checkpoint_lengths: Optional[Sequence[int]] = None,
        unroll: int = 1,
        **kwargs,
    ) -> FilteringTrace:
        """Run filtering over the time grid.

        Args:
            key: PRNG key.
            ts: Time grid of shape ``(T,)``.
            t_o: Observation times.
            x_o: Observation values.
            *args: Positional args forwarded to ``kernel.init``.
            unpack_fn: Optional extractor applied to ``(state, info)`` each step.
                Defaults to ``kernel.default_unpack``.
            checkpoint_lengths: If set, use ``nested_checkpoint_scan``
                with these chunk lengths.
            unroll: Unroll factor for the scan.
            **kwargs: Keyword args forwarded to ``kernel.init``.

        Returns:
            FilteringTrace containing times, initial state, states, infos,
            and outputs.
        """
        kernel = self.kernel
        initial_state = kernel.init(*args, t=ts[0], **kwargs)

        if unpack_fn is None:
            unpack_fn = kernel.default_unpack

        def scan_fn(carry, t):
            state, key, i = carry
            key, subkey = jax.random.split(key)
            is_observed = t == t_o[i]

            def update_fn(subkey, state, i):
                state, info = kernel(state, t=t_o[i], observed=x_o[i], rng_key=subkey)
                return state, info, i + 1

            def predict_fn(subkey, state, i):
                state, info = kernel(state, t=t_o[i], rng_key=subkey)
                return state, info, i

            state, info, i = jax.lax.cond(
                is_observed, update_fn, predict_fn, subkey, state, i
            )
            out = unpack_fn(state, info)
            return (state, key, i), (state, info, out)

        carry = (initial_state, key, 0)
        num_steps = ts[1:].shape[0]

        if self.verbose:
            gamma = self._ema_gamma

            def verbose_scan_fn(carry, t):
                state, key, i, ema_ll = carry
                key, subkey = jax.random.split(key)
                is_observed = t == t_o[i]

                def update_fn(subkey, state, i):
                    state, info = kernel(
                        state, t=t_o[i], observed=x_o[i], rng_key=subkey
                    )
                    return state, info, i + 1

                def predict_fn(subkey, state, i):
                    state, info = kernel(state, t=t_o[i], rng_key=subkey)
                    return state, info, i

                state, info, i = jax.lax.cond(
                    is_observed, update_fn, predict_fn, subkey, state, i
                )
                out = unpack_fn(state, info)

                ll = self._extract_stats(info)[0]
                ema_ll = gamma * ema_ll + (1 - gamma) * ll
                print_rate = num_steps // self._print_rate + 1
                jax.debug.callback(
                    lambda step, total, stats: type(self)._write_progress(
                        type(self), step, total, stats, ("log_likelihood",)
                    ),
                    i,
                    num_steps,
                    (ema_ll,),
                )

                return (state, key, i, ema_ll), (state, info, out)

            carry_v = (initial_state, key, 0, jnp.float32(0.0))
            _, (states, infos, output) = jax.lax.scan(
                verbose_scan_fn, carry_v, ts[1:], unroll=unroll
            )
        elif checkpoint_lengths is None:
            _, (states, infos, output) = jax.lax.scan(
                scan_fn, carry, ts[1:], unroll=unroll
            )
        else:
            _, (states, infos, output) = nested_checkpoint_scan(
                scan_fn, carry, ts[1:], nested_lengths=checkpoint_lengths, unroll=unroll
            )

        return FilteringTrace(
            ts=ts,
            initial_state=initial_state,
            states=states,
            infos=infos,
            outputs=output,
        )

    def smooth(
        self,
        trace: FilteringTrace,
        smoother: Optional[Callable] = None,
        key: Optional[PRNGKey] = None,
        transition_logdensity_fn: Optional[Callable] = None,
    ) -> Any:
        """Smooth a filtering trace.

        For Gaussian filters (KF/EKF/UKF/SqKF), pass ``smoother``.
        For particle filters, pass ``key`` and ``transition_logdensity_fn``.

        Args:
            trace: FilteringTrace from :meth:`filter`.
            smoother: RTS-style smoothing callback (Gaussian filters).
            key: PRNG key for backward simulation (particle filters).
            transition_logdensity_fn: Log-transition density
                ``(x_tp1, x_t, t, tp1) -> scalar`` (particle filters).

        Returns:
            Gaussian: ``(mus_s, covs_s)`` arrays.
            Particle: ``(smoothed_particles, smoothed_log_weights)`` arrays.
        """
        states = trace.states

        if hasattr(states, "particles") and hasattr(states, "log_weights"):
            if key is None or transition_logdensity_fn is None:
                raise ValueError(
                    "Particle smoothing requires `key` and `transition_logdensity_fn`."
                )
            ts = _trace_ts(trace)
            ancestors = (
                trace.infos.ancestors if hasattr(trace.infos, "ancestors") else None
            )
            return particle_smoother(
                key,
                ts,
                states.particles,
                states.log_weights,
                transition_logdensity_fn,
                ancestors=ancestors,
            )

        if smoother is None:
            raise ValueError("Gaussian smoothing requires `smoother` callback.")

        ts = _trace_ts(trace)
        mus = states.mean
        covs = _extract_covariance(states)
        mus_pred = trace.infos.mean_pred
        covs_pred = _extract_pred_covariance(trace.infos)
        return smooth_gaussian(ts, mus, covs, mus_pred, covs_pred, smoother)

    def log_likelihood(
        self,
        key: PRNGKey,
        ts: ArrayLike,
        t_o: Optional[ArrayLike],
        x_o: Optional[ArrayLike],
        *args,
        checkpoint_lengths: Optional[Sequence[int]] = None,
        unroll: int = 1,
        **kwargs,
    ) -> ArrayLike:
        """Run filtering and return the summed log-likelihood.

        Args:
            key: PRNG key.
            ts: Time grid of shape ``(T,)``.
            t_o: Observation times.
            x_o: Observation values.
            *args: Positional args forwarded to ``kernel.init``.
            checkpoint_lengths: If set, use ``nested_checkpoint_scan``.
            unroll: Unroll factor for the scan.
            **kwargs: Keyword args forwarded to ``kernel.init``.

        Returns:
            Scalar log-likelihood.
        """
        output = self.filter(
            key,
            ts,
            t_o,
            x_o,
            *args,
            unpack_fn=unpack_log_likelihood,
            checkpoint_lengths=checkpoint_lengths,
            unroll=unroll,
            **kwargs,
        )
        return output.outputs.sum()


# ----------------------------------------------------------------------
# Backward-compatible free functions
# ----------------------------------------------------------------------


def filter(
    key: PRNGKey,
    ts: ArrayLike,
    t_o: Optional[ArrayLike],
    x_o: Optional[ArrayLike],
    kernel: FilterKernel,
    *args,
    unpack_fn: Optional[Callable] = None,
    checkpoint_lengths: Optional[Sequence[int]] = None,
    unroll: int = 1,
    return_trace: bool = False,
    **kwargs,
):
    """Run filtering.  Deprecated — prefer :class:`Filter`."""
    runner = Filter(kernel)
    trace = runner.filter(
        key,
        ts,
        t_o,
        x_o,
        *args,
        unpack_fn=unpack_fn,
        checkpoint_lengths=checkpoint_lengths,
        unroll=unroll,
        **kwargs,
    )
    if return_trace:
        return trace
    return trace.outputs


def smooth(
    *args,
    smoother: Optional[Callable] = None,
    key: Optional[PRNGKey] = None,
    transition_logdensity_fn: Optional[Callable] = None,
    **kwargs,
):
    """Smooth filtering output.  Deprecated — prefer :meth:`Filter.smooth`.

    For Gaussian filters: ``smooth(trace, smoother=...)``.
    For particle filters: ``smooth(trace, key=..., transition_logdensity_fn=...)``.
    Also accepts raw arrays for backward compatibility:
    ``smooth(ts, mus, covs, mus_pred, covs_pred, smoother)``.
    """
    if not args or not isinstance(args[0], FilteringTrace):
        return smooth_gaussian(*args, **kwargs)

    return Filter(None).smooth(
        args[0],
        smoother=smoother,
        key=key,
        transition_logdensity_fn=transition_logdensity_fn,
    )


def filter_and_smooth(
    key: PRNGKey,
    ts: ArrayLike,
    t_o: Optional[ArrayLike],
    x_o: Optional[ArrayLike],
    kernel: FilterKernel,
    *args,
    smoother: Optional[Callable] = None,
    smooth_key: Optional[PRNGKey] = None,
    transition_logdensity_fn: Optional[Callable] = None,
    checkpoint_lengths: Optional[Sequence[int]] = None,
    unroll: int = 1,
    **kwargs,
):
    """Run filtering and smoothing in one call.  Deprecated — prefer :class:`Filter`."""
    filt = Filter(kernel)
    trace = filt.filter(
        key,
        ts,
        t_o,
        x_o,
        *args,
        checkpoint_lengths=checkpoint_lengths,
        unroll=unroll,
        **kwargs,
    )
    if hasattr(trace.states, "particles") and hasattr(trace.states, "log_weights"):
        if transition_logdensity_fn is None:
            raise ValueError("Particle smoothing requires `transition_logdensity_fn`.")
        if smooth_key is None:
            smooth_key = key
        smoothed = filt.smooth(
            trace,
            key=smooth_key,
            transition_logdensity_fn=transition_logdensity_fn,
        )
    else:
        if smoother is None:
            raise ValueError("Gaussian smoothing requires `smoother` callback.")
        smoothed = filt.smooth(trace, smoother=smoother)

    return trace, smoothed


def filter_log_likelihood(
    key, ts, t_o, x_o, kernel, *args, checkpoint_lengths=None, unroll=1, **kwargs
):
    """Compute log-likelihood via filtering.  Deprecated — prefer :meth:`Filter.log_likelihood`."""
    return Filter(kernel).log_likelihood(
        key,
        ts,
        t_o,
        x_o,
        *args,
        checkpoint_lengths=checkpoint_lengths,
        unroll=unroll,
        **kwargs,
    )
