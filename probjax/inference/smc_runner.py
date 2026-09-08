"""Compiled fixed and adaptive SMC execution with constant-memory summaries."""

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from probjax.inference.base import SMCResult
from probjax.utils.jaxutils import WithProgressBarAPI, print_scan


def _sampler_state(state):
    return getattr(state, "sampler_state", state)


def _ess_from_weights(state, _info):
    state = _sampler_state(state)
    if hasattr(state, "weights"):
        return 1.0 / jnp.sum(state.weights**2)
    weights = state.persistent_weights.ravel()
    return weights.sum() ** 2 / jnp.sum(weights**2)


def _increment(previous, state, info):
    if hasattr(info, "log_likelihood_increment"):
        return info.log_likelihood_increment
    previous, state = _sampler_state(previous), _sampler_state(state)
    if hasattr(state, "log_Z"):
        return state.log_Z - previous.log_Z
    return jnp.array(jnp.nan)  # A custom kernel did not supply evidence information.


def _zero_info(fn, *args, **kwargs):
    shape = jax.eval_shape(fn, *args, **kwargs)[1]
    return jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), shape)


def _initial_evidence(state, supplied):
    if supplied is not None:
        return jnp.asarray(supplied)
    return jnp.asarray(getattr(_sampler_state(state), "log_Z", 0.0))


def _run_fixed(
    key,
    kernel,
    state,
    schedule,
    params,
    *,
    collect=False,
    adaptor=None,
    verbose=None,
    initial_log_evidence=None,
):
    schedule = jnp.asarray(schedule)
    if schedule.ndim < 1:
        raise ValueError("A schedule needs a leading step axis.")
    n = schedule.shape[0]
    prototype = jnp.zeros(schedule.shape[1:], schedule.dtype)
    info = _zero_info(
        kernel.step, key, state, tempering_param=prototype, mcmc_parameters=params
    )
    adaptation = () if adaptor is None else adaptor.init(state, params)
    evidence = _initial_evidence(state, initial_log_evidence)

    def step(carry, temperature):
        key, state, params, adaptation, logz, _ = carry
        key, step_key = jax.random.split(key)
        previous = state
        state, info = kernel.step(
            step_key, state, tempering_param=temperature, mcmc_parameters=params
        )
        logz = logz + _increment(previous, state, info)
        if adaptor is not None:
            adaptation, params, adaptation_info = adaptor.update(
                state, info, adaptation, params
            )
            trace = (state, info, adaptation_info) if collect else None
        else:
            trace = (state, info) if collect else None
        carry = (key, state, params, adaptation, logz, info)
        if verbose is not None:
            return carry, (verbose._extract_stats(state, info), trace)
        return carry, trace

    carry = (key, state, params, adaptation, evidence, info)
    if verbose is None:
        (key, state, params, adaptation, logz, info), trace = jax.lax.scan(
            step, carry, schedule
        )
    else:
        (key, state, params, adaptation, logz, info), (_, trace) = (
            verbose._verbose_scan(
                step, carry, schedule, n, stats_fn=lambda _carry, y: y[0]
            )
        )
    if adaptor is not None:
        params, adaptation_info = adaptor.finalize(adaptation, params)
        trace = (trace, adaptation_info) if collect else None
    params = getattr(state, "parameter_override", params)
    return SMCResult(
        state,
        params,
        trace,
        logz,
        info if n else None,
        jnp.array(n),
        jnp.array(True),
        key,
    )


class SMC(WithProgressBarAPI):
    """Compiled SMC runners, retaining evidence and final diagnostics by default.

    ``info`` remains the optional legacy trace. ``final_info`` is the last kernel
    diagnostic, ``log_evidence`` the accumulated estimate. For a resumed fixed
    run pass initial_log_evidence from the earlier result (persistent states
    already carry it). Custom kernels without evidence diagnostics return NaN.
    """

    _default_tracked_stats = ("ess", "log_likelihood_increment", "acceptance_rate")
    _computed_stats = {"ess": _ess_from_weights}
    _print_rate = 50

    def __init__(
        self,
        kernel,
        verbose=False,
        tracked_stats: Optional[Tuple[str, ...]] = None,
        collect=False,
    ):
        self.kernel, self.verbose, self.collect = kernel, verbose, collect
        self.tracked_stats = (
            self._default_tracked_stats if tracked_stats is None else tracked_stats
        )

    def _stat_objects(self, state, info):
        return (_sampler_state(state), info, getattr(info, "update_info", None))

    @staticmethod
    @partial(jax.jit, static_argnames=("kernel", "collect"))
    def run_kernel(
        key,
        kernel,
        state,
        tempering_params,
        params,
        *,
        collect=False,
        initial_log_evidence=None,
    ):
        return _run_fixed(
            key,
            kernel,
            state,
            tempering_params,
            params,
            collect=collect,
            initial_log_evidence=initial_log_evidence,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("kernel", "adaptor", "collect"))
    def adapt_kernel(
        key,
        kernel,
        adaptor,
        state,
        tempering_params,
        params,
        *,
        collect=False,
        initial_log_evidence=None,
    ):
        return _run_fixed(
            key,
            kernel,
            state,
            tempering_params,
            params,
            collect=collect,
            adaptor=adaptor,
            initial_log_evidence=initial_log_evidence,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("kernel", "max_steps", "adaptor"))
    def run_adaptive_kernel(
        key,
        kernel,
        state,
        params,
        *,
        max_steps=200,
        adaptor=None,
        initial_log_evidence=None,
    ):
        """Advance an adaptive kernel to temperature 1, with a fixed iteration cap.

        No history is allocated. completed=False signals the cap, exhausted
        persistent storage, nonfinite temperature/evidence, or stalled progress.
        Key/state/params can be passed back to resume a capped run.
        """
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        info = _zero_info(kernel.step, key, state, mcmc_parameters=params)
        adaptation = () if adaptor is None else adaptor.init(state, params)
        evidence = _initial_evidence(state, initial_log_evidence)

        def cond(carry):
            _, state, _, _, _, _, count, valid = carry
            raw = _sampler_state(state)
            room = True
            if hasattr(raw, "persistent_log_Z"):
                room = raw.iteration + 1 < raw.persistent_log_Z.shape[0]
            return (count < max_steps) & (raw.tempering_param < 1.0) & valid & room

        def body(carry):
            key, state, params, adaptation, logz, _, count, _ = carry
            key, step_key = jax.random.split(key)
            previous = state
            state, info = kernel.step(step_key, state, mcmc_parameters=params)
            logz += _increment(previous, state, info)
            if adaptor is not None:
                adaptation, params, _ = adaptor.update(state, info, adaptation, params)
            old = _sampler_state(previous).tempering_param
            new = _sampler_state(state).tempering_param
            valid = jnp.isfinite(new) & jnp.isfinite(logz) & (new > old)
            return key, state, params, adaptation, logz, info, count + 1, valid

        key, state, params, adaptation, logz, info, count, valid = jax.lax.while_loop(
            cond,
            body,
            (
                key,
                state,
                params,
                adaptation,
                evidence,
                info,
                jnp.array(0),
                jnp.array(True),
            ),
        )
        if adaptor is not None:
            params, _ = adaptor.finalize(adaptation, params)
        params = getattr(state, "parameter_override", params)
        return SMCResult(
            state,
            params,
            None,
            logz,
            info,
            count,
            valid & (_sampler_state(state).tempering_param >= 1.0),
            key,
        )

    def _verbose_scan(self, f, init, xs, length, stats_fn):
        def wrapped(carry, x):
            state, y = f(carry[0], x)
            return (state,), y

        update_stats, print_fn, init_stats, print_rate = self._make_verbose_fns(
            length, stats_fn=lambda carry, y: stats_fn(carry[0], y)
        )
        (state,), y = print_scan(
            wrapped,
            (init,),
            init_stats,
            xs,
            length,
            update_stats=update_stats,
            print_fn=print_fn,
            print_rate=print_rate,
        )
        return state, y

    def run(self, key, state, tempering_params, params, *, initial_log_evidence=None):
        if self.verbose:
            return _run_fixed(
                key,
                self.kernel,
                state,
                tempering_params,
                params,
                collect=self.collect,
                verbose=self,
                initial_log_evidence=initial_log_evidence,
            )
        return self.run_kernel(
            key,
            self.kernel,
            state,
            tempering_params,
            params,
            collect=self.collect,
            initial_log_evidence=initial_log_evidence,
        )

    def sample(
        self, key, state, tempering_params, params, *, initial_log_evidence=None
    ):
        if not self.verbose:
            return self.run_kernel(
                key,
                self.kernel,
                state,
                tempering_params,
                params,
                collect=True,
                initial_log_evidence=initial_log_evidence,
            )
        return _run_fixed(
            key,
            self.kernel,
            state,
            tempering_params,
            params,
            collect=True,
            verbose=self if self.verbose else None,
            initial_log_evidence=initial_log_evidence,
        )

    def adapt(
        self,
        key,
        adaptor,
        state,
        tempering_params,
        params,
        *,
        initial_log_evidence=None,
    ):
        if not self.verbose:
            return self.adapt_kernel(
                key,
                self.kernel,
                adaptor,
                state,
                tempering_params,
                params,
                collect=self.collect,
                initial_log_evidence=initial_log_evidence,
            )
        return _run_fixed(
            key,
            self.kernel,
            state,
            tempering_params,
            params,
            collect=self.collect,
            adaptor=adaptor,
            verbose=self if self.verbose else None,
            initial_log_evidence=initial_log_evidence,
        )

    def run_adaptive(
        self,
        key,
        state,
        params,
        *,
        max_steps=200,
        adaptor=None,
        initial_log_evidence=None,
    ):
        return self.run_adaptive_kernel(
            key,
            self.kernel,
            state,
            params,
            max_steps=max_steps,
            adaptor=adaptor,
            initial_log_evidence=initial_log_evidence,
        )
