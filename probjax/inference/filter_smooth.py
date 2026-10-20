from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from probjax.inference.base import RunnerMixin
from probjax.inference.filtering.base import (
    FilterInfo,
    FilterKernel,
    FilterState,
    extract_covariance,
    extract_mean,
    extract_pred_covariance,
    is_particle_state,
)
from probjax.inference.filtering.smoothing import (
    particle_smoother,
)
from probjax.inference.filtering.smoothing import (
    smooth as smooth_gaussian,
)
from probjax.utils.guards import has_attrs
from probjax.utils.jaxutils import (
    leading_dim,
    nested_checkpoint_scan,
)
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey


class FilteringTrace(NamedTuple):
    """Trace of a filtering run.

    Attributes:
        ts: Time grid of shape (T,).
        initial_state: Initial filter state.
        states: Filter states at each time step after the initial.
        infos: Filter info at each time step.
        outputs: Unpacked outputs from each step (via unpack_fn).
        obs_mask: Boolean mask indicating which times had observations.
    """

    ts: Array
    initial_state: FilterState
    states: Any
    infos: Any
    outputs: Any
    obs_mask: Array

    def get_mean(self) -> Array:
        """Extract state means from the trace.

        Returns:
            Array of shape (T-1, state_dim) for Gaussian filters,
            or (T-1, num_particles, state_dim) for particle filters.
        """
        return extract_mean(self.states)

    def get_cov(self) -> Array:
        """Extract state covariances from the trace.

        Returns:
            Array of shape (T-1, state_dim, state_dim) for Gaussian filters.
            For particle filters, computes empirical covariance.
        """
        if hasattr(self.states, "cov"):
            return self.states.cov
        elif is_particle_state(self.states):
            # Compute weighted empirical covariance for particles
            particles = self.states.particles  # (T, N, D)
            log_weights = self.states.log_weights  # (T, N)
            weights = jnp.exp(log_weights)
            weights = weights / weights.sum(axis=-1, keepdims=True)

            # Weighted mean
            mean = jnp.sum(weights[..., None] * particles, axis=-2)  # (T, D)

            # Weighted covariance
            diff = particles - mean[:, None, :]  # (T, N, D)
            cov = jnp.sum(
                weights[..., None, None] * diff[..., None, :] * diff[..., None], axis=-3
            )  # (T, D, D)
            return cov
        else:
            raise ValueError("Cannot extract covariance from trace states.")

    def get_var(self) -> Array:
        """Extract state variances (diagonal of covariance) from the trace.

        Returns:
            Array of shape (T-1, state_dim).
        """
        cov = self.get_cov()
        if cov.ndim == 2:
            # Single state - return diagonal
            return jnp.diag(cov)
        else:
            # Batch of states - extract diagonal of each covariance matrix
            return jax.vmap(jnp.diagonal)(cov)

    def get_std(self) -> Array:
        """Extract state standard deviations from the trace.

        Returns:
            Array of shape (T-1, state_dim).
        """
        return jnp.sqrt(self.get_var())

    def get_log_likelihood(self) -> Array:
        """Extract total log-likelihood from the trace.

        Returns:
            Scalar sum of log-likelihoods from all observation steps.
        """
        log_likelihood = getattr(self.infos, "log_likelihood", None)
        if log_likelihood is None:
            raise ValueError("Trace infos have no 'log_likelihood' attribute.")
        return jnp.sum(log_likelihood)


def _extract_covariance(states: Any) -> Array:
    """Back-compat alias for :func:`extract_covariance`."""
    return extract_covariance(states)


def _extract_pred_covariance(infos: Any) -> Array:
    """Back-compat alias for :func:`extract_pred_covariance`."""
    return extract_pred_covariance(infos)


def _trace_ts(trace: FilteringTrace) -> Array:
    num_states = leading_dim(trace.states)
    if trace.ts.shape[0] == num_states + 1:
        return trace.ts[1:]
    return trace.ts


def _unpack_log_likelihood(state: FilterState, info: FilterInfo) -> Array:
    log_likelihood = getattr(info, "log_likelihood", None)
    if log_likelihood is None:
        return jnp.array(0.0)
    return log_likelihood


class Filter(RunnerMixin):
    """Run a filter kernel, optionally displaying a progress bar.

    Supports two API styles:

    **Functional API (frozen=True, default):**
    All parameters passed explicitly to each method. JIT-compatible.
    >>> kernel = kalman_filter(transition_model, observation_model)
    >>> filt = Filter(kernel)  # frozen=True by default
    >>> trace = filt.filter(key, ts, observations, mu0, cov0, obs_mask=mask)

    **Object-Oriented API (frozen=False):**
    Parameters stored internally for convenience. Not JIT-compatible.
    >>> filt = Filter(kernel, frozen=False).set_params(mu0=mu0, cov0=cov0)
    >>> trace = filt.filter(key, ts, observations, obs_mask=mask)  # mu0, cov0 from self

    Args:
        kernel: A :class:`FilterKernel` or a callable that returns one (for ``fit()``).
        verbose: If ``True``, display a progress bar with log-likelihood.
        frozen: If ``True`` (default), use functional API with explicit args.
            If ``False``, use OO API with stored params via ``set_params()``.
    """

    _default_tracked_stats = ("log_likelihood",)

    def __init__(
        self,
        kernel: FilterKernel | Callable[[Any], FilterKernel],
        verbose: bool = False,
        frozen: bool = True,
        tracked_stats: tuple[str, ...] | None = None,
    ) -> None:
        self.kernel = kernel
        self.verbose = verbose
        self.frozen = frozen
        self._stored_params: dict[str, Any] = {}
        self.tracked_stats = tracked_stats or self._default_tracked_stats

    def set_params(self, **kwargs: Any) -> Filter:
        """Store parameters for the OO API (when frozen=False).

        Returns self for method chaining.

        Example:
            >>> filt = Filter(kernel, frozen=False).set_params(mu0=mu0, cov0=cov0)
        """
        self._stored_params.update(kwargs)
        return self

    def _kernel(self) -> FilterKernel:
        """The filter kernel.

        ``fit`` accepts a kernel *factory* instead of a kernel, so the two
        roles are told apart here rather than at every call site.
        """
        kernel = self.kernel
        if not isinstance(kernel, FilterKernel):
            raise TypeError(
                "This Filter holds a kernel factory; use fit() to optimize it."
            )
        return kernel

    def step(
        self,
        state: FilterState,
        t: ArrayLike,
        observed: ArrayLike | None = None,
        rng_key: RngKey | None = None,
    ) -> tuple[FilterState, FilterInfo]:
        """Execute a single filter step.

        This is a pure function that runs one step of the filter kernel.
        Useful for debugging, custom loops, or step-by-step filtering.

        Args:
            state: Current filter state.
            t: Current time.
            observed: Observation at this time (None for predict-only steps).
            rng_key: Random key (required for particle filters).

        Returns:
            Tuple of (next_state, info).
        """
        return self._kernel()(state, t=t, observed=observed, rng_key=rng_key)

    def filter(
        self,
        key: RngKey,
        ts: ArrayLike,
        observations: ArrayLike,
        *args: Any,
        obs_mask: ArrayLike | None = None,
        unpack_fn: Callable[[Any, Any], Any] | None = None,
        checkpoint_lengths: Sequence[int] | None = None,
        unroll: int = 1,
        **kwargs: Any,
    ) -> FilteringTrace:
        """Run filtering over the time grid.

        The time grid `ts` includes all time points. Observations are provided
        at all time points via `observations`, with `obs_mask` indicating which
        times actually have observations (True = has observation, False = no observation).

        For times without observations, the filter performs a predict-only step.

        **Functional API (frozen=True, default):**
        Pass ``mu0``, ``cov0`` as positional args or via ``**kwargs``.

        **OO API (frozen=False):**
        Store params via ``set_params(mu0=..., cov0=...)`` first.
        They will be retrieved automatically.

        Args:
            key: PRNG key.
            ts: Time grid of shape ``(T,)``. Must include all times, both with and
                without observations.
            observations: Observation values of shape ``(T, ...)`` or ``(T,)``.
                Values at times where ``obs_mask`` is False are ignored.
            *args: Positional args forwarded to ``kernel.init``.
                For frozen=False, these can be omitted if stored via ``set_params()``.
            obs_mask: Boolean mask of shape ``(T,)`` indicating which times have
                observations. If None, assumes all times have observations.
            unpack_fn: Optional extractor applied to ``(state, info)`` each step.
                Defaults to ``kernel.default_unpack``.
            checkpoint_lengths: If set, use ``nested_checkpoint_scan``
                with these chunk lengths.
            unroll: Unroll factor for the scan.
            **kwargs: Keyword args forwarded to ``kernel.init``.

        Returns:
            FilteringTrace containing times, initial state, states, infos,
            outputs, and observation mask.
        """
        # Handle OO API: retrieve stored params if frozen=False and args not provided
        if not self.frozen and len(args) == 0:
            # Try to get mu0, cov0 from stored params
            # These are passed as positional args to kernel.init
            stored_args = []
            if "mu0" in self._stored_params:
                stored_args.append(self._stored_params["mu0"])
            if "cov0" in self._stored_params:
                stored_args.append(self._stored_params["cov0"])
            args = tuple(stored_args)

        kernel = self._kernel()
        ts = jnp.asarray(ts)
        observations = jnp.asarray(observations)
        initial_state = kernel.init(*args, t=ts[0], **kwargs)

        unpack = unpack_fn if unpack_fn is not None else kernel.default_unpack

        # Default: all times have observations
        if obs_mask is None:
            obs_mask = jnp.ones(ts.shape[0], dtype=bool)

        # Ensure obs_mask is boolean array
        obs_mask = jnp.asarray(obs_mask, dtype=bool)

        def scan_fn(carry, scan_in):
            state, key = carry
            t, has_obs, obs = scan_in
            key, subkey = jax.random.split(key)

            def update_fn(subkey, state, obs):
                state, info = kernel(state, t=t, observed=obs, rng_key=subkey)
                return state, info

            def predict_fn(subkey, state, obs):
                state, info = kernel(state, t=t, rng_key=subkey)
                return state, info

            state, info = jax.lax.cond(
                has_obs, update_fn, predict_fn, subkey, state, obs
            )
            out = unpack(state, info)
            return (state, key), (state, info, out)

        carry = (initial_state, key)
        scan_in = (ts[1:], obs_mask[1:], observations[1:])
        num_steps = ts[1:].shape[0]

        if self.verbose:
            # The progress bar needs a tuple carry, so the filter's
            # ``(state, key)`` carry is wrapped in a one-element tuple.
            _, (states, infos, output) = self._verbose_scan(
                lambda carry, x: scan_fn(carry, x),
                carry,
                scan_in,
                num_steps,
                stats_fn=lambda _carry, y: self._extract_stats(y[0], y[1]),
                unroll=unroll,
            )
        elif checkpoint_lengths is None:
            _, (states, infos, output) = jax.lax.scan(
                scan_fn, carry, scan_in, unroll=unroll
            )
        else:
            _, (states, infos, output) = nested_checkpoint_scan(
                scan_fn,
                carry,
                scan_in,
                nested_lengths=checkpoint_lengths,
                unroll=unroll,
            )

        return FilteringTrace(
            ts=ts,
            initial_state=initial_state,
            states=states,
            infos=infos,
            outputs=output,
            obs_mask=obs_mask,
        )

    def smooth(
        self,
        trace: FilteringTrace,
        smoother: Callable[..., Any] | None = None,
        key: RngKey | None = None,
        transition_logdensity_fn: Callable[..., Any] | None = None,
    ) -> tuple[Array, Array] | tuple[PyTree, Array]:
        """Smooth a filtering trace.

        Auto-detects the appropriate smoothing algorithm based on trace content:
        - Gaussian filters (KF/EKF/UKF/SqKF): Uses RTS-style smoothing
        - Particle filters: Uses FFBSi (Forward Filter-Backward Simulator)

        For Gaussian filters, provide ``smoother`` callback.
        For particle filters, provide ``key`` and ``transition_logdensity_fn``.

        Args:
            trace: FilteringTrace from :meth:`filter`.
            smoother: RTS-style smoothing callback (Gaussian filters).
                Signature: ``(t0, t1, mu0_s, cov0_s, mu0, cov0, mu0_, cov0_) -> (mu1, cov1)``.
            key: PRNG key for backward simulation (particle filters).
            transition_logdensity_fn: Log-transition density for particle filters.
                Signature: ``(x_tp1, x_t, t, tp1) -> scalar``.

        Returns:
            Gaussian: ``(mus_s, covs_s)`` arrays.
            Particle: ``(smoothed_particles, smoothed_log_weights)`` arrays.

        Raises:
            ValueError: If trace type cannot be determined or required arguments
                are missing for the detected type.
        """
        states = trace.states

        # Auto-detect: check if this is a particle filter trace
        if is_particle_state(states):
            # Particle filter: use FFBSi
            if key is None:
                raise ValueError(
                    "Particle filter trace detected. "
                    "Please provide `key` for FFBSi smoothing."
                )
            if transition_logdensity_fn is None:
                raise ValueError(
                    "Particle filter trace detected. "
                    "Please provide `transition_logdensity_fn` for FFBSi smoothing. "
                    "Signature: (x_tp1, x_t, t, tp1) -> scalar log-density."
                )
            ts = _trace_ts(trace)
            ancestors = (
                trace.infos.ancestors if has_attrs(trace.infos, "ancestors") else None
            )
            return particle_smoother(
                key,
                ts,
                states.particles,
                states.log_weights,
                transition_logdensity_fn,
                ancestors=ancestors,
            )
        else:
            # Gaussian filter: use RTS smoothing
            if smoother is None:
                raise ValueError(
                    "Gaussian filter trace detected. "
                    "Please provide `smoother` callback for RTS smoothing. "
                    "Example: `partial(rauch_tung_stribel_smoother, transition_matrix_fn)`."
                )
            ts = _trace_ts(trace)
            mus = states.mean
            covs = _extract_covariance(states)
            mus_pred = trace.infos.mean_pred
            covs_pred = _extract_pred_covariance(trace.infos)
            return smooth_gaussian(ts, mus, covs, mus_pred, covs_pred, smoother)

    def log_likelihood(
        self,
        key: RngKey,
        ts: ArrayLike,
        observations: ArrayLike,
        *args: Any,
        obs_mask: ArrayLike | None = None,
        checkpoint_lengths: Sequence[int] | None = None,
        unroll: int = 1,
        **kwargs: Any,
    ) -> Array:
        """Run filtering and return the summed log-likelihood.

        Args:
            key: PRNG key.
            ts: Time grid of shape ``(T,)``.
            observations: Observation values of shape ``(T, ...)`` or ``(T,)``.
            *args: Positional args forwarded to ``kernel.init``.
            obs_mask: Boolean mask of shape ``(T,)`` indicating which times have
                observations. If None, assumes all times have observations.
            checkpoint_lengths: If set, use ``nested_checkpoint_scan``.
            unroll: Unroll factor for the scan.
            **kwargs: Keyword args forwarded to ``kernel.init``.

        Returns:
            Scalar log-likelihood (sum of log-likelihoods at observation times).
        """
        trace = self.filter(
            key,
            ts,
            observations,
            *args,
            obs_mask=obs_mask,
            unpack_fn=_unpack_log_likelihood,
            checkpoint_lengths=checkpoint_lengths,
            unroll=unroll,
            **kwargs,
        )
        return trace.get_log_likelihood()

    def fit(
        self,
        key: RngKey,
        ts: ArrayLike,
        observations: ArrayLike,
        *args: Any,
        obs_mask: ArrayLike | None = None,
        params_init: dict[str, Any] | None = None,
        num_steps: int = 100,
        learning_rate: float = 0.01,
        checkpoint_lengths: Sequence[int] | None = None,
        unroll: int = 1,
        **kwargs: Any,
    ) -> tuple[Any, Array]:
        """Estimate parameters by maximizing the log-likelihood.

        This method optimizes model parameters (e.g., noise covariances,
        transition matrices) to maximize the filtering log-likelihood.
        Uses Adam optimizer by default.

        The kernel must be a **parameterized kernel factory** — a callable that
        takes a parameter dict and returns a FilterKernel.

        Args:
            key: PRNG key.
            ts: Time grid of shape ``(T,)``.
            observations: Observation values of shape ``(T, ...)`` or ``(T,)``.
            *args: Additional positional args forwarded to ``kernel.init``.
            obs_mask: Boolean mask of shape ``(T,)`` indicating which times have
                observations. If None, assumes all times have observations.
            params_init: Initial parameter values as a dict. Keys and values
                depend on your kernel factory. Common: ``{'log_Q': ..., 'log_R': ...}``.
            num_steps: Number of optimization steps.
            learning_rate: Adam learning rate.
            checkpoint_lengths: If set, use ``nested_checkpoint_scan``.
            unroll: Unroll factor for the scan.
            **kwargs: Additional keyword args forwarded to ``kernel.init``.

        Returns:
            Tuple of (optimized_params, final_log_likelihood).

        Example:
            >>> def make_kernel(params):
            ...     Q = jnp.exp(params['log_Q'])
            ...     R = jnp.exp(params['log_R'])
            ...     # ... create kernel with Q, R
            ...     return kalman_filter(transition_model, observation_model)
            >>>
            >>> filt = Filter(make_kernel)
            >>> params, ll = filt.fit(
            ...     key, ts, obs, mu0, cov0, params_init={'log_Q': 0.0, 'log_R': 0.0}
            ... )
        """
        if params_init is None:
            raise ValueError("Must provide params_init dict for optimization.")

        # The kernel must be a factory function
        kernel_factory = self.kernel
        if isinstance(kernel_factory, FilterKernel):
            raise TypeError(
                "fit() needs a kernel factory that takes the parameter dict, "
                "not an already built kernel."
            )

        def objective(params):
            """Negative log-likelihood (to minimize)."""
            kernel = kernel_factory(params)
            filt = Filter(kernel, verbose=False)
            ll = filt.log_likelihood(
                key,
                ts,
                observations,
                *args,
                obs_mask=obs_mask,
                checkpoint_lengths=checkpoint_lengths,
                unroll=unroll,
                **kwargs,
            )
            return -ll  # minimize negative log-likelihood

        # Setup optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params_init)

        # Optimization loop
        def step_fn(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(objective)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params_final, _), losses = jax.lax.scan(
            step_fn, (params_init, opt_state), None, length=num_steps
        )

        final_ll = -objective(params_final)
        return params_final, final_ll
