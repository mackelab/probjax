"""
Mixture Distribution (:mod:`probjax.stats.mixture`)
=================================================

This module implements mixture distributions that combine multiple component distributions
with mixing probabilities.
"""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax, random

from probjax.stats.base import (
    rv_continuous_frozen,
    rv_discrete_frozen,
    rv_frozen,
    rv_generic,
)
from probjax.stats.constraints import distribution, simplex
from probjax.utils.typing import ArrayLike, RngKey

try:
    from probjax.stats.continuous.norm import norm as _norm_distribution
except ImportError:  # pragma: no cover
    _norm_distribution = None

_NORM_GEN = _norm_distribution.__class__ if _norm_distribution is not None else None

__all__ = ["mixture"]


class mixture_frozen(rv_continuous_frozen, rv_discrete_frozen):
    """Frozen mixture distribution."""

    def __init__(self, dist, mixing_probs, components, **kwds):
        super().__init__(dist, mixing_probs=mixing_probs, components=components, **kwds)

    def _compute_batch_and_event_shape(self, mixing_probs, components, **kwds):
        """Compute the batch and event shape of the distribution."""
        batch_shape1 = mixing_probs.shape[:-1]
        num_components = mixing_probs.shape[-1]
        assert len(components) == num_components, (
            "Number of components must match number of mixing probabilities"
        )
        event_shape = components[0].event_shape
        assert all(comp.event_shape == event_shape for comp in components), (
            "All components must have the same event shape"
        )
        batch_shape2 = components[0].batch_shape
        assert all(comp.batch_shape == batch_shape2 for comp in components), (
            "All components must have the same batch shape"
        )
        batch_shape = jnp.broadcast_shapes(batch_shape1, batch_shape2)
        return batch_shape, event_shape


class mixture_gen(rv_generic):
    """A mixture distribution that combines multiple component distributions."""

    parameters = {
        "mixing_probs": simplex,
        "components": distribution,
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, mixing_probs, components, **kwargs) -> rv_frozen:
        """Create a frozen mixture distribution."""
        return self.freeze(mixing_probs=mixing_probs, components=components, **kwargs)

    def freeze(self, mixing_probs, components, **kwargs):
        """Freeze the mixture distribution with the given parameters."""
        return mixture_frozen(
            self, mixing_probs=mixing_probs, components=components, **kwargs
        )

    @classmethod
    def support(cls, mixing_probs, components, **kwds):
        """Get the support of the mixture distribution."""
        supports = [comp.support() for comp in components]
        if all(isinstance(s, tuple) and len(s) == 2 for s in supports):
            return (min(s[0] for s in supports), max(s[1] for s in supports))
        return tuple(
            set().union(*[s if isinstance(s, tuple) else (s,) for s in supports])
        )

    @classmethod
    def logpdf(cls, x: ArrayLike, mixing_probs, components, **kwds):
        """Log probability density function of the mixture distribution."""
        x = jnp.asarray(x)
        log_pdfs = jnp.stack([comp.logpdf(x) for comp in components], axis=-1)
        return jax.scipy.special.logsumexp(jnp.log(mixing_probs) + log_pdfs, axis=-1)

    @classmethod
    def cdf(cls, x: ArrayLike, mixing_probs, components, **kwds):
        """Cumulative distribution function of the mixture distribution."""
        x = jnp.asarray(x)
        cdfs = jnp.stack([comp.cdf(x) for comp in components], axis=-1)
        return jnp.sum(mixing_probs * cdfs, axis=-1)

    @classmethod
    def ppf(cls, q: ArrayLike, mixing_probs, components, **kwds):
        """Percent point function of the mixture distribution."""
        q = jnp.asarray(q)
        x0 = jnp.mean([comp.ppf(q) for comp in components], axis=0)
        return jax.scipy.optimize.root(
            lambda x: cls.cdf(x, mixing_probs, components) - q,
            x0,
        ).x

    @classmethod
    def rvs(
        cls,
        rng: RngKey,
        mixing_probs=None,
        components=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the mixture distribution."""
        key_sample, key_cluster_membership = random.split(rng, 2)

        # Sample from all components at once
        component_samples = jnp.stack(
            [comp.rvs(key_sample, shape=shape) for comp in components], axis=-1
        )

        # Sample cluster membership
        cluster_membership = random.categorical(
            key_cluster_membership,
            mixing_probs,
            shape=shape,
        )
        while cluster_membership.ndim < component_samples.ndim:
            cluster_membership = jnp.expand_dims(cluster_membership, axis=-1)
        # Select samples based on cluster membership
        samples = jnp.take_along_axis(component_samples, cluster_membership, axis=-1)

        return jnp.squeeze(samples, axis=-1)

    @classmethod
    def mean(cls, mixing_probs, components, **kwds):
        """Mean of the mixture distribution."""
        means = jnp.stack([jnp.asarray(comp.mean()) for comp in components], axis=-1)
        return jnp.sum(mixing_probs * means, axis=-1)

    @classmethod
    def var(cls, mixing_probs, components, **kwds):
        """Variance of the mixture distribution."""
        means = jnp.stack([jnp.asarray(comp.mean()) for comp in components], axis=-1)
        vars = jnp.stack([jnp.asarray(comp.var()) for comp in components], axis=-1)
        mean = jnp.sum(mixing_probs * means, axis=-1)
        return jnp.sum(mixing_probs * (vars + (means - mean[..., None]) ** 2), axis=-1)

    @classmethod
    def mode(cls, mixing_probs, components, **kwds):
        """Mode of the mixture distribution (supports univariate mixtures)."""
        if not components:
            raise ValueError("At least one component is required to compute the mode.")

        event_shape = components[0].event_shape
        if event_shape not in ((), (1,)):
            raise NotImplementedError(
                "mixture.mode currently supports only univariate mixtures."
            )

        # Restrict implementation to mixtures of univariate Normal components for now.
        if not (
            _NORM_GEN and all(isinstance(comp.dist, _NORM_GEN) for comp in components)
        ):
            raise NotImplementedError(
                "Mode computation currently implemented only for univariate Normal mixtures."
            )

        dtype = mixing_probs.dtype

        def log_prob(x):
            return cls.logpdf(x, mixing_probs, components, **kwds)

        candidates = []
        for comp in components:
            try:
                comp_mode = jnp.asarray(comp.mode())
                candidates.append(comp_mode.reshape(()))
            except NotImplementedError:
                try:
                    comp_mean = jnp.asarray(comp.mean())
                    candidates.append(comp_mean.reshape(()))
                except NotImplementedError:
                    pass

        try:
            mixture_mean = jnp.asarray(
                cls.mean(mixing_probs, components, **kwds)
            ).reshape(())
            candidates.append(mixture_mean)
        except NotImplementedError:
            pass

        if not candidates:
            raise NotImplementedError(
                "Unable to construct candidate modes for the mixture."
            )

        candidates_arr = jnp.stack([jnp.asarray(c, dtype=dtype) for c in candidates])
        candidate_logp = log_prob(candidates_arr)
        best_idx = jnp.argmax(candidate_logp)
        best_candidate = candidates_arr[best_idx]

        # Discrete mixtures: return the best candidate directly.
        if hasattr(components[0], "pmf") or hasattr(components[0], "logpmf"):
            return best_candidate

        # Attempt a simple grid search around the mixture mean if variance is finite.
        try:
            mixture_var = jnp.asarray(
                cls.var(mixing_probs, components, **kwds)
            ).reshape(())
        except NotImplementedError:
            mixture_var = jnp.asarray(jnp.nan, dtype=dtype)

        finite_var = jnp.isfinite(mixture_var) & (mixture_var > 0)
        std = jnp.sqrt(jnp.maximum(mixture_var, jnp.asarray(1e-12, dtype=dtype)))
        span = jnp.asarray(5.0, dtype=dtype) * std
        all_points = jnp.concatenate([candidates_arr, jnp.array([best_candidate])])
        low = jnp.min(all_points)
        high = jnp.max(all_points)
        try:
            mean_val = jnp.asarray(
                cls.mean(mixing_probs, components, **kwds)
            ).reshape(())
        except NotImplementedError:
            mean_val = best_candidate

        if bool(finite_var):
            low = jnp.minimum(low, mean_val - span)
            high = jnp.maximum(high, mean_val + span)
        else:
            width = jnp.maximum(high - low, jnp.asarray(1.0, dtype=dtype))
            low = low - 0.5 * width
            high = high + 0.5 * width

        if not jnp.isfinite(low):
            low = best_candidate - jnp.asarray(5.0, dtype=dtype)
        if not jnp.isfinite(high):
            high = best_candidate + jnp.asarray(5.0, dtype=dtype)

        if high <= low:
            high = low + jnp.asarray(1.0, dtype=dtype)

        grid = jnp.linspace(low, high, num=512, dtype=dtype)
        grid_logp = log_prob(grid)
        grid_best_idx = jnp.argmax(grid_logp)
        grid_best = grid[grid_best_idx]

        # Local refinement around the best grid point.
        step = (high - low) / jnp.asarray(511.0, dtype=dtype)
        left = jnp.maximum(low, grid_best - 3 * step)
        right = jnp.minimum(high, grid_best + 3 * step)
        fine_grid = jnp.linspace(left, right, num=256, dtype=dtype)
        fine_logp = log_prob(fine_grid)
        fine_best = fine_grid[jnp.argmax(fine_logp)]

        return fine_best

    @classmethod
    def entropy(cls, mixing_probs, components, **kwds):
        """Entropy of the mixture distribution."""
        raise NotImplementedError("Entropy not implemented for mixture distribution")

    @classmethod
    def fit(
        cls,
        x: ArrayLike,
        components: Sequence[rv_frozen],
        mixing_probs_init: Optional[ArrayLike] = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        rng_key: Optional[RngKey] = None,
    ):
        """Fit a finite mixture model using a plain EM loop."""
        if not components:
            raise ValueError("mixture.fit requires at least one component.")
        if not all(isinstance(comp, rv_frozen) for comp in components):
            raise TypeError(
                "mixture.fit expects frozen component distributions (e.g. ``norm(loc, scale)``)."
            )

        if rng_key is None:
            rng_key = random.PRNGKey(0)

        data = jnp.asarray(x)
        event_shape = components[0].event_shape
        if any(comp.event_shape != event_shape for comp in components[1:]):
            raise ValueError("All components must share the same event shape.")
        if any(comp.batch_shape for comp in components):
            raise NotImplementedError(
                "mixture.fit does not currently support batched component parameters."
            )

        if event_shape:
            if data.ndim < len(event_shape):
                raise ValueError(
                    "Observations must have enough trailing dimensions to match the component event shape."
                )
            if tuple(data.shape[-len(event_shape) :]) != event_shape:
                raise ValueError(
                    "Trailing dimensions of the observations must match the component event shape."
                )
            data = jnp.reshape(data, (-1,) + event_shape)
        else:
            data = jnp.reshape(jnp.asarray(data), (-1,))

        if data.shape[0] == 0:
            raise ValueError("mixture.fit requires at least one observation.")

        numeric_dtype = jnp.result_type(data.dtype, jnp.float32)
        n_components = len(components)

        if mixing_probs_init is None:
            mixing_probs = jnp.full(
                (n_components,), 1.0 / n_components, dtype=numeric_dtype
            )
        else:
            mixing_probs = jnp.asarray(mixing_probs_init, dtype=numeric_dtype)
            if mixing_probs.shape != (n_components,):
                raise ValueError("mixing_probs_init must have shape (n_components,)")
            mixing_probs = jnp.clip(mixing_probs, 1e-12)
            mixing_probs = mixing_probs / jnp.sum(mixing_probs)

        component_dists = tuple(comp.dist for comp in components)
        component_params = [
            (
                tuple(comp.args),
                dict(comp.kwds),
            )
            for comp in components
        ]
        component_params = tuple(component_params)

        tol_value = jnp.asarray(tol, dtype=numeric_dtype)
        data = data.astype(numeric_dtype)
        prev_log_likelihood = jnp.asarray(-jnp.inf, dtype=numeric_dtype)

        max_iter_value = jnp.asarray(max_iter, dtype=jnp.int32)

        initial_diff = jnp.asarray(jnp.inf, dtype=numeric_dtype)

        init_state = (
            jnp.asarray(0, dtype=jnp.int32),
            mixing_probs,
            component_params,
            prev_log_likelihood,
            initial_diff,
            rng_key,
        )

        def cond_fun(state):
            iteration, _, _, _, diff_prev, _ = state
            return jnp.logical_and(iteration < max_iter_value, diff_prev > tol_value)

        def body_fun(state):
            (
                iteration,
                mixing_probs_curr,
                params_curr,
                prev_ll_curr,
                diff_curr,
                key_curr,
            ) = state
            del diff_curr

            log_pdfs = jnp.stack(
                [
                    dist.logpdf(data, *args_i, **kwds_i)
                    for dist, (args_i, kwds_i) in zip(component_dists, params_curr, strict=False)
                ],
                axis=1,
            )
            log_weights = jnp.log(jnp.clip(mixing_probs_curr, 1e-12)) + log_pdfs
            log_norm = jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
            responsibilities = jnp.exp(log_weights - log_norm)

            Nk = jnp.sum(responsibilities, axis=0)
            mixing_probs_new = jnp.clip(Nk / jnp.sum(Nk), 1e-12)
            mixing_probs_new = mixing_probs_new / jnp.sum(mixing_probs_new)

            split_keys = random.split(key_curr, n_components + 1)
            key_new = split_keys[0]
            component_keys = split_keys[1:]

            data_count = data.shape[0]
            new_params = []

            for idx, (dist, (args_i, kwds_i)) in enumerate(
                zip(component_dists, params_curr, strict=False)
            ):
                weights = jnp.asarray(responsibilities[:, idx])
                dtype = jnp.result_type(data.dtype, weights.dtype, jnp.float32)
                weights = weights.astype(dtype)
                typed_data = data.astype(dtype)
                total_weight = jnp.sum(weights)

                def coerce(value):
                    if isinstance(value, jnp.ndarray):
                        return value.astype(dtype)
                    return value

                args_typed = tuple(coerce(arg) for arg in args_i)
                kwds_typed = {k: coerce(v) for k, v in kwds_i.items()}

                def apply_params(params_out):
                    if isinstance(params_out, dict):
                        updated_args = args_typed
                        updated_kwds = dict(kwds_typed)
                        for name, value in params_out.items():
                            updated_kwds[name] = coerce(value)
                        return updated_args, updated_kwds

                    params_seq = params_out if isinstance(params_out, tuple) else (params_out,)
                    updated_args_list = list(args_typed)
                    max_pos = min(len(params_seq), len(updated_args_list))
                    for pos in range(max_pos):
                        updated_args_list[pos] = coerce(params_seq[pos])

                    updated_kwds = dict(kwds_typed)
                    remaining = params_seq[len(updated_args_list) :]
                    if remaining:
                        param_names = list(getattr(dist, "parameters", {}).keys())
                        positional = param_names[: len(updated_args_list)]
                        remaining_names = [name for name in param_names if name not in positional]
                        preferred = [name for name in remaining_names if name in updated_kwds]
                        fallback = [name for name in remaining_names if name not in updated_kwds]
                        ordered = (preferred + fallback)[: len(remaining)]
                        if len(ordered) < len(remaining):
                            extra = [
                                name
                                for name in param_names
                                if name not in positional and name not in ordered
                            ]
                            ordered += extra[: len(remaining) - len(ordered)]
                        for name, value in zip(ordered, remaining, strict=False):
                            updated_kwds[name] = coerce(value)

                    return tuple(updated_args_list), updated_kwds

                def reinit_branch(key_inner):
                    del key_inner  # PRNG key unused when returning initial parameters
                    return args_typed, kwds_typed

                def update_branch(key_inner):
                    del key_inner
                    normalised_weights = weights / jnp.asarray(
                        jnp.maximum(total_weight, jnp.asarray(1e-12, dtype=dtype)), dtype=dtype
                    )

                    try:
                        params_out = dist.fit(typed_data, weights=normalised_weights)
                    except TypeError:
                        params_out = dist.fit(typed_data)
                    return apply_params(params_out)

                params_updated = lax.cond(
                    total_weight <= jnp.asarray(1e-10, dtype=dtype),
                    reinit_branch,
                    update_branch,
                    component_keys[idx],
                )
                new_params.append(params_updated)

            params_next = tuple(new_params)

            log_likelihood = jnp.mean(log_norm)
            diff = jnp.abs(log_likelihood - prev_ll_curr)
            diff = jnp.where(
                jnp.isfinite(prev_ll_curr),
                diff,
                jnp.asarray(jnp.inf, dtype=numeric_dtype),
            )

            return (
                iteration + jnp.asarray(1, dtype=jnp.int32),
                mixing_probs_new,
                params_next,
                log_likelihood,
                diff,
                key_new,
            )

        final_state = lax.while_loop(cond_fun, body_fun, init_state)
        _, mixing_probs_final, params_final, _, _, _ = final_state

        fitted_components = [
            dist(*args, **kwds)
            for dist, (args, kwds) in zip(component_dists, params_final, strict=False)
        ]
        return mixing_probs_final, fitted_components


mixture = mixture_gen(name="mixture")
