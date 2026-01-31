from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from probjax.utils.protocols import (
    LossFn,
    ModelFn,
    ReductionFn,
    TimeDependentModelFn,
    WeightFn,
)

__all__ = ["build_denoising_loss", "build_time_dependent_denoising_loss"]


def _validate_prediction_target(prediction_target: str) -> str:
    valid_targets = {"eps", "v", "x0"}
    if prediction_target not in valid_targets:
        raise ValueError(
            f"Invalid prediction target '{prediction_target}'. "
            f"Supported targets are: {', '.join(sorted(valid_targets))}."
        )
    return prediction_target


def _compute_target(
    prediction_target: str, *, x0: Array, eps: Array, scale: ArrayLike, std: ArrayLike
) -> Array:
    if prediction_target == "x0":
        return x0
    if prediction_target == "eps":
        return eps
    if prediction_target == "v":
        alpha = jnp.asarray(scale)
        sigma = jnp.asarray(std)
        total_variance = jnp.sqrt(alpha**2 + sigma**2)
        normalized_alpha = alpha / total_variance
        normalized_sigma = sigma / total_variance
        return normalized_alpha * eps - normalized_sigma * x0
    raise ValueError(f"Unsupported prediction target: {prediction_target}")


def _finalize_loss(
    loss: Array,
    *,
    weight: Optional[ArrayLike],
    loss_mask: Optional[ArrayLike],
    axis: int | tuple[int, ...] | None,
    adaptive_weight_p: float,
    adaptive_weight_eps: float,
) -> Array:
    if loss_mask is not None:
        loss = jnp.where(~loss_mask, loss, jnp.zeros_like(loss))
    if weight is not None:
        loss = loss * weight
    loss = jnp.sum(loss, axis=axis)
    if adaptive_weight_p > 0:
        adaptive_weight = jax.lax.stop_gradient(
            1 / (loss + adaptive_weight_eps) ** adaptive_weight_p
        )
        loss = loss * adaptive_weight
    return loss


def _compute_prediction_loss(
    model_fn: ModelFn | TimeDependentModelFn,
    *,
    prediction_target: str,
    args_with_noisy: tuple,
    model_kwargs: dict,
    x0: Array,
    eps: Array,
    scale: ArrayLike,
    std: ArrayLike,
    weight: Optional[ArrayLike],
    loss_mask: Optional[ArrayLike],
    axis: int | tuple[int, ...] | None,
    adaptive_weight_p: float,
    adaptive_weight_eps: float,
) -> Array:
    prediction = model_fn(*args_with_noisy, **model_kwargs)
    target = _compute_target(prediction_target, x0=x0, eps=eps, scale=scale, std=std)
    loss = (prediction - target) ** 2
    return _finalize_loss(
        loss,
        weight=weight,
        loss_mask=loss_mask,
        axis=axis,
        adaptive_weight_p=adaptive_weight_p,
        adaptive_weight_eps=adaptive_weight_eps,
    )


def build_denoising_loss(
    model_fn: ModelFn,
    scale: ArrayLike,
    std: ArrayLike,
    weight: Optional[ArrayLike] = None,
    argnums: int = 0,
    axis: int | tuple[int, ...] | None = -1,
    reduction_fn: ReductionFn = jnp.mean,
    prediction_target: str = "x0",
) -> LossFn:
    """Build a denoising loss function with configurable prediction targets.

    Args:
        model_fn: Callable that predicts the target quantity from noisy inputs.
        scale: Scaling factor used when ``prediction_target`` is ``"v"``.
        std: Standard deviation of the perturbation noise.
        weight: Optional multiplicative weight applied before reduction.
        adaptive_weight_p: Power for adaptive re-weighting; set to 0.0 to disable.
        adaptive_weight_eps: Stabiliser added before adaptive re-weighting.
        argnums: Index of the argument corresponding to the clean sample.
        axis: Axis (or tuple of axes) reduced after computing element-wise losses.
        reduction_fn: Function applied to the batch of losses.
        prediction_target: One of ``"x0"``, ``"eps"``, or ``"v"``.
        **extra_kwargs: Captures legacy keyword arguments (e.g. ``addaptive_weight_p``).

    Returns:
        A callable loss function accepting the same positional arguments as ``model_fn``.
    """
    prediction_target = _validate_prediction_target(prediction_target)

    scale_array = jnp.asarray(scale)
    std_array = jnp.asarray(std)

    def loss_fn(
        *args,
        rng=None,
        loss_mask=None,
        adaptive_weight_p=0.0,
        adaptive_weight_eps=1e-3,
        **kwargs,
    ):
        if rng is None:
            raise ValueError(
                "loss_fn requires an RNG key. Pass it via the 'rng' keyword."
            )
        model_kwargs = dict(kwargs)
        axis_override = model_kwargs.pop("axis", axis)

        x0 = args[argnums]
        eps = jax.random.normal(rng, shape=x0.shape)

        alpha = jnp.sqrt(1.0 - std_array**2)
        x_noisy = alpha * x0 + std_array * eps

        args_with_noisy = args[:argnums] + (x_noisy,) + args[argnums + 1 :]

        loss = _compute_prediction_loss(
            model_fn,
            prediction_target=prediction_target,
            args_with_noisy=args_with_noisy,
            model_kwargs=model_kwargs,
            x0=x0,
            eps=eps,
            scale=scale_array,
            std=std_array,
            weight=weight,
            loss_mask=loss_mask,
            axis=axis_override,
            adaptive_weight_p=adaptive_weight_p,
            adaptive_weight_eps=adaptive_weight_eps,
        )

        return reduction_fn(loss)

    return loss_fn


def build_time_dependent_denoising_loss(
    model_fn: TimeDependentModelFn,
    scale_fn: Callable[[ArrayLike], Array],
    std_fn: Callable[[ArrayLike], Array],
    weight_fn: WeightFn,
    argnums: int = 0,
    reduction_fn: ReductionFn = jnp.mean,
    prediction_target: str = "x0",
) -> LossFn:
    """Build a time-dependent denoising loss with shared prediction logic.

    Args:
        model_fn: Callable predicting the target quantity at time ``t``.
        scale_fn: Function returning the scaling factor (alpha) at time ``t``.
        std_fn: Function returning the noise standard deviation at time ``t``.
        weight_fn: Function producing weights evaluated at ``t``.
        adaptive_weight_p: Power for adaptive re-weighting; set to 0.0 to disable.
        adaptive_weight_eps: Stabiliser added before adaptive re-weighting.
        argnums: Index of the clean sample within ``*args`` (after ``t``).
        reduction_fn: Function applied to the batch of losses.
        prediction_target: One of ``"x0"``, ``"eps"``, or ``"v"``.
        **extra_kwargs: Captures legacy keyword arguments (e.g. ``addaptive_weight_p``).

    Returns:
        A callable time-dependent loss function.
    """
    prediction_target = _validate_prediction_target(prediction_target)

    def loss_fn(
        t,
        *args,
        rng=None,
        loss_mask=None,
        adaptive_weight_p=0.0,
        adaptive_weight_eps=1e-3,
        **kwargs,
    ):
        if rng is None:
            raise ValueError(
                "loss_fn requires an RNG key. Pass it via the 'rng' keyword."
            )
        model_kwargs = dict(kwargs)
        axis = model_kwargs.pop("axis", -1)

        x0 = args[argnums]
        alpha_t = jnp.asarray(scale_fn(t))
        sigma_t = jnp.asarray(std_fn(t))
        eps = jax.random.normal(rng, shape=x0.shape)

        x_noisy = alpha_t * x0 + sigma_t * eps

        args_with_noisy = (t,) + args[:argnums] + (x_noisy,) + args[argnums + 1 :]

        weight = weight_fn(t)

        loss = _compute_prediction_loss(
            model_fn,
            prediction_target=prediction_target,
            args_with_noisy=args_with_noisy,
            model_kwargs=model_kwargs,
            x0=x0,
            eps=eps,
            scale=alpha_t,
            std=sigma_t,
            weight=weight,
            loss_mask=loss_mask,
            axis=axis,
            adaptive_weight_p=adaptive_weight_p,
            adaptive_weight_eps=adaptive_weight_eps,
        )

        return reduction_fn(loss)

    return loss_fn
