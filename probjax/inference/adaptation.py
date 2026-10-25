from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from probjax.inference.base import Adaptor, Density
from probjax.inference.smc.base import _ensure_param_batch
from probjax.utils import guards as _guards
from probjax.utils.guards import params_to_dict

__all__ = [
    "FitState",
    "compose_adaptors",
    "fit_adaptor",
    "get_param",
    "maximum_likelihood",
    "pooled",
    "replace_params",
]


def get_param(params: Any, name: str) -> Any:
    """Read ``params[name]`` for mappings, else ``params.<name>``."""
    return _guards.get_field(params, name)


def replace_params(params: Any, **updates: Any) -> Any:
    """Replace fields in a NamedTuple, dataclass, or mapping parameter PyTree."""
    return _guards.replace_field(params, **updates)


def compose_adaptors(**adaptors: Adaptor) -> Adaptor:
    """Compose adaptors, threading parameter updates in declaration order."""
    if not adaptors:
        raise ValueError("At least one adaptor is required")
    names = tuple(adaptors)
    children = tuple(adaptors.values())

    def init(state: Any, params: Any) -> tuple[Any, ...]:
        return tuple(child.init(state, params) for child in children)

    def update(
        state: Any, info: Any, adaptor_states: tuple[Any, ...], params: Any
    ) -> tuple[tuple[Any, ...], Any, dict[str, Any]]:
        new_states = []
        child_info = {}
        for name, child, child_state in zip(
            names, children, adaptor_states, strict=True
        ):
            child_state, params, info_out = child.update(
                state, info, child_state, params
            )
            new_states.append(child_state)
            child_info[name] = info_out
        return tuple(new_states), params, child_info

    def finalize(
        adaptor_states: tuple[Any, ...], params: Any
    ) -> tuple[Any, dict[str, Any]]:
        child_info = {}
        for name, child, child_state in zip(
            names, children, adaptor_states, strict=True
        ):
            params, info_out = child.finalize(child_state, params)
            child_info[name] = info_out
        return params, child_info

    return Adaptor(init, update, finalize)


def _stateless_adaptor(tune_fn: Callable[..., dict[str, Any]]) -> Adaptor:
    """Adaptor that retunes shared-batched params from each transition.

    ``tune_fn(params_dict, state, info)`` computes the tuned parameter dict;
    init/finalize are no-ops. Used by the SMC tuning adaptors.
    """

    def init(_state: Any, _params: Any) -> tuple[Any, ...]:
        return ()

    def update(
        state: Any, info: Any, adaptor_state: Any, params: Any
    ) -> tuple[Any, Any, None]:
        tuned = tune_fn(params_to_dict(params), state, info)
        return adaptor_state, _ensure_param_batch(tuned, shared=True), None

    def finalize(_adaptor_state: Any, params: Any) -> tuple[Any, None]:
        return params, None

    return Adaptor(init, update, finalize)


def pooled(adaptor: Adaptor, axis_name: str = "chain") -> Adaptor:
    """Share one adaptation across chains run under ``jax.vmap``.

    Run the chains with ``jax.vmap(..., axis_name=axis_name)``. After every
    update each chain's floating-point adaptor statistics and parameters are
    replaced by their mean over chains, so all chains adapt as one. For dual
    averaging this equals adapting on the mean acceptance rate. Welford
    estimates measure deviations from the pooled mean, so they include the
    spread between chains; they start slightly low (the bias shrinks like
    ``log(n) / n`` after ``n`` updates).

    Do not pool a :func:`fit_adaptor`: averaging would mix the chains'
    buffered points. Give it ``axis_name`` instead, which averages gradients.
    """

    def mean_over_chains(tree: Any) -> Any:
        def mean(value):
            if jnp.issubdtype(jnp.result_type(value), jnp.inexact):
                return jax.lax.pmean(value, axis_name)
            return value

        return jax.tree.map(mean, tree)

    def update(
        state: Any, info: Any, adaptor_state: Any, params: Any
    ) -> tuple[Any, Any, Any]:
        adaptor_state, params, adaptor_info = adaptor.update(
            state, info, adaptor_state, params
        )
        return mean_over_chains(adaptor_state), mean_over_chains(params), adaptor_info

    return Adaptor(adaptor.init, update, adaptor.finalize)


class FitState(NamedTuple):
    """State of :func:`fit_adaptor`: a ring buffer and the optimizer."""

    points: Any
    weights: Any
    num_seen: Any
    num_updates: Any
    opt_state: Any
    loss: Any


def _position_data(state: Any, info: Any) -> tuple[Any, Any]:
    del info
    points = jax.tree.map(lambda leaf: leaf[None], state.position)
    return points, jnp.ones((1,))


def fit_adaptor(
    loss_fn: Callable[[Any, Any, Any], Any],
    optimizer: optax.GradientTransformation,
    *,
    field: str | None = None,
    data_fn: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    buffer_size: int = 4096,
    batch_size: int | None = 256,
    every: int = 1,
    num_fit_steps: int = 1,
    axis_name: str | None = None,
    seed: int = 0,
) -> Adaptor:
    """Train learned parameters on the chain's own output, inside the scan.

    Each transition appends ``data_fn(state, info) -> (points, weights)`` to a
    ring buffer of ``buffer_size`` points shaped like ``state.position``.
    Every ``every`` transitions the adaptor takes ``num_fit_steps`` optimizer
    steps on ``loss_fn(params, points, weights)`` over minibatches of
    ``batch_size`` buffered points (``None``: the whole buffer). ``every=1,
    num_fit_steps=1`` trains a little after every transition;
    ``every=200, num_fit_steps=200`` alternates sampling and fitting blocks.
    Both run fully compiled.

    Args:
        loss_fn: the training loss, e.g. :func:`maximum_likelihood`.
        optimizer: an optax transformation.
        field: the parameter field to train; ``None`` trains all of ``params``.
        data_fn: what to store per transition; defaults to the new position
            with weight one.
        axis_name: average gradients over this ``jax.vmap`` axis, so chains
            run in parallel train one shared set of parameters.
        seed: seeds the minibatch draws (adaptors receive no key).
    """
    if data_fn is None:
        data_fn = _position_data

    def read(params: Any) -> Any:
        return params if field is None else get_param(params, field)

    def write(params: Any, value: Any) -> Any:
        return value if field is None else replace_params(params, **{field: value})

    def init(state: Any, params: Any) -> FitState:
        points = jax.tree.map(
            lambda leaf: jnp.zeros(
                (buffer_size,) + jnp.shape(leaf), jnp.result_type(leaf)
            ),
            state.position,
        )
        return FitState(
            points,
            jnp.zeros((buffer_size,)),
            jnp.zeros((), jnp.int32),
            jnp.zeros((), jnp.int32),
            optimizer.init(read(params)),
            jnp.full((), jnp.nan),
        )

    def fit(key: Any, fit_state: FitState, trained: Any) -> tuple[Any, Any, Any]:
        num_filled = jnp.minimum(fit_state.num_seen, buffer_size)

        def one_step(carry, step_key):
            trained, opt_state = carry
            if batch_size is None:
                points, weights = fit_state.points, fit_state.weights
            else:
                index = jax.random.randint(step_key, (batch_size,), 0, num_filled)
                points = jax.tree.map(lambda leaf: leaf[index], fit_state.points)
                weights = fit_state.weights[index]
            loss, grads = jax.value_and_grad(loss_fn)(trained, points, weights)
            if axis_name is not None:
                loss, grads = jax.lax.pmean((loss, grads), axis_name)
            updates, opt_state = optimizer.update(grads, opt_state, trained)
            return (optax.apply_updates(trained, updates), opt_state), loss

        (trained, opt_state), losses = jax.lax.scan(
            one_step,
            (trained, fit_state.opt_state),
            jax.random.split(key, num_fit_steps),
        )
        return trained, opt_state, losses[-1]

    def update(
        state: Any, info: Any, fit_state: FitState, params: Any
    ) -> tuple[FitState, Any, Any]:
        points, weights = data_fn(state, info)
        slots = (fit_state.num_seen + jnp.arange(weights.shape[0])) % buffer_size
        fit_state = fit_state._replace(
            points=jax.tree.map(
                lambda buffer, new: buffer.at[slots].set(new), fit_state.points, points
            ),
            weights=fit_state.weights.at[slots].set(weights),
            num_seen=fit_state.num_seen + weights.shape[0],
            num_updates=fit_state.num_updates + 1,
        )
        # Wait for a full batch: fitting a flow to a few copies of the
        # starting point collapses it onto that point.
        do_fit = (fit_state.num_updates % every == 0) & (
            fit_state.num_seen >= (batch_size or 1)
        )
        key = jax.random.fold_in(jax.random.key(seed), fit_state.num_updates)
        trained, opt_state, loss = jax.lax.cond(
            do_fit,
            lambda: fit(key, fit_state, read(params)),
            lambda: (read(params), fit_state.opt_state, fit_state.loss),
        )
        fit_state = fit_state._replace(opt_state=opt_state, loss=loss)
        return fit_state, write(params, trained), loss

    def finalize(fit_state: FitState, params: Any) -> tuple[Any, Any]:
        return params, fit_state.loss

    return Adaptor(init, update, finalize)


def maximum_likelihood(density: Density) -> Callable[[Any, Any, Any], Any]:
    """Weighted negative log-likelihood of ``density``, for :func:`fit_adaptor`."""

    def loss(params: Any, points: Any, weights: Any) -> Any:
        log_q = jax.vmap(density.logpdf, in_axes=(0, None))(points, params)
        # Empty buffer slots carry weight zero; keep their -inf out of the sum.
        weighted = jnp.where(weights > 0, weights * log_q, 0.0)
        return -jnp.sum(weighted) / jnp.maximum(jnp.sum(weights), 1e-12)

    return loss
