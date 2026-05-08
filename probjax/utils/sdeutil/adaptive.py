"""Step-size adaptors for adaptive SDE integration.

Two flavors are exposed, distinguished by their treatment of the Brownian path
under step rejection:

- :class:`WeakStepSizeAdaptor` — draws fresh noise on each retry. Within a
  single proposed step, the increment is split into two bridge-consistent
  halves so step-doubling gives an unbiased local error estimate **for that
  step**. Across rejections the realized trajectory is no longer a sample
  from a single Brownian path — only its distributional / weak quantities
  (moments, terminal-state densities) are valid.

- :class:`StrongStepSizeAdaptor` — backed by a virtual Brownian tree
  (:func:`probjax.utils.sdeutil.brownian.brownian_tree`). Every increment is
  derived from a single keyed path, so refinements on rejection share that
  path and the realized trajectory is a sample from one consistent ``W``.
  Required for per-path quantities (filtering, parameter inference,
  sample-wise loss).

Both inherit the controller machinery from :class:`StepSizeAdaptor`
(``error_ratio`` / ``next_step_size`` / ``accept_step``) so the integrator
itself stays adaptor-agnostic.
"""

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from probjax.utils.odeutil.adaptive import StepSizeAdaptor
from probjax.utils.sdeutil.brownian import brownian_tree


class WeakBrownianState(NamedTuple):
    """Sampling state for :class:`WeakStepSizeAdaptor` — just an RNG key."""

    key: Array


class StrongBrownianState(NamedTuple):
    """Sampling state for :class:`StrongStepSizeAdaptor`.

    Stores the virtual-tree key plus the current ``(t, W(t))`` so increments
    are computed by querying ``W`` at ``t + dt`` and subtracting.
    """

    key: Array
    t0: Array
    t_end: Array
    w0: Array
    t_curr: Array
    w_curr: Array


class SDEStepSizeAdaptor(StepSizeAdaptor):
    """Base class for SDE step-size controllers.

    Extends the ODE :class:`StepSizeAdaptor` with a noise-sampling protocol:

    - :meth:`init_brownian` constructs the per-trajectory sampling state.
    - :meth:`propose_increments` produces the increment for a proposed
      ``dt`` plus the bridge-consistent halves used by step doubling.
    - :meth:`on_reject` updates the sampling state when a step is rejected
      before the next attempt.

    Subclasses set :attr:`is_path_consistent` so downstream code can decide
    whether per-path quantities are meaningful.
    """

    is_path_consistent: bool = False

    def __init__(
        self,
        *,
        max_inner_steps: int = 256,
        unroll: int = 1,
        warn_on_boundary: bool = True,
        **kwargs,
    ):
        """
        Args:
            max_inner_steps: Static cap on inner adaptive iterations between
                two output points. Drives the bounded :func:`jax.lax.scan`
                length so the loop is reverse-mode differentiable. Set
                generously: rejected steps and tight tolerances both
                consume budget.
            unroll: Static unroll factor for the inner scan. Higher values
                let XLA fuse more work per loop iteration at the cost of
                larger compiled code; default ``1`` keeps compile time low.
                Try ``2`` or ``4`` if the inner step is small relative to
                scheduling overhead.
            warn_on_boundary: If ``True`` (default), emit a host-side
                ``RuntimeWarning`` via ``jax.debug.callback`` whenever
                ``max_inner_steps`` is exhausted before reaching the next
                output time — that's a sign the budget is too small or
                tolerances are too tight.
        """
        super().__init__(**kwargs)
        self.max_inner_steps = int(max_inner_steps)
        self.unroll = int(unroll)
        self.warn_on_boundary = bool(warn_on_boundary)

    def init_brownian(self, rng, t0, t_end, noise_shape: Tuple[int, ...]):
        raise NotImplementedError

    def propose_increments(
        self, state, t_curr, dt, noise_shape: Tuple[int, ...]
    ):
        """Return ``(committed_state, dW_full, dW_first_half, dW_second_half)``.

        The two halves satisfy ``dW_first_half + dW_second_half == dW_full``
        within the proposed step (Brownian-bridge consistent), enabling an
        unbiased step-doubling local error estimate.

        ``committed_state`` is the new sampling state to keep on accept; the
        integrator passes it through :meth:`on_reject` instead on rejection.
        """
        raise NotImplementedError

    def on_reject(self, state):
        """Sampling state to use for the next retry after a rejected step."""
        return state


class WeakStepSizeAdaptor(SDEStepSizeAdaptor):
    """Free resampling on rejection — valid for distributional quantities only.

    The trajectory is *not* a sample from a single Brownian path: each rejected
    step's noise is discarded and the next attempt draws fresh randomness.
    Suitable for diffusion-model sampling or any setting where you only care
    about moments / densities of the terminal state.
    """

    is_path_consistent = False

    def init_brownian(
        self, rng, t0, t_end, noise_shape: Tuple[int, ...]
    ) -> WeakBrownianState:
        del t0, t_end, noise_shape
        return WeakBrownianState(key=rng)

    def propose_increments(
        self,
        state: WeakBrownianState,
        t_curr,
        dt,
        noise_shape: Tuple[int, ...],
    ):
        k_full, k_bridge, k_next = jax.random.split(state.key, 3)
        sqrt_dt = jnp.sqrt(jnp.abs(dt))
        dW_full = jax.random.normal(k_full, noise_shape) * sqrt_dt
        # Brownian bridge midpoint of *this* step:
        # dW_first_half = 0.5 dW_full + 0.5 sqrt(dt) Z, var = dt/2 ✓
        z = jax.random.normal(k_bridge, noise_shape)
        dW1 = 0.5 * dW_full + 0.5 * sqrt_dt * z
        dW2 = dW_full - dW1
        committed = WeakBrownianState(key=k_next)
        return committed, dW_full, dW1, dW2

    def on_reject(self, state: WeakBrownianState) -> WeakBrownianState:
        # Advance the key so the next attempt draws fresh noise.
        new_key, _ = jax.random.split(state.key)
        return WeakBrownianState(key=new_key)


class StrongStepSizeAdaptor(SDEStepSizeAdaptor):
    """Path-consistent sampling via a virtual Brownian tree.

    All increments come from a single keyed path, so rejected steps refine
    the same trajectory and the integrator's output is a sample from one
    consistent ``W``. Use this whenever per-path quantities matter (state
    estimation, log-likelihood under a given path, gradient methods that
    assume realized-path consistency).

    Args:
        tree_depth: Static refinement depth for the underlying virtual
            Brownian tree. Per-query cost is ``O(tree_depth)``. Default
            ``14`` ≈ 6e-5 leaf width on a unit interval — fine enough for
            any controller dt above ``T·2^-14``. Increase only if you set
            ``rtol``/``atol`` so tight that the controller wants steps
            below that.
        tree_tol: Backward-compatibility alias; converted to
            ``ceil(-log2(tree_tol))`` at construction time. Mutually
            exclusive with ``tree_depth``.
    """

    is_path_consistent = True

    def __init__(
        self,
        *,
        tree_depth: int = 14,
        tree_tol: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if tree_tol is not None:
            tree_depth = max(1, int(jnp.ceil(-jnp.log2(jnp.asarray(tree_tol)))))
        self.tree_depth = int(tree_depth)

    def init_brownian(
        self, rng, t0, t_end, noise_shape: Tuple[int, ...]
    ) -> StrongBrownianState:
        zeros = jnp.zeros(noise_shape)
        return StrongBrownianState(
            key=rng,
            t0=jnp.asarray(t0),
            t_end=jnp.asarray(t_end),
            w0=zeros,
            t_curr=jnp.asarray(t0),
            w_curr=zeros,
        )

    def propose_increments(
        self,
        state: StrongBrownianState,
        t_curr,
        dt,
        noise_shape: Tuple[int, ...],
    ):
        del noise_shape  # tree query infers it from ``state.w0``
        t_half = t_curr + 0.5 * dt
        t_full = t_curr + dt
        w_half = brownian_tree(
            state.key,
            t_half,
            state.t0,
            state.t_end,
            state.w0,
            depth=self.tree_depth,
        )
        w_full = brownian_tree(
            state.key,
            t_full,
            state.t0,
            state.t_end,
            state.w0,
            depth=self.tree_depth,
        )
        dW_full = w_full - state.w_curr
        dW1 = w_half - state.w_curr
        dW2 = w_full - w_half
        committed = state._replace(t_curr=t_full, w_curr=w_full)
        return committed, dW_full, dW1, dW2

    # on_reject inherits from base — tree is global, no state change needed.
