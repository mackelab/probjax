from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array

from probjax.utils.protocols import ReductionFn

__all__ = ["build_time_dependent_multinomial_diffusion_loss"]


def _broadcast_weight_like(weight: Array, ref: Array) -> Array:
    w = jnp.asarray(weight)
    if w.shape != ref.shape:
        w = jnp.broadcast_to(w, ref.shape)
    return w


def build_time_dependent_multinomial_diffusion_loss(
    model_fn: Callable[..., Array],
    q_sample_fn: Callable[[jax.Array, Array, Array], Array],
    sample_timesteps_fn: Callable[[jax.Array, tuple[int, ...]], Array],
    q_xt_given_x0_probs_fn: Optional[Callable[[Array, Array], Array]] = None,
    alpha_bar_fn: Optional[Callable[[Array], Array]] = None,
    base_probs: Optional[ArrayLike] = None,
    rao_blackwellize_xt: bool = False,
    rao_blackwellize_xt_num_samples: int = 4,
    rao_blackwellize_xt_num_features: Optional[int] = None,
    num_classes: Optional[int] = None,
    weight_fn: Optional[Callable[[Array], Array]] = None,
    reduction_fn: ReductionFn = jnp.mean,
) -> Callable[..., Array]:
    """Build CE loss for multinomial diffusion with x0-prediction parameterization.

    If ``rao_blackwellize_xt=True``, uses the cheap mixture RB estimator:
      alpha_bar(t) * f(x0) + (1 - alpha_bar(t)) * E_{j~pi}[f(j)],
    where E_{j~pi} is Monte-Carlo approximated with
    ``rao_blackwellize_xt_num_samples`` samples.

    For multi-dimensional x0 (e.g. vectors/sequences), this builder uses a
    conditional elementwise RB variant that integrates selected x_t elements while
    sampling the remaining context, which is unbiased for factorized forward
    kernels and avoids the biased whole-vector two-component approximation.

    ``rao_blackwellize_xt_num_features`` controls how many features are RB'ed
    per batch element (uniform random subset). If None, all features are used.
    """

    def loss_fn(
        x0: Array,
        *args,
        rng: Optional[jax.Array] = None,
        t: Optional[Array] = None,
        loss_mask: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Array:
        if rng is None:
            raise ValueError(
                "loss_fn requires an RNG key. Pass it via the 'rng' keyword."
            )

        rng_t, rng_q = jax.random.split(rng, 2)
        if t is None:
            time_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
            t = sample_timesteps_fn(rng_t, time_shape)

        x0_idx = jnp.asarray(x0, dtype=jnp.int32)
        if rao_blackwellize_xt:
            if rao_blackwellize_xt_num_samples < 1:
                raise ValueError(
                    "rao_blackwellize_xt_num_samples must be >= 1."
                )
            if (
                rao_blackwellize_xt_num_features is not None
                and int(rao_blackwellize_xt_num_features) < 1
            ):
                raise ValueError("rao_blackwellize_xt_num_features must be >= 1.")
            if base_probs is None:
                raise ValueError("base_probs is required when rao_blackwellize_xt=True.")
            pi = jnp.asarray(base_probs, dtype=jnp.float32)
            if pi.ndim != 1:
                raise ValueError("base_probs must be a 1D probability vector.")
            pi = pi / jnp.maximum(jnp.sum(pi), 1e-12)

            k = int(num_classes) if num_classes is not None else int(pi.shape[0])
            if pi.shape[0] != k:
                raise ValueError("base_probs has mismatched class dimension.")

            if alpha_bar_fn is not None:
                alpha = jnp.asarray(alpha_bar_fn(t), dtype=jnp.float32)
            else:
                if q_xt_given_x0_probs_fn is None:
                    raise ValueError(
                        "Need alpha_bar_fn or q_xt_given_x0_probs_fn when "
                        "rao_blackwellize_xt=True."
                    )
                q_probs = jnp.asarray(q_xt_given_x0_probs_fn(x0_idx, t), dtype=jnp.float32)
                if q_probs.shape[-1] != k:
                    raise ValueError(
                        "q_xt_given_x0_probs_fn returned mismatched class dimension."
                    )
                pi_x0 = jnp.take(pi, x0_idx)
                q_self = jnp.take_along_axis(q_probs, x0_idx[..., None], axis=-1).squeeze(-1)
                alpha = (q_self - pi_x0) / jnp.maximum(1.0 - pi_x0, 1e-6)

            alpha = _broadcast_weight_like(jnp.clip(alpha, 0.0, 1.0), x0_idx.astype(jnp.float32))

            # Scalar-categorical case: exact two-component mixture in one variable.
            if x0_idx.ndim == 1:
                log_pi = jnp.log(jnp.maximum(pi, 1e-12))
                x_pi = jax.random.categorical(
                    rng_q,
                    log_pi,
                    shape=(rao_blackwellize_xt_num_samples,) + x0_idx.shape,
                    axis=-1,
                )
                xt_all = jnp.concatenate([x0_idx[None, ...], x_pi], axis=0)

                logits_per_xt = jax.vmap(
                    lambda x_t_cand: model_fn(t, x_t_cand, *args, **kwargs)
                )(xt_all)
                log_probs = jax.nn.log_softmax(logits_per_xt, axis=-1)
                target = jnp.broadcast_to(
                    x0_idx[None, ..., None],
                    logits_per_xt.shape[:-1] + (1,),
                ).squeeze(-1)
                ce_per_xt = -jnp.take_along_axis(
                    log_probs, target[..., None], axis=-1
                ).squeeze(-1)

                f_x0 = ce_per_xt[0]
                f_pi = jnp.mean(ce_per_xt[1:], axis=0)
                ce = alpha * f_x0 + (1.0 - alpha) * f_pi
            else:
                # Multi-dimensional case:
                # 1) compute sampled CE at x_base for all features
                # 2) replace a random feature subset by conditional RB estimates
                #    given sampled context x_{t,-i}
                rng_base, rng_feat, rng_pi = jax.random.split(rng_q, 3)
                x_base = q_sample_fn(rng_base, x0_idx, t)

                b = x0_idx.shape[0]
                feat_shape = x0_idx.shape[1:]
                num_feat = int(np.prod(feat_shape))
                m = int(rao_blackwellize_xt_num_samples)
                r = (
                    num_feat
                    if rao_blackwellize_xt_num_features is None
                    else min(int(rao_blackwellize_xt_num_features), num_feat)
                )

                x0_flat = x0_idx.reshape(b, num_feat)
                x_base_flat = jnp.asarray(x_base, dtype=jnp.int32).reshape(b, num_feat)
                alpha_flat = alpha.reshape(b, num_feat)
                logits_base = model_fn(t, x_base, *args, **kwargs)
                log_probs_base = jax.nn.log_softmax(logits_base, axis=-1)
                ce_base = -jnp.take_along_axis(
                    log_probs_base, x0_idx[..., None], axis=-1
                ).squeeze(-1)
                ce_flat = ce_base.reshape(b, num_feat)

                if r < num_feat:
                    feat_idx = jax.random.choice(
                        rng_feat,
                        a=jnp.arange(num_feat, dtype=jnp.int32),
                        shape=(r,),
                        replace=False,
                    )
                else:
                    feat_idx = jnp.arange(num_feat, dtype=jnp.int32)

                x0_sel = jnp.take(x0_flat, feat_idx, axis=1)  # [B, R]
                alpha_sel = jnp.take(alpha_flat, feat_idx, axis=1)  # [B, R]
                feat_mask = jax.nn.one_hot(
                    feat_idx, num_feat, dtype=x_base_flat.dtype
                )  # [R, F]

                # Keep-candidates for selected features.
                x_keep_flat = (
                    x_base_flat[None, :, :] * (1 - feat_mask[:, None, :])
                    + jnp.swapaxes(x0_sel, 0, 1)[..., None] * feat_mask[:, None, :]
                )  # [R, B, F]

                # Pi-candidates per selected feature and MC sample.
                log_pi = jnp.log(jnp.maximum(pi, 1e-12))
                x_pi_flat = jax.random.categorical(
                    rng_pi,
                    log_pi,
                    shape=(m, b, r),
                    axis=-1,
                )  # [M, B, R]
                x_pi_cand_flat = (
                    x_base_flat[None, None, :, :] * (1 - feat_mask[None, :, None, :])
                    + jnp.transpose(x_pi_flat, (0, 2, 1))[..., None]
                    * feat_mask[None, :, None, :]
                )  # [M, R, B, F]

                x_keep = x_keep_flat.reshape((r, b) + feat_shape)
                x_pi = x_pi_cand_flat.reshape((m * r, b) + feat_shape)
                x_all = jnp.concatenate([x_keep, x_pi], axis=0)

                logits_all = jax.vmap(
                    lambda x_t_cand: model_fn(t, x_t_cand, *args, **kwargs)
                )(x_all)
                log_probs_all = jax.nn.log_softmax(logits_all, axis=-1)
                targets = jnp.broadcast_to(
                    x0_idx[None, ..., None], logits_all.shape[:-1] + (1,)
                )
                ce_all = -jnp.take_along_axis(log_probs_all, targets, axis=-1).squeeze(-1)

                ce_all_flat = ce_all.reshape((r + m * r, b, num_feat))
                feat_idx_keep = jnp.broadcast_to(
                    feat_idx[:, None, None], (r, b, 1)
                )
                ce_keep = jnp.swapaxes(
                    jnp.take_along_axis(
                        ce_all_flat[:r], feat_idx_keep, axis=-1
                    ).squeeze(-1),
                    0,
                    1,
                )  # [B, R]

                ce_pi_flat = ce_all_flat[r:].reshape(m, r, b, num_feat)
                feat_idx_pi = jnp.broadcast_to(
                    feat_idx[None, :, None, None], (m, r, b, 1)
                )
                ce_pi = jnp.swapaxes(
                    jnp.mean(
                        jnp.take_along_axis(ce_pi_flat, feat_idx_pi, axis=-1).squeeze(
                            -1
                        ),
                        axis=0,
                    ),
                    0,
                    1,
                )  # [B, R]

                ce_rb = alpha_sel * ce_keep + (1.0 - alpha_sel) * ce_pi
                ce_flat = ce_flat.at[:, feat_idx].set(ce_rb)
                ce = ce_flat.reshape(x0_idx.shape)
        else:
            x_t = q_sample_fn(rng_q, x0_idx, t)
            logits = model_fn(t, x_t, *args, **kwargs)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            ce = -jnp.take_along_axis(log_probs, x0_idx[..., None], axis=-1).squeeze(-1)

        if weight_fn is not None:
            ce = ce * _broadcast_weight_like(weight_fn(t), ce)

        if loss_mask is not None:
            mask = jnp.asarray(loss_mask, dtype=bool)
            if mask.shape != ce.shape:
                mask = jnp.broadcast_to(mask, ce.shape)
            ce = jnp.where(~mask, ce, jnp.zeros_like(ce))

        return reduction_fn(ce)

    return loss_fn
