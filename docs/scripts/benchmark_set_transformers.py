"""Small set-learning benchmark and invariance audit for Set Transformer tools.

Run from the repository root:
    JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_set_transformers

Compares equivariant DeepSet, InducedSelfAttention (ISAB) with standard and
scalable-softmax query scaling (SSMax / QASSMax) and flex_attention,
MultiHeadLatentAttention (MLA), and full self-attention (Transformer / SAB) on:
  1. ``cluster_denoise``      -- equivariant mixture-of-Gaussians cluster denoising.
  2. ``padded_outlier_mean``  -- variable-length padded sets with outliers (KVLenMask).
  3. ``in_context_regression``-- split context/query set regression (ISAB ``x_kv``).
Also runs exact invariance audits (permutation equivariance, KVLenMask padding
invariance, x_kv query isolation, BatchedLowRankBias token-merging equivalence,
and bfloat16 dtype preservation) and token-count scaling timings.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from probjax.nn import (
    BatchedLowRankBias,
    DeepSet,
    InducedSelfAttention,
    KVLenMask,
    MultiHeadLatentAttention,
    PerHeadQueryScale,
    QASSMaxQueryScale,
    SSMaxQueryScale,
    Transformer,
)
from probjax.nn.layers.attention import dot_product_attention, flex_attention
from probjax.nn.layers.feedforward import FeedForward

ARMS = (
    "deepset",
    "isab",
    "isab_per_head",
    "isab_ssmax",
    "isab_qassmax",
    "mla",
    "sab",
)
# Set by --no-zero-init: build ISAB blocks without zero-initialised branches.
ZERO_INIT = True
ALL_ARMS = (*ARMS, "isab_flex")
TASKS = (
    "cluster_denoise",
    "padded_outlier_mean",
    "in_context_regression",
)
NUM_CONTEXT = 24


# ---------------------------------------------------------------------------
# Synthetic set benchmark problems
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("batch_size", "set_size", "dim"))
def sample_cluster_denoise(
    key: jax.Array,
    batch_size: int = 32,
    set_size: int = 32,
    dim: int = 4,
    noise_std: float = 0.3,
) -> dict[str, jax.Array]:
    """Two-cluster Gaussian mixture per set; target is each point's cluster center."""
    k_centers, k_assign, k_noise = jax.random.split(key, 3)
    centers = 2.0 * jax.random.normal(k_centers, (batch_size, 2, dim))
    assign = jax.random.bernoulli(k_assign, 0.5, (batch_size, set_size)).astype(
        jnp.int32
    )
    target = jnp.take_along_axis(centers, assign[:, :, None], axis=1)
    points = target + noise_std * jax.random.normal(
        k_noise, (batch_size, set_size, dim)
    )
    # Oracle empirical cluster mean (using true cluster assignments).
    one_hot = jax.nn.one_hot(assign, 2)  # [B, N, 2]
    counts = jnp.maximum(jnp.sum(one_hot, axis=1, keepdims=True), 1.0)  # [B, 1, 2]
    emp_centers = jnp.einsum("bnk,bnd->bkd", one_hot, points) / jnp.swapaxes(
        counts, 1, 2
    )
    oracle = jnp.take_along_axis(emp_centers, assign[:, :, None], axis=1)
    return {
        "x": points,
        "y": target,
        "oracle": oracle,
    }


@partial(jax.jit, static_argnames=("batch_size", "max_set_size", "min_set_size", "dim"))
def sample_padded_outlier_mean(
    key: jax.Array,
    batch_size: int = 32,
    max_set_size: int = 32,
    min_set_size: int = 8,
    dim: int = 4,
) -> dict[str, jax.Array]:
    """Variable-length sets with 15% outliers and large random padding."""
    k_mean, k_len, k_out, k_core, k_outlier, k_pad = jax.random.split(key, 6)
    true_mean = 2.0 * jax.random.normal(k_mean, (batch_size, dim))
    lengths = jax.random.randint(
        k_len, (batch_size,), min_set_size, max_set_size + 1, dtype=jnp.int32
    )
    is_outlier = jax.random.bernoulli(k_out, 0.15, (batch_size, max_set_size))
    core = true_mean[:, None, :] + 0.25 * jax.random.normal(
        k_core, (batch_size, max_set_size, dim)
    )
    outlier = true_mean[:, None, :] + 3.0 * jax.random.normal(
        k_outlier, (batch_size, max_set_size, dim)
    )
    valid_points = jnp.where(is_outlier[:, :, None], outlier, core)
    padding = 10.0 * jax.random.normal(k_pad, (batch_size, max_set_size, dim))
    valid_mask = jnp.arange(max_set_size)[None, :] < lengths[:, None]
    points = jnp.where(valid_mask[:, :, None], valid_points, padding)

    # Naive masked mean baseline (corrupted by the 15% heavy outliers).
    naive_mean = jnp.sum(
        points * valid_mask[:, :, None], axis=1
    ) / jnp.maximum(lengths[:, None], 1)
    # Oracle clean-inlier mean baseline.
    inlier_mask = valid_mask & (~is_outlier)
    inlier_count = jnp.maximum(jnp.sum(inlier_mask, axis=1, keepdims=True), 1)
    oracle_mean = jnp.sum(points * inlier_mask[:, :, None], axis=1) / inlier_count
    return {
        "x": points,
        "lengths": lengths,
        "valid_mask": valid_mask,
        "y": true_mean,
        "naive": naive_mean,
        "oracle": oracle_mean,
    }


@partial(jax.jit, static_argnames=("batch_size", "num_context", "num_query", "dim"))
def sample_in_context_regression(
    key: jax.Array,
    batch_size: int = 32,
    num_context: int = NUM_CONTEXT,
    num_query: int = 8,
    dim: int = 3,
) -> dict[str, jax.Array]:
    """Per-set linear+quadratic ridge task: infer function from context tokens."""
    k_w1, k_w2, k_u, k_noise = jax.random.split(key, 4)
    total = num_context + num_query
    w_lin = jax.random.normal(k_w1, (batch_size, dim))
    w_quad = 0.5 * jax.random.normal(k_w2, (batch_size, dim))
    u = jax.random.normal(k_u, (batch_size, total, dim))
    clean_y = jnp.einsum("bd,bnd->bn", w_lin, u) + jnp.einsum(
        "bd,bnd->bn", w_quad, u**2 - 1.0
    )
    noisy_y = clean_y + 0.1 * jax.random.normal(k_noise, (batch_size, total))

    # Context tokens carry observed noisy_y and flag=1; query tokens carry y=0 and flag=0.
    is_ctx = (jnp.arange(total) < num_context).astype(u.dtype)[None, :, None]
    is_ctx = jnp.broadcast_to(is_ctx, (batch_size, total, 1))
    obs_y = jnp.where(is_ctx[..., 0] > 0.5, noisy_y, 0.0)[..., None]
    tokens = jnp.concatenate([u, obs_y, is_ctx], axis=-1)
    return {
        "x": tokens,
        "y": clean_y[:, num_context:],
    }


# ---------------------------------------------------------------------------
# Set encoder backbones
# ---------------------------------------------------------------------------


class EquivariantDeepSetBlock(nnx.Module):
    """Permutation-equivariant DeepSet layer combining token and pooled features."""

    def __init__(self, model_dim: int, *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(model_dim, rngs=rngs)
        self.deepset = DeepSet(
            phi=FeedForward(
                model_dim, model_dim, hidden_features=2 * model_dim, rngs=rngs
            ),
            rho= nnx.Linear(model_dim, model_dim, rngs=rngs),
            reduction=jnp.mean,
            axis=-2,
            rngs=rngs,
        )
        self.ff = FeedForward(
            model_dim, model_dim, hidden_features=4 * model_dim, rngs=rngs
        )
        self.ff_norm = nnx.LayerNorm(model_dim, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        valid_mask: jax.Array | None = None,
        num_context: int | None = None,
    ) -> jax.Array:
        h = self.norm(x)
        if num_context is not None:
            phi_h = self.deepset.phi(h[:, :num_context])
            pooled = self.deepset.rho(jnp.mean(phi_h, axis=-2))
        elif valid_mask is not None:
            phi_h = self.deepset.phi(h)
            mask_f = valid_mask[:, :, None].astype(phi_h.dtype)
            denom = jnp.maximum(jnp.sum(mask_f, axis=1), 1.0)
            pooled = self.deepset.rho(jnp.sum(phi_h * mask_f, axis=1) / denom)
        else:
            pooled = self.deepset(h)
        x = x + pooled[:, None, :]
        return x + self.ff(self.ff_norm(x))


class MLASetBlock(nnx.Module):
    """Pre-norm set transformer block using MultiHeadLatentAttention."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        kv_lora_rank: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm_q = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(model_dim, rngs=rngs)
        self.attn = MultiHeadLatentAttention(
            num_heads=num_heads,
            in_features=model_dim,
            kv_lora_rank=kv_lora_rank,
            mode="absorbed",
            query_scale_cls=SSMaxQueryScale,
            attention_fn=dot_product_attention,
            rngs=rngs,
        )
        self.ff_norm = nnx.LayerNorm(model_dim, rngs=rngs)
        self.ff = FeedForward(
            model_dim, model_dim, hidden_features=4 * model_dim, rngs=rngs
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        x_kv: jax.Array | None = None,
        mask: Any = None,
        kv_len: jax.Array | int | None = None,
    ) -> jax.Array:
        kv = x if x_kv is None else x_kv
        h = x + self.attn(
            self.norm_q(x),
            self.norm_kv(kv),
            mask=mask,
            kv_len=kv_len,
        )
        return h + self.ff(self.ff_norm(h))


class SetBackbone(nnx.Module):
    """Unified wrapper around DeepSet, ISAB variants, MLA, and SAB (Transformer)."""

    def __init__(
        self,
        kind: str,
        in_dim: int,
        out_dim: int,
        *,
        model_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        num_inducing_points: int = 8,
        rngs: nnx.Rngs,
    ):
        self.kind = kind
        self.encoder = nnx.Linear(in_dim, model_dim, rngs=rngs)
        self.decoder = nnx.Linear(model_dim, out_dim, rngs=rngs)

        if kind == "deepset":
            self.blocks = nnx.List([
                EquivariantDeepSetBlock(model_dim, rngs=rngs)
                for _ in range(num_layers)
            ])
        elif kind.startswith("isab"):
            q_scale_map = {
                "isab": None,
                "isab_per_head": PerHeadQueryScale,
                "isab_ssmax": SSMaxQueryScale,
                "isab_qassmax": QASSMaxQueryScale,
                "isab_flex": SSMaxQueryScale,
            }
            attn_fn = flex_attention if kind == "isab_flex" else dot_product_attention
            q_scale_cls = q_scale_map[kind]
            self.blocks = nnx.List([
                InducedSelfAttention(
                    model_dim,
                    num_inducing_points=num_inducing_points,
                    num_heads=num_heads,
                    attn_size=model_dim // num_heads,
                    q_scale_cls=q_scale_cls,
                    output_q_scale_cls=q_scale_cls,
                    attention_fn=attn_fn,
                    zero_init=ZERO_INIT,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ])
        elif kind == "mla":
            self.blocks = nnx.List([
                MLASetBlock(
                    model_dim,
                    num_heads=num_heads,
                    kv_lora_rank=model_dim // 2,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ])
        elif kind == "sab":
            self.transformer = Transformer(
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                attn_size=model_dim // num_heads,
                attention_fn=dot_product_attention,
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unknown set model kind: {kind!r}")

    def encode_tokens(
        self,
        x: jax.Array,
        *,
        lengths: jax.Array | None = None,
        valid_mask: jax.Array | None = None,
        num_context: int | None = None,
    ) -> jax.Array:
        h = self.encoder(x)
        kv_mask = KVLenMask(lengths) if lengths is not None else None
        if self.kind == "deepset":
            for block in self.blocks:
                h = block(h, valid_mask=valid_mask, num_context=num_context)
            return h
        if self.kind.startswith("isab") or self.kind == "mla":
            for block in self.blocks:
                h_kv = h[:, :num_context] if num_context is not None else None
                h = block(h, x_kv=h_kv, mask=kv_mask, kv_len=lengths)
            return h
        # SAB (Transformer)
        if num_context is not None:
            ctx_lengths = jnp.full((x.shape[0],), num_context, dtype=jnp.int32)
            kv_mask = KVLenMask(ctx_lengths)
        return self.transformer(h, mask=kv_mask, kv_len=lengths)

    def __call__(
        self,
        x: jax.Array,
        *,
        lengths: jax.Array | None = None,
        valid_mask: jax.Array | None = None,
        num_context: int | None = None,
        pool: bool = False,
    ) -> jax.Array:
        h = self.encode_tokens(
            x, lengths=lengths, valid_mask=valid_mask, num_context=num_context
        )
        if pool:
            if valid_mask is not None:
                mask_f = valid_mask[:, :, None].astype(h.dtype)
                h = jnp.sum(h * mask_f, axis=1) / jnp.maximum(
                    jnp.sum(mask_f, axis=1), 1.0
                )
            else:
                h = jnp.mean(h, axis=1)
        return self.decoder(h)


# ---------------------------------------------------------------------------
# Invariance and correctness audit for ISAB
# ---------------------------------------------------------------------------


def run_isab_audit() -> dict[str, Any]:
    """Verify exact invariances and dtype contracts of InducedSelfAttention."""
    rngs = nnx.Rngs(0)
    isab = InducedSelfAttention(
        16,
        num_inducing_points=4,
        num_heads=4,
        q_scale_cls=SSMaxQueryScale,
        output_q_scale_cls=SSMaxQueryScale,
        zero_init=False,
        rngs=rngs,
    )
    x = jax.random.normal(jax.random.key(0), (2, 12, 16))

    # 1. Permutation equivariance: ISAB(P x) == P ISAB(x)
    perm = jnp.array([5, 2, 11, 0, 8, 3, 1, 9, 4, 10, 6, 7])
    y = isab(x)
    y_perm = isab(x[:, perm, :])
    perm_err = float(jnp.max(jnp.abs(y[:, perm, :] - y_perm)))

    # 2. Padding mask isolation: perturbing masked padding tokens changes valid outputs by 0.
    lengths = jnp.array([5, 8], dtype=jnp.int32)
    mask = KVLenMask(lengths)
    x_perturbed = x.at[0, 5:, :].add(50.0).at[1, 8:, :].add(-30.0)
    y_masked_1 = isab(x, mask=mask)
    y_masked_2 = isab(x_perturbed, mask=mask)
    pad_leak_err = float(
        jnp.maximum(
            jnp.max(jnp.abs(y_masked_1[0, :5] - y_masked_2[0, :5])),
            jnp.max(jnp.abs(y_masked_1[1, :8] - y_masked_2[1, :8])),
        )
    )

    # 3. Unpadded vs padded+KVLenMask equivalence (including SSMax on both MABs).
    x_short = x[:, :6, :]
    len_short = jnp.array([6, 6], dtype=jnp.int32)
    y_unpadded = isab(x_short)
    y_padded = isab(x, mask=KVLenMask(len_short))[:, :6, :]
    unpadded_vs_padded_err = float(jnp.max(jnp.abs(y_unpadded - y_padded)))

    # 4. Separate x_kv context isolation: perturbing query tokens leaves context outputs unchanged.
    x_query_perturbed = x.at[:, 7:, :].add(25.0)
    y_kv_1 = isab(x, x_kv=x[:, :7, :])
    y_kv_2 = isab(x_query_perturbed, x_kv=x_query_perturbed[:, :7, :])
    x_kv_leak_err = float(jnp.max(jnp.abs(y_kv_1[:, :7, :] - y_kv_2[:, :7, :])))

    # 5. Proportional attention (token merging) equivalence on dense and flex_attention.
    x_unique = jax.random.normal(jax.random.key(1), (2, 3, 16))
    counts = jnp.array([[3.0, 1.0, 2.0], [2.0, 2.0, 2.0]])
    x_expanded = jnp.stack([
        jnp.concatenate(
            [
                jnp.repeat(x_unique[0, 0:1], 3, axis=0),
                jnp.repeat(x_unique[0, 1:2], 1, axis=0),
                jnp.repeat(x_unique[0, 2:3], 2, axis=0),
            ],
            axis=0,
        ),
        jnp.concatenate(
            [
                jnp.repeat(x_unique[1, 0:1], 2, axis=0),
                jnp.repeat(x_unique[1, 1:2], 2, axis=0),
                jnp.repeat(x_unique[1, 2:3], 2, axis=0),
            ],
            axis=0,
        ),
    ])
    bias = BatchedLowRankBias.from_key_bias(jnp.log(counts))
    total_len = jnp.sum(counts, axis=-1)
    y_exp = isab(x_expanded)
    y_uniq_dense = isab(x_unique, bias=bias, kv_len=total_len)
    isab_flex = InducedSelfAttention(
        16,
        num_inducing_points=4,
        num_heads=4,
        q_scale_cls=SSMaxQueryScale,
        output_q_scale_cls=SSMaxQueryScale,
        attention_fn=flex_attention,
        zero_init=False,
        rngs=nnx.Rngs(0),
    )
    y_uniq_flex = jax.jit(
        lambda x_u, b, tl: isab_flex(x_u, bias=b, kv_len=tl)
    )(x_unique, bias, total_len)
    rep_idx_0 = jnp.array([0, 3, 4])
    rep_idx_1 = jnp.array([0, 2, 4])
    y_exp_unique = jnp.stack([y_exp[0, rep_idx_0], y_exp[1, rep_idx_1]])
    merge_dense_err = float(jnp.max(jnp.abs(y_exp_unique - y_uniq_dense)))
    merge_flex_err = float(jnp.max(jnp.abs(y_exp_unique - y_uniq_flex)))

    # 6. bfloat16 preservation across both MABs and norm_cls=None.
    isab_bf16 = InducedSelfAttention(
        16,
        num_inducing_points=4,
        num_heads=4,
        q_scale_cls=SSMaxQueryScale,
        output_q_scale_cls=PerHeadQueryScale,
        norm_cls=None,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        zero_init=False,
        rngs=nnx.Rngs(0),
    )
    y_bf16 = isab_bf16(x.astype(jnp.bfloat16))

    return {
        "permutation_equivariance_max_err": perm_err,
        "padding_perturbation_max_err": pad_leak_err,
        "unpadded_vs_padded_max_err": unpadded_vs_padded_err,
        "x_kv_query_perturbation_max_err": x_kv_leak_err,
        "proportional_attention_merge_dense_max_err": merge_dense_err,
        "proportional_attention_merge_flex_max_err": merge_flex_err,
        "bfloat16_output_is_bf16": y_bf16.dtype == jnp.bfloat16,
    }


# ---------------------------------------------------------------------------
# Training & runtime evaluation
# ---------------------------------------------------------------------------


def evaluate_task_arm(
    task: str,
    kind: str,
    *,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    seeds: int,
    set_size: int = 64,
    num_context: int = 48,
    num_query: int = 16,
    num_inducing_points: int = 32,
    model_dim: int = 64,
    num_layers: int = 2,
) -> list[dict[str, Any]]:
    if task in ("cluster_denoise", "padded_outlier_mean"):
        in_dim, out_dim = 4, 4
    elif task == "in_context_regression":
        in_dim, out_dim = 5, 1
    else:
        raise ValueError(f"Unknown task: {task}")

    def build_pair(seed: int):
        model = SetBackbone(
            kind,
            in_dim,
            out_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_inducing_points=num_inducing_points,
            rngs=nnx.Rngs(seed),
        )
        optimizer = nnx.Optimizer(
            model,
            optax.chain(optax.clip_by_global_norm(5.0), optax.adam(learning_rate)),
            wrt=nnx.Param,
        )
        return model, optimizer

    model0, opt0 = build_pair(0)
    params_count = sum(
        leaf.size for leaf in jax.tree.leaves(nnx.state(model0, nnx.Param))
    )
    graphdef, _ = nnx.split((model0, opt0))

    if task == "cluster_denoise":

        def loss_fn(m: SetBackbone, batch: dict[str, jax.Array]) -> jax.Array:
            pred = m(batch["x"])
            return jnp.mean((pred - batch["y"]) ** 2)

        sampler = lambda k, b: sample_cluster_denoise(
            k, batch_size=b, set_size=set_size
        )
    elif task == "padded_outlier_mean":

        def loss_fn(m: SetBackbone, batch: dict[str, jax.Array]) -> jax.Array:
            pred = m(
                batch["x"],
                lengths=batch["lengths"],
                valid_mask=batch["valid_mask"],
                pool=True,
            )
            return jnp.mean((pred - batch["y"]) ** 2)

        sampler = lambda k, b: sample_padded_outlier_mean(
            k,
            batch_size=b,
            max_set_size=set_size,
            min_set_size=max(8, set_size // 4),
        )
    else:

        def loss_fn(m: SetBackbone, batch: dict[str, jax.Array]) -> jax.Array:
            pred = m(batch["x"], num_context=num_context)[:, num_context:, 0]
            return jnp.mean((pred - batch["y"]) ** 2)

        sampler = lambda k, b: sample_in_context_regression(
            k, batch_size=b, num_context=num_context, num_query=num_query
        )

    @jax.jit
    def run_steps(state, base_key, n_steps):
        def body(step_idx, carry_state):
            m, opt = nnx.merge(graphdef, carry_state)
            batch = sampler(jax.random.fold_in(base_key, step_idx), batch_size)
            _, grads = nnx.value_and_grad(loss_fn)(m, batch)
            opt.update(m, grads)
            return nnx.state((m, opt))

        return jax.lax.fori_loop(0, n_steps, body, state)

    @jax.jit
    def eval_fn(state, batch):
        m, _ = nnx.merge(graphdef, state)
        return loss_fn(m, batch)

    results = []
    for seed in range(seeds):
        model, optimizer = build_pair(seed)
        _, state = nnx.split((model, optimizer))
        train_key = jax.random.key(1000 + seed)
        start = time.perf_counter()
        state = jax.block_until_ready(run_steps(state, train_key, steps))
        fit_seconds = time.perf_counter() - start

        eval_batch = sampler(jax.random.key(9999 + seed), eval_batch_size)
        test_mse = float(jax.block_until_ready(eval_fn(state, eval_batch)))

        t0 = time.perf_counter()
        jax.block_until_ready(run_steps(state, train_key, 10))
        step_ms = 1000.0 * (time.perf_counter() - t0) / 10.0

        baselines: dict[str, float] = {}
        if task == "cluster_denoise":
            baselines["identity_mse"] = float(
                jnp.mean((eval_batch["x"] - eval_batch["y"]) ** 2)
            )
            baselines["oracle_mse"] = float(
                jnp.mean((eval_batch["oracle"] - eval_batch["y"]) ** 2)
            )
        elif task == "padded_outlier_mean":
            baselines["naive_masked_mean_mse"] = float(
                jnp.mean((eval_batch["naive"] - eval_batch["y"]) ** 2)
            )
            baselines["oracle_inlier_mean_mse"] = float(
                jnp.mean((eval_batch["oracle"] - eval_batch["y"]) ** 2)
            )
        else:
            baselines["zero_pred_mse"] = float(jnp.mean(eval_batch["y"] ** 2))

        row = {
            "task": task,
            "arm": kind,
            "seed": seed,
            "parameters": params_count,
            "test_mse": test_mse,
            **baselines,
            "fit_seconds": fit_seconds,
            "train_step_ms": step_ms,
            "finite": bool(np.isfinite(test_mse)),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


def run_scaling(
    token_counts: tuple[int, ...] = (64, 256, 1024, 2048),
    batch_size: int = 4,
    repeats: int = 3,
    num_inducing_points: int = 64,
    model_dim: int = 64,
) -> list[dict[str, Any]]:
    """Compare forward and backward runtime scaling across set sizes N."""
    rows = []
    for kind in ("deepset", "isab", "isab_ssmax", "mla", "sab"):
        for tokens in token_counts:
            model = SetBackbone(
                kind,
                4,
                4,
                model_dim=model_dim,
                num_heads=4,
                num_layers=2,
                num_inducing_points=num_inducing_points,
                rngs=nnx.Rngs(0),
            )
            x = jax.random.normal(jax.random.key(0), (batch_size, tokens, 4))
            graphdef, state = nnx.split(model)
            fwd = jax.jit(lambda st, inp, gd=graphdef: nnx.merge(gd, st)(inp))
            bwd = jax.jit(
                jax.value_and_grad(
                    lambda st, inp, gd=graphdef: jnp.mean(
                        nnx.merge(gd, st)(inp) ** 2
                    )
                )
            )
            for op_name, fn in (("forward", fwd), ("backward", bwd)):
                t0 = time.perf_counter()
                out = jax.block_until_ready(fn(state, x))
                cold_s = time.perf_counter() - t0
                durations = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(state, x))
                    durations.append(time.perf_counter() - t0)
                rows.append({
                    "arm": kind,
                    "tokens": tokens,
                    "operation": op_name,
                    "cold_seconds": cold_s,
                    "warm_ms": 1000.0 * float(np.median(durations)),
                    "finite": all(
                        bool(jnp.isfinite(leaf).all())
                        for leaf in jax.tree.leaves(out)
                    ),
                })
            jax.clear_caches()
    return rows


def summarize(payload: dict[str, Any]) -> None:
    print("\n=== ISAB Invariance Audit ===", flush=True)
    for k, v in payload["audit"].items():
        print(f"  {k:44s}: {v}", flush=True)

    print("\n=== Set Benchmark Tasks (mean test MSE across seeds) ===", flush=True)
    rows = payload["task_results"]
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    for task in tasks:
        print(f"\nTask: {task}", flush=True)
        for arm in arms:
            sub = [r for r in rows if r["task"] == task and r["arm"] == arm]
            if not sub:
                continue
            mses = [r["test_mse"] for r in sub]
            step_ms = np.mean([r["train_step_ms"] for r in sub])
            print(
                f"  {arm:16s}  MSE={np.mean(mses):.4f}"
                + (f" ± {np.std(mses, ddof=1):.4f}" if len(mses) > 1 else "")
                + f"  (params={sub[0]['parameters']:6d}, step={step_ms:.2f} ms)",
                flush=True,
            )

    if payload.get("scaling"):
        print("\n=== Runtime Scaling (warm ms) ===", flush=True)
        for row in payload["scaling"]:
            print(
                f"  {row['arm']:12s} N={row['tokens']:4d} {row['operation']:8s}: "
                f"{row['warm_ms']:6.2f} ms",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ARMS))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--set-size", type=int, default=64)
    parser.add_argument("--num-context", type=int, default=48)
    parser.add_argument("--num-query", type=int, default=16)
    parser.add_argument("--num-inducing", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tokens", nargs="+", type=int, default=[64, 256, 1024, 2048])
    parser.add_argument("--scaling-inducing", type=int, default=64)
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--no-zero-init", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/set_transformers.json"),
    )
    args = parser.parse_args()
    global ZERO_INIT
    ZERO_INIT = not args.no_zero_init

    if args.summary_only:
        summarize(json.loads(args.output.read_text()))
        return

    audit = run_isab_audit()
    task_results = []
    for task in args.tasks:
        for arm in args.arms:
            task_results.extend(
                evaluate_task_arm(
                    task,
                    arm,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size,
                    learning_rate=args.learning_rate,
                    seeds=args.seeds,
                    set_size=args.set_size,
                    num_context=args.num_context,
                    num_query=args.num_query,
                    num_inducing_points=args.num_inducing,
                    model_dim=args.model_dim,
                    num_layers=args.num_layers,
                )
            )
        jax.clear_caches()

    scaling = (
        []
        if args.skip_scaling
        else run_scaling(
            tuple(args.tokens),
            batch_size=4,
            repeats=3,
            num_inducing_points=args.scaling_inducing,
            model_dim=args.model_dim,
        )
    )

    payload = {
        "jax": jax.__version__,
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "settings": {
            "tasks": args.tasks,
            "arms": args.arms,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "set_size": args.set_size,
            "num_context": args.num_context,
            "num_query": args.num_query,
            "num_inducing": args.num_inducing,
            "model_dim": args.model_dim,
            "num_layers": args.num_layers,
            "tokens": args.tokens,
            "scaling_inducing": args.scaling_inducing,
        },
        "audit": audit,
        "task_results": task_results,
        "scaling": scaling,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    summarize(payload)


if __name__ == "__main__":
    main()
