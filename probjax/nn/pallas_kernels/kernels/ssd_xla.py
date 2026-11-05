"""Pure-JAX/XLA chunked SSD (Mamba-2 state-space duality).

Computes the same map as :func:`probjax.nn.pallas_kernels.kernels.ssd.ssd`::

    H_t = exp(log_alpha_t) * H_{t-1} + k_t v_t^T,      H_{-1} = h0
    o_t = q_t^T H_t

using the "chunked" SSD algorithm of Dao & Gu (2024, Sec. 6 / ``ssd_minimal``):

1. Split the sequence into chunks of ``chunk_size`` tokens.
2. Intra-chunk: a masked ``C x C`` attention-like matmul with segment-sum
   decays ``exp(cumsum_i - cumsum_j)`` (``i >= j``).
3. Chunk states: ``S_c = sum_j exp(cumsum_end - cumsum_j) k_j v_j^T``.
4. Inter-chunk: the ``nc = L / C`` chunk states are propagated with a
   sequential ``lax.scan`` (``inter_chunk="scan"``) or with a dense
   ``nc x nc`` segment-sum matmul (``inter_chunk="matmul"``).
5. ``o = intra + exp(cumsum_i) * q_i^T H_{chunk start}``.

Everything is plain ``jnp``/``lax`` so it runs on any backend, supports any
``dk``/``dv``/sequence length (the tail is zero-padded with ``log_alpha=0``,
which leaves outputs and the final state unchanged) and is differentiable in
forward and reverse mode via ordinary JAX autodiff. One q/k group serves
``H / G`` value heads: the ``q k^T`` scores are computed once per group.
"""

from __future__ import annotations

import functools
from typing import Literal, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name

# Tuned on an RTX 3090 / many-core x86 CPU (docs/benchmarks/kernel_ssd_*.json):
# larger chunks feed bigger matmuls on GPU; on CPU 128 balances the C x C
# intra-chunk work against the number of sequential chunk steps.
_DEFAULT_CHUNK_SIZE = {"gpu": 256, "cpu": 128}
_FALLBACK_CHUNK_SIZE = 256


def default_chunk_size(backend: str | None = None) -> int:
    backend = backend or jax.default_backend()
    return _DEFAULT_CHUNK_SIZE.get(backend, _FALLBACK_CHUNK_SIZE)


_STATES_NAME = "ssd_chunk_states"


def _segsum_exp(cs: jax.Array) -> jax.Array:
    """``exp(cs[..., i] - cs[..., j])`` for ``i >= j`` else ``0``."""
    n = cs.shape[-1]
    seg = cs[..., :, None] - cs[..., None, :]
    mask = jnp.tril(jnp.ones((n, n), dtype=bool))
    return jnp.exp(jnp.where(mask, seg, -jnp.inf))


@functools.partial(
    jax.jit,
    static_argnames=(
        "chunk_size",
        "precision",
        "inter_chunk",
        "remat",
        "return_final_state",
    ),
)
def ssd_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    h0: Optional[jax.Array] = None,
    *,
    chunk_size: int | None = None,
    precision: lax.Precision | str | None = lax.Precision.HIGHEST,
    inter_chunk: Literal["scan", "matmul"] = "scan",
    remat: Literal["full", "states", "none"] = "full",
    return_final_state: bool = False,
):
    """Chunked SSD in pure JAX.

    Args:
        q, k: ``[B, G, L, dk]``.
        v: ``[B, H, L, dv]`` with ``H % G == 0``.
        log_alpha: ``[B, H, L]`` per-step log decay.
        h0: optional initial state ``[B, H, dk, dv]``.
        chunk_size: tokens per chunk (clipped to ``L``); default
            :func:`default_chunk_size` (256 on GPU, 128 on CPU).
        precision: matmul precision (default ``HIGHEST``: true fp32 on GPU,
            no TF32).
        inter_chunk: ``"scan"`` (sequential over chunks, O(nc) work) or
            ``"matmul"`` (dense ``nc x nc`` decay matmul, O(nc^2) work but no
            sequential dependency).
        remat: what reverse mode keeps from the forward pass. ``"full"``
            (default) saves only the inputs and recomputes the forward in the
            backward, like the Pallas kernel -- measured within ~3% of plain
            autodiff on GPU, and no slower than ``"states"``;
            ``"states"`` saves the chunk-boundary states ``[B, H, L/C, dk,
            dv]`` and recomputes the intra-chunk blocks; ``"none"`` is plain
            autodiff (stores every intermediate).
        return_final_state: also return ``H_{L-1}`` ``[B, H, dk, dv]``.

    Returns:
        ``o`` ``[B, H, L, dv]`` (dtype of ``v``), or ``(o, final_state)``.
    """
    if chunk_size is None:
        chunk_size = default_chunk_size()
    impl = functools.partial(
        _ssd_chunked_impl,
        chunk_size=chunk_size,
        precision=precision,
        inter_chunk=inter_chunk,
    )
    if remat == "states":
        impl = jax.checkpoint(
            impl, policy=jax.checkpoint_policies.save_only_these_names(_STATES_NAME)
        )
    elif remat == "full":
        impl = jax.checkpoint(impl)
    elif remat != "none":
        raise ValueError(f"remat must be 'states', 'none' or 'full', got {remat!r}")
    out, final = impl(q, k, v, log_alpha, h0)
    return (out, final) if return_final_state else out


def _ssd_chunked_impl(q, k, v, log_alpha, h0, *, chunk_size, precision, inter_chunk):
    bs, ng, seq_len, dk = q.shape
    _, nh, _, dv = v.shape
    if nh % ng != 0:
        raise ValueError(f"num_heads={nh} must be divisible by num_groups={ng}.")
    if inter_chunk not in ("scan", "matmul"):
        raise ValueError(f"inter_chunk must be 'scan' or 'matmul', got {inter_chunk}")
    rep = nh // ng
    out_dtype = v.dtype
    # Compute in fp32 (or wider if the inputs already are, e.g. under x64).
    f32 = jnp.promote_types(jnp.result_type(q, k, v, log_alpha), jnp.float32)
    q, k, v, log_alpha = (x.astype(f32) for x in (q, k, v, log_alpha))
    if h0 is None:
        h0 = jnp.zeros((bs, nh, dk, dv), f32)
    h0 = h0.astype(f32)

    c = max(1, min(int(chunk_size), seq_len))
    nc = -(-seq_len // c)
    pad = nc * c - seq_len
    if pad:
        # Zero k/v/q and log_alpha=0 (decay 1): padded steps leave the state
        # untouched and their outputs are sliced off.
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
        log_alpha = jnp.pad(log_alpha, ((0, 0), (0, 0), (0, pad)))

    ein = functools.partial(jnp.einsum, precision=precision, preferred_element_type=f32)

    q = q.reshape(bs, ng, nc, c, dk)
    k = k.reshape(bs, ng, nc, c, dk)
    v = v.reshape(bs, ng, rep, nc, c, dv)
    cs = jnp.cumsum(log_alpha.reshape(bs, ng, rep, nc, c), axis=-1)
    h0 = h0.reshape(bs, ng, rep, dk, dv)

    # 1) intra-chunk (diagonal blocks); scores are shared by the rep heads.
    scores = ein("bgnid,bgnjd->bgnij", q, k)
    m = scores[:, :, None] * _segsum_exp(cs)
    y = ein("bgrnij,bgrnjv->bgrniv", m, v)

    # 2) per-chunk states (contribution of chunk tokens to the chunk-end state).
    decay_to_end = jnp.exp(cs[..., -1:] - cs)
    # Chunk axis leading: the layout lax.scan consumes/produces (no transposes).
    states = ein("bgncd,bgrncv->nbgrdv", k, v * decay_to_end[..., None])
    log_chunk_decay = jnp.moveaxis(cs[..., -1], -1, 0)  # [nc, b, g, r]

    # 3) inter-chunk recurrence -> state entering each chunk.
    if inter_chunk == "scan":

        def step(h, xs):
            s_c, a_c = xs
            return jnp.exp(a_c)[..., None, None] * h + s_c, h

        final, h_in = lax.scan(step, h0, (states, log_chunk_decay))
        h_in = checkpoint_name(h_in, _STATES_NAME)  # [nc, b, g, r, dk, dv]
    else:
        all_states = jnp.concatenate([h0[None], states], axis=0)
        chunk_cs = jnp.cumsum(
            jnp.pad(jnp.moveaxis(log_chunk_decay, 0, -1), ((0, 0),) * 3 + ((1, 0),)),
            axis=-1,
        )
        decay_chunk = _segsum_exp(chunk_cs)  # [b, g, r, nc+1, nc+1]
        new = ein("bgrzc,cbgrdv->zbgrdv", decay_chunk, all_states)
        h_in, final = new[:-1], new[-1]
        h_in = checkpoint_name(h_in, _STATES_NAME)

    # 4) state -> output contribution.
    y = y + ein("bgncd,nbgrdv->bgrncv", q, h_in) * jnp.exp(cs)[..., None]

    out = y.reshape(bs, nh, nc * c, dv)[:, :, :seq_len].astype(out_dtype)
    return out, final.reshape(bs, nh, dk, dv)
