"""Attention by implementation on one GPU, for device-specific defaults.

Times forward and forward+backward (grads w.r.t. q, k, v) of each candidate
implementation, all bf16, with the KV-length mask the model uses
(KVLenMask / key_value_seq_lengths).  Outputs and grads are compared against
the current default (flex_attention, triton_split, 64x64 tiles).

    python -m docs.scripts.benchmark_gpu_attention_device --out results.jsonl \
        --shapes mla isab smallq generic --impls flex:default cudnn flex:triton torch:flex

torch:* rows need PyTorch; run them in a process with XLA_PYTHON_CLIENT_PREALLOCATE=false
(JAX otherwise reserves most of the GPU).
"""

from __future__ import annotations

import argparse
import itertools
import json
import socket
import statistics
import time
import traceback

import jax
import jax.numpy as jnp

# name: (B, Lq, Lkv, H, D, masked)
SHAPES = {
    # MLA test-row cross-attention (16 heads x tokens x test rows folded into Lq, one latent KV head)
    "mla_32k84": (2, 131072, 32768, 1, 128, True),
    "mla_128k36": (1, 65536, 131072, 1, 128, True),
    "mla_1k756": (4, 262144, 1024, 1, 128, True),
    # TabICL ISAB stages (8 heads, head_dim 32)
    "isab_ind_rows": (56, 256, 32768, 8, 32, True),   # inducing points attend context rows
    "isab_rows_ind": (56, 32768, 256, 8, 32, False),  # rows attend inducing points
    "isab_1k_ind": (1008, 1024, 256, 8, 32, False),
    "isab_ind_1k": (1008, 256, 1024, 8, 32, True),
}
# Phase-1 (v10, 2048x84 / 1024x756) call shapes, for checking what "auto" ran
SHAPES.update({
    "p1_mla_2k84": (2, 131072, 2048, 1, 128, True),
    "p1_mla_1k756": (2, 262144, 1024, 1, 128, True),
    "p1_isab_ind_rows": (112, 256, 2048, 8, 32, True),
    "p1_isab_rows_ind": (112, 2048, 256, 8, 32, False),
    "p1_isab_1k_ind": (256, 1024, 256, 8, 32, False),
    "p1_isab_ind_1k": (256, 256, 1024, 8, 32, True),
    "p1_self_84": (64, 2048, 2048, 8, 64, False),
})
# Small queries over long KV (ISAB inducing points over all context rows, decoding-like)
SHAPES.update({
    "sq_isab_256x131072": (28, 256, 131072, 8, 32, True),
    "sq_isab_256x32768": (56, 256, 32768, 8, 32, True),
    "sq_128x262144_d64": (4, 128, 262144, 8, 64, True),
    "sq_64x131072_d64": (8, 64, 131072, 8, 64, True),
    "sq_16x65536_d128": (16, 16, 65536, 8, 128, True),
    "sq_512x131072_d32": (8, 512, 131072, 8, 32, True),
})
# Generic flex-attention grid for device defaults: (B, Lq, Lkv, H, D, masked)
for d in (32, 64, 128):
    for t in (512, 2048, 8192, 16384):
        SHAPES[f"self_d{d}_t{t}"] = (4, t, t, 8, d, False)
        SHAPES[f"selfm_d{d}_t{t}"] = (4, t, t, 8, d, True)
    for lq, lkv in ((256, 16384), (16384, 256), (1024, 65536), (65536, 1024)):
        SHAPES[f"cross_d{d}_{lq}x{lkv}"] = (4, lq, lkv, 8, d, True)
GROUPS = {
    "phase1": [k for k in SHAPES if k.startswith("p1_")],
    "smallq": [k for k in SHAPES if k.startswith("sq_")],
    "generic": [k for k in SHAPES if k.startswith(("self", "cross"))],
    "mla": [k for k in SHAPES if k.startswith("mla")],
    "isab": [k for k in SHAPES if k.startswith("isab")],
}


def _impls(names):
    from probjax.nn import KVLenMask
    from probjax.nn.layers.attention import flex_attention

    def flex(backward, blocks=None, fwd=None, warps=None, stages=2):
        kw = dict(backward_pass_impl=backward, num_stages=stages)
        if blocks:
            kw.update(zip(("block_q_dkv", "block_kv_dkv", "block_q_dq", "block_kv_dq"), blocks, strict=True))
        if fwd:
            kw.update(block_q=fwd[0], block_k=fwd[1])
        if warps:
            kw["num_warps"] = warps

        def f(q, k, v, lens):
            return flex_attention(q, k, v, mask=None if lens is None else KVLenMask(lens), **kw)
        return f

    def cudnn(q, k, v, lens):
        return jax.nn.dot_product_attention(q, k, v, key_value_seq_lengths=lens, implementation="cudnn")

    def mosaic(q, k, v, lens):  # no mask support: full lengths only (speed ceiling)
        from probjax.nn.pallas_kernels.kernels.flash_attention3 import mha_flash
        return mha_flash(q, k, v)

    out = {}
    for n in names:
        if n.startswith("torch:"):
            out[n] = None
            continue
        if n == "cudnn":
            out[n] = cudnn
        elif n == "mosaic_fa3":
            out[n] = mosaic
        elif n.startswith("mosaic:"):
            # mosaic:bq,bkv,steps,wgs_bwd[,bq_dkv,bkv_dkv,bq_dq,bkv_dq]
            t = [int(x) for x in n.split(":")[1].split(",")]
            kw = dict(block_q=t[0], block_kv=t[1], max_concurrent_steps=t[2], compute_wgs_bwd=t[3])
            if len(t) == 8:
                kw.update(block_q_dkv=t[4], block_kv_dkv=t[5], block_q_dq=t[6], block_kv_dq=t[7])

            def mk(kw=kw):
                from probjax.nn.pallas_kernels.kernels.flash_attention3 import mha_flash
                return lambda q, k, v, lens: mha_flash(q, k, v, **kw)
            out[n] = mk()
        elif n in ("flex:mosaic", "flex:cudnn", "flex:triton"):
            out[n] = (lambda impl: lambda q, k, v, lens: flex_attention(
                q, k, v, mask=None if lens is None else KVLenMask(lens), enable_gqa=True,
                implementation=impl))(n.split(":")[1])
        elif n == "flex:triton_generic":
            out[n] = lambda q, k, v, lens: flex_attention(
                q, k, v, mask=None if lens is None else KVLenMask(lens), implementation="triton")
        elif n == "flex:default":
            out[n] = lambda q, k, v, lens: flex_attention(q, k, v, mask=None if lens is None else KVLenMask(lens))
        elif n.startswith("flex:"):
            # flex:<backward>[:qdkv,kvdkv,qdq,kvdq][:fq,fk][:w<warps>][:s<stages>]
            parts = n.split(":")[1:]
            backward, blocks, fwd, warps, stages = parts[0], None, None, None, 2
            for p in parts[1:]:
                if p.startswith("w"):
                    warps = int(p[1:])
                elif p.startswith("s"):
                    stages = int(p[1:])
                elif p.count(",") == 3:
                    blocks = tuple(int(x) for x in p.split(","))
                elif p.count(",") == 1:
                    fwd = tuple(int(x) for x in p.split(","))
            out[n] = flex(backward, blocks, fwd, warps, stages)
        else:
            raise ValueError(n)
    return out


# ---------------- torch ----------------
_TORCH_FLEX = None


def _torch_case(impl, q, k, v, do, lens, reps):
    """Time a torch implementation on the same data; returns (fwd_ms, fwd_bwd_ms, out, grads)."""
    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    global _TORCH_FLEX
    dev = "cuda"
    tt = lambda x: torch.from_numpy(np.asarray(x.astype(jnp.float32))).to(dev, torch.bfloat16)
    # torch layout [B, H, T, D]
    tq, tk, tv = (tt(x).transpose(1, 2).contiguous().requires_grad_(True) for x in (q, k, v))
    tdo = tt(do).transpose(1, 2).contiguous()
    B, H, Lq, D = tq.shape
    Hkv, Lkv = tk.shape[1], tk.shape[2]
    gqa = Hkv != H
    if impl == "torch:flex":
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        if _TORCH_FLEX is None:
            # One compiled entry per (shape, mask, grad mode): the default limit
            # of 8 made later shapes fall back to eager flex_attention (dense
            # scores, 10-100x slower or OOM).
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 1024
            torch._dynamo.config.accumulated_cache_size_limit = 4096
            _TORCH_FLEX = torch.compile(flex_attention, dynamic=False)
        if lens is not None:
            tl = torch.from_numpy(np.asarray(lens)).to(dev)
            bm = create_block_mask(lambda b, h, qi, ki: ki < tl[b], B, None, Lq, Lkv, device=dev)
        else:
            bm = None
        f = lambda: _TORCH_FLEX(tq, tk, tv, block_mask=bm, enable_gqa=gqa)
    else:
        backend = {"torch:flash": SDPBackend.FLASH_ATTENTION,
                   "torch:efficient": SDPBackend.EFFICIENT_ATTENTION}[impl]
        mask = None
        if lens is not None:
            if impl == "torch:flash":
                raise RuntimeError("SDPA flash takes no attention mask")
            tl = torch.from_numpy(np.asarray(lens)).to(dev)
            mask = (torch.arange(Lkv, device=dev)[None, None, None, :] < tl[:, None, None, None])
            mask = mask.expand(B, 1, Lq, Lkv)
        def f():
            with sdpa_kernel(backend):
                return torch.nn.functional.scaled_dot_product_attention(
                    tq, tk, tv, attn_mask=mask, enable_gqa=gqa)

    def fwd():
        with torch.no_grad():
            return f()

    def fwd_bwd():
        for t in (tq, tk, tv):
            t.grad = None
        out = f()
        out.backward(tdo)
        return out

    def timed(fn):
        fn(); fn(); torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)

    try:
        fwd_ms = timed(fwd)
        fb_ms = timed(fwd_bwd)
        out = fwd_bwd()
        back = lambda t: jnp.asarray(t.detach().float().transpose(1, 2).cpu().numpy())
        res = (fwd_ms, fb_ms, back(out), tuple(back(t.grad) for t in (tq, tk, tv)))
    finally:
        # Cached blocks of earlier cases; this case's tensors are released when
        # the caller drops them (see `_release_torch_memory`).
        torch.cuda.empty_cache()
    return res


def _release_torch_memory():
    """Return torch's cached blocks so the next JAX reference has the memory."""
    import torch

    torch.cuda.empty_cache()


def _time(fn, args, reps):
    jax.block_until_ready(fn(*args))  # compile + warm
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", nargs="*", default=["mla", "isab"])
    ap.add_argument("--impls", nargs="*", default=["flex:triton_split", "flex:triton_fused", "cudnn"])
    ap.add_argument("--impls-file", default=None, help="one impl spec per line (avoids commas in sbatch --export)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--label", default="")
    ap.add_argument("--unmasked", action="store_true", help="drop the KV-length mask (to compare with mosaic_fa3)")
    args = ap.parse_args()

    shapes = list(itertools.chain.from_iterable(GROUPS.get(s, [s]) for s in args.shapes))
    if args.impls_file:
        with open(args.impls_file) as fh:
            args.impls = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    impls = _impls(args.impls)
    ref_impl = _impls(["flex:triton_split"])["flex:triton_split"]
    dev = jax.devices()[0]
    meta = dict(label=args.label, host=socket.gethostname(), gpu=dev.device_kind,
                cc=str(getattr(dev, "compute_capability", "")), jax=jax.__version__)
    import probjax
    meta["probjax"] = probjax.__file__
    print(meta, flush=True)

    for sname in shapes:
        B, Lq, Lkv, H, D, masked = SHAPES[sname]
        masked = masked and not args.unmasked
        kq, kk, kv, kl, kd = jax.random.split(jax.random.key(0), 5)
        q = jax.random.normal(kq, (B, Lq, H, D), jnp.bfloat16)
        k = jax.random.normal(kk, (B, Lkv, H, D), jnp.bfloat16)
        v = jax.random.normal(kv, (B, Lkv, H, D), jnp.bfloat16)
        do = jax.random.normal(kd, (B, Lq, H, D), jnp.bfloat16)
        lens = (jax.random.randint(kl, (B,), Lkv // 2, Lkv + 1).astype(jnp.int32) if masked else None)

        def fwd_bwd(f, lens=lens, do=do):
            def loss(q_, k_, v_):
                return jnp.sum(f(q_, k_, v_, lens).astype(jnp.float32) * do.astype(jnp.float32))
            return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))

        ref_out = jax.jit(lambda q, k, v, lens=lens: ref_impl(q, k, v, lens))(q, k, v)
        ref_g = fwd_bwd(ref_impl)(q, k, v)
        for iname, f in impls.items():
            rec = dict(meta, shape=sname, B=B, Lq=Lq, Lkv=Lkv, H=H, D=D, masked=masked, impl=iname)
            if iname.startswith("torch:"):
                try:
                    fm, fbm, o, g = _torch_case(iname, q, k, v, do, lens, args.reps)
                    _release_torch_memory()
                    rec.update(fwd_ms=round(fm, 3), fwd_bwd_ms=round(fbm, 3), status="ok",
                               out_err=float(jnp.max(jnp.abs(o - ref_out.astype(jnp.float32)))),
                               grad_err=[float(jnp.max(jnp.abs(a - b.astype(jnp.float32))))
                                         for a, b in zip(g, ref_g, strict=True)])
                except Exception as exc:  # noqa: BLE001
                    rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
                print(json.dumps(rec), flush=True)
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                continue
            try:
                if iname.startswith("mosaic") and masked:
                    rec["note"] = "unmasked (no mask support)"
                unmasked = iname.startswith("mosaic")
                fj = jax.jit(lambda q, k, v, f=f, m=None if unmasked else lens: f(q, k, v, m))
                rec["fwd_ms"] = round(_time(fj, (q, k, v), args.reps), 3)
                gj = fwd_bwd((lambda q, k, v, _lens, f=f: f(q, k, v, None)) if unmasked else f)
                rec["fwd_bwd_ms"] = round(_time(gj, (q, k, v), args.reps), 3)
                if not iname.startswith("mosaic"):
                    o = fj(q, k, v)
                    g = gj(q, k, v)
                    rec["out_err"] = float(jnp.max(jnp.abs(o.astype(jnp.float32) - ref_out.astype(jnp.float32))))
                    rec["grad_err"] = [float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
                                       for a, b in zip(g, ref_g, strict=True)]
                rec["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                rec["status"] = "error"
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                traceback.print_exc()
            print(json.dumps(rec), flush=True)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
