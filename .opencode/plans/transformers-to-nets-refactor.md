# Plan: `examples/nn/transformers/` → `examples/nn/nets/` (+ generative parts to `generative/`)

## Motivation

`examples/nn/transformers/` is the last un-refactored example folder. Its 8 notebooks are
mostly undocumented scratchpads (`attention.ipynb` = 72 cells, 1 markdown cell, benchmark
timings and half-finished Jacobian checks), several import APIs that no longer exist, and
three of them are really *generative models* that happen to use a transformer.

Stale imports found:

| notebook | broken import | current location |
|---|---|---|
| `flex_attention.ipynb` | `probjax.nn.attention` | `probjax.nn.layers.attention` |
| `simformer.ipynb` | `probjax.nn.nets.denoising_diffusion_model.EDM` | `probjax.nn.generative.EDM` |
| `simformer.ipynb` | `probjax.nn.loss_fn.denoising` | `probjax.nn.losses` |
| `transformers_binary_mnist.ipynb` | `probjax.nn.nets.denoising_diffusion_model.EDM` | `probjax.nn.generative.EDM` |
| `sharding.ipynb` | old `ShardingCfg` API (deleted, see `probjax/nn/sharding.py`) | ambient `jax.set_mesh` + logical axes |

## Decisions (confirmed with user)

1. Generative notebooks move to `examples/nn/generative/` — including `simformer`.
2. Notebooks are **rewritten from scratch** in the compact, narrated style of the already
   refactored `examples/nn/generative/*.ipynb` (6–10 cells, real markdown between cells).
3. Notebooks are **executed**, sized to run on CPU in seconds-to-a-minute.

## Target layout

```
examples/nn/
  nets/                        (new)
    attention.ipynb            <- new; the attention explainer (core ask)
    transformer.ipynb          <- new; Transformer module tour + 2 tasks
    architectures.ipynb        <- new; the rest of probjax.nn.nets
    sharding.ipynb             <- rewritten for the current sharding API
  generative/
    diffusion.ipynb            (unchanged)
    flow_matching.ipynb        (unchanged)
    normalizing_flows.ipynb    (unchanged)
    data.py                    (unchanged)
    autoregressive.ipynb       <- new; merge of transformers_mnist + _binary_mnist
    simformer.ipynb            <- rewritten against current API
  images/
    edm.py, fid.py             (unchanged)
    ar_transformer.py          <- git mv of transformers/transformers_mnist.py
```

Deleted: `examples/nn/transformers/` (whole directory).

`transformers_mnist.py` is a 683-line real multi-device training script — it is not a
notebook duplicate and must not be lost. `examples/nn/images/` is already the home for
full-scale image-model scripts (`edm.py`, `fid.py`), so it moves there as
`ar_transformer.py` with its imports fixed.

## Notebook contents

### `nets/attention.ipynb` — the explainer

The one the user explicitly asked for. Narrative, not benchmarks.

1. **What attention computes.** Hand-rolled `softmax(QKᵀ/√d)V` in ~5 lines of `jnp`, on a
   toy sequence where the right answer is visible (e.g. tokens attending to their nearest
   neighbour). Plot the attention matrix.
2. **`dot_product_attention`.** probjax's wrapper over the flax one; show it matches the
   hand-rolled version. Explain the `(batch, seq, heads, head_dim)` layout.
3. **`MultiHeadAttention`.** The `nnx.Module`: projections, heads, `normalize_qk`, and the
   probjax-only query-scaling variants (`SSMaxQueryScale`, `PerHeadQueryScale`,
   `QASSMaxQueryScale`) — one paragraph on why length-dependent scaling exists.
4. **Masks and biases as objects.** `CausalMask`, `LocalWindowMask`, `SeqLenMask`,
   `SameSegmentMask`, `KeyPaddingMask`, `MarginalizationMask`; `SymmetricAlibiBias`,
   `DistanceDecayBias`; composition via `ComposeMask` / `SumBias` / `NotMask`. Grid of
   `mask.dense(q_len, kv_len)` images — this is the strongest visual in the notebook and
   the part that is currently undocumented anywhere.
5. **Masks are not just cosmetic.** `jax.jacobian` of the output w.r.t. the input, showing
   the sparsity pattern matches the mask. (Salvaged from the old scratchpad, which had this
   idea but no explanation.) Small `seq_len=32` so it's instant.
6. **`flex_attention`.** The pallas/Triton fused path: same signature, same masks, never
   materializes the mask. State plainly that it needs a GPU; the cell runs with
   `interpret=True` at `seq_len=64` on CPU so the notebook executes, and asserts it agrees
   with `dot_product_attention` to bf16 tolerance. A markdown note gives the real GPU
   invocation and points at `probjax.utils.bmutil.benchmark` for timing, rather than
   shipping machine-specific numbers in the docs.
7. **Cross-attention and GQA.** `enable_gqa=True`, distinct q/kv lengths.

Absorbs `flex_attention.ipynb` entirely (its only content is items 4–6, undocumented).

### `nets/transformer.ipynb` — the architecture

1. `Transformer(model_dim, num_heads, num_layers, attn_size, ...)` — the block structure
   (pre-norm attention + widened MLP, drop-path), what each argument does.
2. Position encodings: `PosEncode` / `LearnablePosEncode` / `RotaryPosEncode`, side by side.
3. `enable_cross_attention` + `kv_in_features`; `context_dim` and `context_fusion_cls`
   (`AffineFuse` — this is the FiLM-conditioning path used by every generative model in
   `generative/`, and it is currently documented nowhere).
4. **Task A — sorting (encoder).** From `sorting_transformer.ipynb`: sort a length-16
   integer sequence. Trains to ~100% in a few hundred steps on CPU.
5. **Task B — causal decoding.** From `decoder_transformer.ipynb`: `CausalMask`, teach a
   binary sequence with known covariance structure, autoregressive sampling, compare the
   sample covariance to ground truth. Keeps the one genuinely good plot from that notebook.
6. `InducedSelfAttention` for set-structured inputs, briefly.

Absorbs `sorting_transformer.ipynb` and `decoder_transformer.ipynb`.

### `nets/architectures.ipynb` — the rest of `probjax.nn.nets`

Short tour so the folder is `nets`, not `transformers-in-disguise`: `MLP`, `ResNet`,
`MaskedMLP` (autoregressive masking — show the Jacobian is triangular), `DeepSet`,
`TimeMLP`, `UNet` (shape walkthrough on a 32×32 input, no training), `LRUModel` and the
`LRUCell`/`MambaCell`/`SSDCell` recurrent cells. One cell each, no training except where a
plot needs it. This recovers coverage lost when `examples/nn/rnns/` was deleted (still
referenced by `docs/tutorials.rst`).

### `nets/sharding.ipynb` — rewritten

The current one documents the deleted `ShardingCfg` API. Rewrite against
`probjax/nn/sharding.py`: logical axis names baked into modules, ambient `jax.set_mesh`,
`nnx.logical_axis_rules` vs `default_rules()`, megatron column/row MLP, head-sharded
attention, `constrain`/`replicate`. Uses 8 simulated CPU devices via
`XLA_FLAGS=--xla_force_host_platform_device_count=8` in the first cell, so it executes here.
Includes the Auto-vs-Explicit mesh caveat (jax 0.10 `make_mesh` defaults to Explicit, where
`with_sharding_constraint` asserts). Closes the follow-up recorded in the sharding memory.

### `generative/autoregressive.ipynb` — merged MNIST AR

`transformers_mnist.ipynb` (256-way categorical pixels) and `transformers_binary_mnist.ipynb`
(binarized) are the same notebook twice. One notebook, binarized MNIST as the main path
(faster, cleaner) with the 256-way categorical head as a variant cell: data → causal
transformer over 784 pixel tokens → training loop → bits/dim → `jax.vmap`'d autoregressive
sampling → 10×10 sample grid.

**Dependency issue:** `datasets` is not installed in `.venv` (neither is `sklearn`), so the
existing MNIST notebooks cannot currently run here. Plan: `uv add --dev datasets` and fetch
MNIST once (~11 MB), matching what `flows/mean_flow_matching_mnist.ipynb` already assumes.
If the install or download is unavailable, fall back to 16×16 synthetic binary digits
generated in JAX and note the swap — the notebook is about the model, not the dataset.
Training is scaled to a few hundred steps on CPU; samples will be blurry, and the notebook
says so and points at `images/ar_transformer.py` for the real run.

### `generative/simformer.ipynb` — rewritten

The interesting idea (transformer denoiser over *(node_id, value)* tokens, so a single
trained model does arbitrary conditionals via a marginalization mask) survives; the code
does not — it imports two deleted modules. Rewrite against `probjax.nn.generative.EDM`,
`probjax.nn.losses.build_time_dependent_denoising_loss`, and `MarginalizationMask`, on a
2D toy joint from `generative/data.py` so it trains on CPU. Show p(x₀|x₁), p(x₁|x₀) and the
joint from one model — that is the whole point of the architecture.

## Docs wiring

`docs/examples` is a symlink to `examples/`, and `docs/tutorials.rst` has an 11-entry
transformer/diffusion/flows toctree where **most entries already point at deleted
notebooks** (`/examples/nn/flows/ar`, `/examples/nn/rnns/rnns`, `/examples/nn/bnns/bnn`,
all five old diffusion notebooks…). Update the Neural Networks section to the real current
set: the four `nets/` notebooks, the five `generative/` notebooks, and the two leftovers.
This unbreaks the docs build, which is currently referencing ~12 missing files.

## Out of scope (flagged, not done)

`examples/nn/diffusion/discrete_diffusion.ipynb` and
`examples/nn/flows/mean_flow_matching_mnist.ipynb` are leftovers of the *previous* refactor
— single stale notebooks keeping two directories alive, and
`mean_flow_matching_mnist.ipynb` still imports the deleted
`probjax.nn.nets.flow_matching_model`. They belong in `generative/` as
`discrete_diffusion.ipynb` and `mean_flow_matching_mnist.ipynb`. Say the word and I will
fold them into this pass; otherwise `examples/nn/` keeps two vestigial folders.

## Execution order

1. `git mv transformers/transformers_mnist.py images/ar_transformer.py`, fix its imports,
   verify `--help` and a 2-step smoke run.
2. Write + execute `nets/attention.ipynb`.
3. Write + execute `nets/transformer.ipynb`.
4. Write + execute `nets/architectures.ipynb`.
5. Write + execute `nets/sharding.ipynb`.
6. Write + execute `generative/simformer.ipynb`.
7. Resolve the `datasets` dependency; write + execute `generative/autoregressive.ipynb`.
8. `git rm -r examples/nn/transformers/`.
9. Update `docs/tutorials.rst`; confirm every toctree entry resolves to a real file.

Each notebook is executed via `uv run jupyter nbconvert --execute --inplace`, so a failure
is loud rather than silently shipped with empty outputs. Steps 2–7 are independent, so any
one can be dropped or reordered without blocking the others.
