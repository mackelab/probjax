# Transformer defaults and simple runtime checks

The standard profile is **model width 256, eight heads, 32 dimensions per head,
four layers, and 4× feed-forward expansion**. It is shared by `Transformer`,
`DiffusionTransformer`, direct autoregressive/coupling transformer layers,
`TransformerARConditionerConfig`, and `TransformerConditionerConfig`.

`attn_size` is the dimension of one head; the QKV projection width is
`num_heads * attn_size`. It may differ from the residual model width. Thus an
explicit compact profile with width 128 and eight 32D heads is supported.
Input-adaptive induced/spatial attention uses the input width as the total QKV
width when `attn_size=None`, rather than dividing by the head count twice.

```python
from flax import nnx

from probjax.nn import Transformer, TransformerARConditionerConfig

standard = Transformer(rngs=nnx.Rngs(0))
assert standard.model_dim == 256
assert standard.attention_blocks[0].head_dim == 32

compact = Transformer(model_dim=128, num_layers=2, rngs=nnx.Rngs(1))
ar_config = TransformerARConditionerConfig(model_dim=128, num_layers=2)
```

For test-sized models, explicitly set width/head/layer counts rather than
making production constructors silently shrink according to event size.
Callable width rules remain available as opt-in conditioner configuration.

## Measured CPU costs

Apple Silicon CPU, JAX 0.9.2, float32, batch four, 32 tokens, four layers,
three synchronized warm calls. All 36 forward/gradient cases were finite.
Attention uses the dense softmax kernel for these CPU measurements.
AR decoder weights are made nonzero before gradient timing. No models were
trained here, so the table is a capacity/runtime comparison, not a quality rank.

| Model | Width | Heads × head dim | Parameters | Forward ms | Loss + gradient ms |
| --- | ---: | --- | ---: | ---: | ---: |
| Transformer | 64 | 4 × 16 | 199,936 | 1.73 | 3.05 |
| Transformer | 128 | 8 × 32 | 1,056,768 | 2.72 | 5.47 |
| Transformer | 256 | 8 × 32 | 3,159,040 | 3.49 | 8.32 |
| AR conditioner | 64 | 4 × 16 | 200,192 | 1.88 | 3.15 |
| AR conditioner | 128 | 8 × 32 | 1,057,280 | 2.90 | 5.66 |
| AR conditioner | 256 | 8 × 32 | 3,160,064 | 4.04 | 8.92 |
| Diffusion transformer | 64 | 4 × 16 | 489,314 | 2.69 | 3.67 |
| Diffusion transformer | 128 | 8 × 32 | 1,872,802 | 3.49 | 7.17 |
| Diffusion transformer | 256 | 8 × 32 | 5,814,818 | 4.88 | 13.27 |

Width 256 is the standard capacity choice; use 128 when its memory/runtime
trade-off is preferable. Dropout remains explicit. The AR attention kernel is
also configurable; `dot_product_attention` is useful for small CPU workloads.

## Reproduce

```bash
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_nn_defaults
```

[Raw measurements](nn_defaults.json) include eight-token results, compilation
costs, parameter counts, and finite flags. For larger token counts and dtype
comparisons, use `benchmark_transformer_audit.py` as described in the
[earlier audit](diffusion_audit.md).
