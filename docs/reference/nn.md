# Neural models

Flax NNX modules. The generative families share one interface — construct with
`nnx.Rngs`, `fit` to train, `as_dist()` for a frozen distribution — so they are
interchangeable. See [Density estimation](../guides/density-estimation.md).

This page covers the ProbJax public models and their building blocks. Standard
layers, activations, pooling functions, and initializers re-exported unchanged
from Flax NNX use the [Flax API reference](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/index.html).

For measured configuration comparisons and short runnable scripts, see
[Configuration benchmarks](../benchmarks/index.md). The
[transformer profile table](../benchmarks/nn_defaults.md) explains the standard
256-wide, eight-head defaults and the explicit compact options.

## Training and EMA

Training has three levels, each built on the one before:

* `train_step(loss_fn, optimizer)` is one pure update `(state, batch) -> (state,
  info)` for your own loop (sharding, buffer donation, custom logging).
* `fit(loss_fn, params, rng, data, **options)` minimises
  `loss_fn(params, rng, batch)` over a parameter pytree.
* `fit_module(model, rng, data, **options)` does the same for an NNX module,
  trained in place. It calls `model.loss(rng, batch)` unless you pass
  `loss=lambda model, rng, batch: ...`, and trains the variables selected by
  `trainable=` (default `nnx.Param`). Generative models wrap it as
  `model.fit(rng, data, **options)`.

The loss receives the whole batch. A mapping, an array or a tuple is one batch,
minibatched on the device; any other iterable (a list, a generator, a
`DataLoader`) is a stream, pulled one batch per update and never restarted.
Every level returns a `FitResult` with the trained parameters, the losses and
the final state.

```python
import jax
import jax.numpy as jnp
from flax import nnx
from probjax.nn import MLP, fit_module

model = MLP([2, 16, 2], rngs=nnx.Rngs(0))
key = jax.random.key(1)
data = jax.random.normal(jax.random.key(2), (128, 2))

def reconstruction(model, rng, batch):
    return jnp.mean((model(batch) - batch) ** 2)

def monitor(state, info):
    print(int(state.step), float(info.loss))
    # state.params and state.ema_params are device arrays: evaluate validation
    # metrics or save a snapshot here. Return False to stop.

result = fit_module(
    model, key, data, loss=reconstruction,
    num_steps=1000, ema_decay=0.999, use_ema=True,
    callback=monitor, every=100,
)
print(result.losses.shape)
```

The options are plain keywords. `num_steps` (1000) and `batch_size` (512;
`None` for the full batch; ignored for a stream) set the run. The optimizer
defaults to `default_optimizer(learning_rate, num_steps, schedule, clip_norm)`:
Adam with a one-cycle cosine schedule peaking at 1e-3 and gradients clipped to
a global norm of 10. `schedule` also accepts `"warmup_cosine"`, `"constant"`
or `None`, and any optax schedule; `optimizer=` replaces the whole thing with
your own `optax.GradientTransformation`. A generative family's
`fit_defaults()` is its recipe, and the keywords passed to `model.fit`
override only the entries they name.

EMA starts from the initial parameters and updates after every optimizer step:
`ema = decay * ema + (1 - decay) * params`. It is disabled by default.
`use_ema=True` installs the average into the model at completion;
`result.state.params` retains the raw optimizer parameters and
`result.state.ema_params` holds the average. Non-parameter state, including
BatchNorm statistics and RNG counters, follows the training updates and is not
averaged. Existing train/eval flags are respected; use `model.train()` when
needed. Each call starts a fresh optimizer; `state=previous_result.state`
continues raw parameters, optimizer state, RNG, EMA and module state, and
`num_steps` then counts additional updates. Pass the same optimizer settings
again: a snapshot does not store them or a data-loader position.

`callback(state, info)` runs on the host every `every` updates and after the
last one; returning `False` stops the run, and the history then holds exactly
the updates that ran. `info.loss` is the loss before the latest update and
`info.metrics` any auxiliary diagnostics. Without a callback, array data trains
in one compiled scan.

### Loss and metrics

A loss returns a scalar, or `(loss, metrics)` to have the metrics stacked into
`result.metrics` and handed to the callback.

```python
from probjax.nn import fit

def objective(params, key, batch):
    error = params["mean"] - batch["x"]
    return jnp.mean(error ** 2), {"mean_error": jnp.mean(error, axis=0)}

result = fit(objective, {"mean": jnp.zeros(2)}, key, {"x": data}, num_steps=10)
assert result.metrics["mean_error"].shape == (10, 2)
```

Every update splits its key three ways (next key, batch key, loss key) and
records the next key in the state, so array and stream runs over the same
batches agree exactly. Resumed history contains only the new updates.

### JIT

`fit` on array data without a callback is JIT-compatible, including metrics
and resume, and can be differentiated. `FitResult` is a pytree:

```python
@jax.jit
def train(params, key, batch):
    return fit(objective, params, key, batch, num_steps=10)

compiled = train({"mean": jnp.zeros(2)}, key, {"x": data})
assert compiled.losses.shape == (10,)
```

Module fitting runs under `nnx.jit` too; flow standardization supports this
and only fits once. Callbacks need the host and are rejected inside JIT, and
the JIT path skips the non-finite-loss warning: inspect the losses yourself.

For full control, drive `train_step` yourself:

```python
import optax
from probjax.nn.training import init_state, train_step

optimizer = optax.adam(1e-3)
step = jax.jit(train_step(objective, optimizer))
state = init_state({"mean": jnp.zeros(2)}, optimizer, key)
for _ in range(10):
    state, info = step(state, {"x": data})
```

::: probjax.nn.training
::: probjax.nn.fit
::: probjax.nn.fit_module
::: probjax.nn.train_step
::: probjax.nn.default_optimizer
::: probjax.nn.FitState
::: probjax.nn.FitInfo
::: probjax.nn.FitResult
::: probjax.nn.training.init_state
::: probjax.nn.training.is_stream

## Overriding architecture components

The main network classes resolve omitted builder arguments from class
attributes. Explicit constructor arguments take precedence, including `None`
where disabling a component is supported:

```python
from flax import nnx
from probjax.nn import MLP

class NormalizedMLP(MLP):
    norm_cls = nnx.LayerNorm
    linear_cls = nnx.Linear  # Replace with a compatible custom layer.

net = NormalizedMLP([4, 32, 2], rngs=nnx.Rngs(0))
plain = NormalizedMLP([4, 32, 2], norm_cls=None, rngs=nnx.Rngs(0))
```

| Network | Builder attributes and matching constructor arguments |
| --- | --- |
| `MLP`, `ResNet` | `linear_cls`, `norm_cls`, `context_fuse_cls` |
| `Transformer` | `mha_cls`, `mlp_cls`, `linear_cls`, `norm_cls`, `dropout_cls`, context and residual fusion builders |
| `SSMModel` | `recurrent_cls`, `mlp_cls`, `linear_cls`, `norm_cls`, `dropout_cls` |
| `TimeMLP` | `fourier_cls`, `time_mlp_cls`, `body_cls`, `norm_cls`, `context_fuse_cls` |
| `UNet` | `resnet_block_cls`, `conv_down_cls`, `conv_up_cls`, `attn_cls`, `conv_cls` |
| `DeepSet` | `dropout_cls`; `phi` and `rho` remain supplied modules |

Builders must accept the constructor and call arguments required by their role.
An SSM `linear_cls` controls its projections and block heads; the nested MLP can
be customized separately via `mlp_cls`. A Transformer `linear_cls` controls its
dense blocks; attention projections remain the responsibility of `mha_cls`.
Existing component sequences (such as per-layer MLP linears and U-Net sampling
layers) remain supported.

## Normalizing flows

::: probjax.nn.maf
::: probjax.nn.nsf
::: probjax.nn.naf
::: probjax.nn.unaf
::: probjax.nn.sospf
::: probjax.nn.bpf
::: probjax.nn.gf
::: probjax.nn.NormalizingFlow
::: probjax.nn.NFlowConfig

## Autoregressive models

::: probjax.nn.MADE
::: probjax.nn.MixtureAutoregressive
::: probjax.nn.SplineAutoregressive
::: probjax.nn.HistogramAutoregressive
::: probjax.nn.CategoricalAutoregressive
::: probjax.nn.Autoregressive
::: probjax.nn.ARFamily

## Diffusion and flow matching

Flow matching (`FlowMatcher`, `LinearFlow`), mean-flow (`MeanFlowMatcher`,
`LinearMeanFlow`), continuous diffusion (`DiffusionDenoiser`, `EDM`, `VP`, `VE`, `CosineDM`) and
categorical diffusion (`MultinomialDiffusion` and its presets) use
`event_spec` to describe **one event**, excluding sample and
batch axes. An integer is shorthand for a feature vector; tuples describe image
or sequence events. Continuous diffusion and flow matching also accept pytrees of shapes and
`jax.ShapeDtypeStruct` leaves to specify dtypes.

Continuous diffusion and flow matching accept an optional constructor default;
without one, pass `event_spec` to `as_dist`. Mean-flow and categorical diffusion
require the default at construction.

```python
from probjax.nn import EDM

class FlexibleDenoiser(nnx.Module):
    def __call__(self, t, x):
        return 0.1 * x

diffusion = EDM(FlexibleDenoiser(), event_spec=3, num_steps=3)
default_view = diffusion.as_dist()
samples = default_view.sample(key, (5,))
assert samples.shape == (5, 3)

# Override one distribution without changing the model's default.
sequence_view = diffusion.as_dist(event_spec=(8, 3))
assert sequence_view.sample(key, (2,)).shape == (2, 8, 3)
assert diffusion.event_shape == (3,)

# Change the default for future views. Existing views keep their bound shapes.
diffusion.set_event_spec((16, 3))  # Equivalently: diffusion.event_spec = (16, 3)
assert diffusion.as_dist().event_shape == (16, 3)
assert default_view.event_shape == (3,)
```

Defaults are static metadata, not a restriction on training or forward inputs.
Views with the same sampling options share one jitted operation per model; it
compiles once per sample shape, and a structural change to the model (a new
solver config, a replaced submodule) compiles it again. The backbone must
already support the requested shape; changing metadata does not resize learned
projections or embeddings.

Categorical event shapes count token positions and exclude the one-hot class
axis: for example, `MultinomialCosineDM(net, num_classes=10, event_spec=(32,))`.
Their default event dtype is `int32`, and they require a single-array event spec.
Continuous defaults use `float32`; use a `ShapeDtypeStruct` or
`as_dist(dtype=...)` to select another floating dtype. `model.event_spec` exposes
the normalized default, and `model.event_shape` exposes just its shape(s).

::: probjax.nn.EDM
::: probjax.nn.VP
::: probjax.nn.VE
::: probjax.nn.MultinomialDiffusion
::: probjax.nn.DiffusionDenoiser
::: probjax.nn.FlowMatcher
::: probjax.nn.MeanFlowMatcher
::: probjax.nn.LinearFlow

## Architectures

::: probjax.nn.MLP
::: probjax.nn.ResNet
::: probjax.nn.Transformer
::: probjax.nn.UNet
::: probjax.nn.DeepSet


### DiffusionTransformer

`DiffusionTransformer` adapts `Transformer` to
`net(t, x, r=None, context=None, local_context=None)`; `local_context` has shape
`[..., tokens, local_context_dim]` and is concatenated to the tokens at the input.
It projects token features into the model width, embeds time with Fourier
features and an MLP, and modulates each attention/MLP block using time plus
optional global context. Modulation starts at the identity and learns during
training. The output has the same feature width and token count as the input.

```python
from probjax.nn import DiffusionTransformer

backbone = DiffusionTransformer(
    3, model_dim=16, num_heads=2, num_layers=1, attn_size=8,
    time_embed_dim=16, fourier_dim=16, context_dim=4, rngs=nnx.Rngs(0),
)
output = backbone(0.5, jnp.ones((2, 8, 3)), context=jnp.ones((2, 4)))
assert output.shape == (2, 8, 3)
```

Inputs are `[..., tokens, features]`; time accepts a scalar or one value per
batch entry, including trailing singleton axes used by diffusion losses.
Global context excludes token axes and is required when `context_dim` is set.
Attention masks, biases, cross-attention inputs and deterministic/dropout flags
are forwarded to `Transformer`. Token counts can change without rebuilding
params. Images need an external patch/token embedding.

Subclass the existing Transformer factories or `projection_cls`, `fourier_cls`,
`time_mlp_cls`, and `position_cls` to customize the architecture. Explicit
constructor overrides take precedence. `position_cls=None` disables positional
encoding for permutation-equivariant inputs.

::: probjax.nn.DiffusionTransformer

## Pallas kernels

Use `probjax.nn.pallas_kernels` for attention, Flash3, kernel matrix products,
Mamba scans, and SSD. The primitive-based implementation is the sole kernel
package; the frozen legacy implementation has been removed.

Attention retains mask/bias pytrees, block sparsity, dropout, forward/reverse
differentiation, and batch/head sharding. Mamba retains batch sharding; SSD
retains batch and head/group sharding. Flash3 retains pipeline-emitter and
residual-output options, with compatibility fallbacks to JAX's public attention
API when private kernel entry points are unavailable. Hardware restrictions
still apply, including Flash3's compatible-GPU requirement.

Numerical reference coverage lives in `tests/test_pallas_references.py`.
Run its `mesh` tests explicitly to check sharded forward, gradients, and JVPs;
on CPU these use interpreted Pallas kernels across host devices.

Mamba uses an associative JAX scan for multi-tile GPU inputs because GPU
programs cannot safely exchange the TPU kernel's sequential carry. Single-tile
GPU inputs retain the fused Pallas kernel. The fallback supports autodiff;
its GPU performance has not yet been benchmarked.

### Attention mask and bias types

::: probjax.nn.pallas_kernels.AttentionBias
::: probjax.nn.pallas_kernels.AttentionMask
::: probjax.nn.pallas_kernels.BatchedLowRankBias
::: probjax.nn.pallas_kernels.BlockDiagonalCausalMask
::: probjax.nn.pallas_kernels.BlockDiagonalMask
::: probjax.nn.pallas_kernels.CausalAlibiBias
::: probjax.nn.pallas_kernels.CausalFromBottomRightMask
::: probjax.nn.pallas_kernels.CausalLocalWindowMask
::: probjax.nn.pallas_kernels.CausalMask
::: probjax.nn.pallas_kernels.ComposeMask
::: probjax.nn.pallas_kernels.ConstantBias
::: probjax.nn.pallas_kernels.DenseBias
::: probjax.nn.pallas_kernels.DistanceDecayBias
::: probjax.nn.pallas_kernels.FeatureMap
::: probjax.nn.pallas_kernels.IdentityBias
::: probjax.nn.pallas_kernels.KVLenMask
::: probjax.nn.pallas_kernels.KeyPaddingMask
::: probjax.nn.pallas_kernels.LearnedAlibiBias
::: probjax.nn.pallas_kernels.LearnedRelativePositionBias
::: probjax.nn.pallas_kernels.LocalWindowMask
::: probjax.nn.pallas_kernels.LowRankBias
::: probjax.nn.pallas_kernels.MarginalizationMask
::: probjax.nn.pallas_kernels.NoMask
::: probjax.nn.pallas_kernels.NotMask
::: probjax.nn.pallas_kernels.PerHeadScaleBias
::: probjax.nn.pallas_kernels.PrefixLMMask
::: probjax.nn.pallas_kernels.QKVLengthMask
::: probjax.nn.pallas_kernels.SameSegmentMask
::: probjax.nn.pallas_kernels.SeqLenMask
::: probjax.nn.pallas_kernels.SoftCappingBias
::: probjax.nn.pallas_kernels.SumBias
::: probjax.nn.pallas_kernels.SymmetricAlibiBias
::: probjax.nn.pallas_kernels.T5RelativePositionBias

### Block sizes and tuning

`TuningConfig` is JAX's `jax.experimental.pallas.ops.gpu.attention_mgpu.TuningConfig`,
re-exported for convenience and documented there.

::: probjax.nn.pallas_kernels.BlockSizes

### Attention kernels

::: probjax.nn.pallas_kernels.attention
::: probjax.nn.pallas_kernels.attention_with_pipeline_emitter
::: probjax.nn.pallas_kernels.mha
::: probjax.nn.pallas_kernels.mha_flash

### Mamba and SSD kernels

::: probjax.nn.pallas_kernels.compute_mamba_scan
::: probjax.nn.pallas_kernels.ssd
::: probjax.nn.pallas_kernels.ssd_linear_scan
::: probjax.nn.pallas_kernels.ssd_linear_scan_w_hidden_states
::: probjax.nn.pallas_kernels.ssd_linear_scan_w_timestep

### Kernel matrix products and KDE

::: probjax.nn.pallas_kernels.kernel_mv
::: probjax.nn.pallas_kernels.kernel_mv_naive
::: probjax.nn.pallas_kernels.rbf_kernel_mv
::: probjax.nn.pallas_kernels.rbf_kernel_mv_naive
::: probjax.nn.pallas_kernels.kde_density
::: probjax.nn.pallas_kernels.rbf_kde_density

### Linear attention

::: probjax.nn.pallas_kernels.residual_linear_attention
::: probjax.nn.pallas_kernels.residual_linear_attention_pallas
::: probjax.nn.pallas_kernels.residual_linear_attention_scan
::: probjax.nn.pallas_kernels.right_shift_and_zero_pad

## Layers and architectures

::: probjax.nn.AdditiveBinaryFuse
::: probjax.nn.AdditiveFuse
::: probjax.nn.AffineFuse
::: probjax.nn.BinaryFuse
::: probjax.nn.ConcatFuse
::: probjax.nn.ContextFuse
::: probjax.nn.ConvBlock
::: probjax.nn.DropPath
::: probjax.nn.ExpertSwiGLU
::: probjax.nn.FeedForward
::: probjax.nn.GatedFuse
::: probjax.nn.GaussianFourierEmbedding
::: probjax.nn.InducedSelfAttention
::: probjax.nn.LRUCell
::: probjax.nn.LRUModel
::: probjax.nn.LearnablePosEncode
::: probjax.nn.LowRankLinear
::: probjax.nn.LpNorm
::: probjax.nn.LpNormClip
::: probjax.nn.MambaCell
::: probjax.nn.MaskedLinear
::: probjax.nn.MaskedMLP
::: probjax.nn.MoEAux
::: probjax.nn.MoELayer
::: probjax.nn.MultiHeadAttention
::: probjax.nn.MultiHeadLatentAttention
::: probjax.nn.OneHot
::: probjax.nn.PerHeadQueryScale
::: probjax.nn.PosEncode
::: probjax.nn.QASSMaxQueryScale
::: probjax.nn.RecurrentCell
::: probjax.nn.RescaleConv
::: probjax.nn.ResizeConv
::: probjax.nn.ResnetBlock
::: probjax.nn.RotaryPosEncode
::: probjax.nn.SSDCell
::: probjax.nn.SSMModel
::: probjax.nn.SSMaxQueryScale
::: probjax.nn.Sequential
::: probjax.nn.SpatialSelfAttention
::: probjax.nn.TimeMLP
::: probjax.nn.load_balance_loss

## Flow building blocks and configuration

::: probjax.nn.CirculantDiagonal
::: probjax.nn.CirculantDiagonalMixingConfig
::: probjax.nn.CompositeMixingConfig
::: probjax.nn.AdditiveAutoregressiveFlow
::: probjax.nn.AdditiveCouplingFlow
::: probjax.nn.Affine
::: probjax.nn.AffineAutoregressiveFlow
::: probjax.nn.AffineBijectorConfig
::: probjax.nn.AffineCouplingFlow
::: probjax.nn.AutoregressiveMLP
::: probjax.nn.AutoregressiveNFlowConfig
::: probjax.nn.AutoregressiveSSM
::: probjax.nn.AutoregressiveTransformer
::: probjax.nn.BernsteinBijectorConfig
::: probjax.nn.BernsteinPolynomialFlow
::: probjax.nn.BijectorConfigProtocol
::: probjax.nn.ConditionerConfigProtocol
::: probjax.nn.CouplingMLP
::: probjax.nn.CouplingNFlowConfig
::: probjax.nn.CouplingTransformer
::: probjax.nn.DeepSigmoidBijectorConfig
::: probjax.nn.ElementwiseNFlowConfig
::: probjax.nn.Flip
::: probjax.nn.FlipMixingConfig
::: probjax.nn.GaussianizationFlow
::: probjax.nn.MLPConditionerConfig
::: probjax.nn.MixingConfigProtocol
::: probjax.nn.MixtureCDFBijectorConfig
::: probjax.nn.MonotoneHermiteCubicSplineConfig
::: probjax.nn.NFlow
::: probjax.nn.NeuralAutoregressiveFlow
::: probjax.nn.NeuralSplineFlow
::: probjax.nn.NoMixingConfig
::: probjax.nn.Permute
::: probjax.nn.PermuteMixingConfig
::: probjax.nn.PiecewiseAffineSplineConfig
::: probjax.nn.RationalLinearSplineConfig
::: probjax.nn.RationalQuadraticSplineConfig
::: probjax.nn.Rotate
::: probjax.nn.RotationMixingConfig
::: probjax.nn.SSMConditionerConfig
::: probjax.nn.ShiftBijectorConfig
::: probjax.nn.SplineAutoregressiveFlow
::: probjax.nn.SplineCouplingFlow
::: probjax.nn.SumOfSquaresBijectorConfig
::: probjax.nn.SumOfSquaresPolynomialFlow
::: probjax.nn.TransformerConditionerConfig
::: probjax.nn.UMNNBijectorConfig
::: probjax.nn.UnconstrainedNeuralAutoregressiveFlow
::: probjax.nn.nice
::: probjax.nn.realnvp

## Generative extension points and schedules

::: probjax.nn.ARConditionerConfig
::: probjax.nn.AutodiffInterpolationSchedule
::: probjax.nn.CategoricalEDMPreconditioning
::: probjax.nn.CategoricalPreconditioningProtocol
::: probjax.nn.CategoricalScheduleProtocol
::: probjax.nn.CategoricalTrainingConfigProtocol
::: probjax.nn.CosineInterpolationSchedule
::: probjax.nn.FlowPairTrainingConfigProtocol
::: probjax.nn.FlowPreconditioningProtocol
::: probjax.nn.FlowSolverConfigProtocol
::: probjax.nn.FlowTrainingConfigProtocol
::: probjax.nn.GaussianFlowPreconditioning
::: probjax.nn.GeneralInterpolationSchedule
::: probjax.nn.GenerativeModel
::: probjax.nn.GenerativeModelProtocol
::: probjax.nn.ImportanceContinuousTimeTrainingConfig
::: probjax.nn.InterpolationScheduleProtocol
::: probjax.nn.LinearFlowSolverConfig
::: probjax.nn.LinearInterpolationSchedule
::: probjax.nn.LinearMeanFlow
::: probjax.nn.LogitNormalFlowTrainingConfig
::: probjax.nn.MLPARConditionerConfig
::: probjax.nn.MultinomialCosineDM
::: probjax.nn.MultinomialDiffusionSchedule
::: probjax.nn.MultinomialLogSNRDM
::: probjax.nn.QuadraticInterpolationSchedule
::: probjax.nn.RhoFlowSolverConfig
::: probjax.nn.SSMARConditionerConfig
::: probjax.nn.SigmoidPairFlowTrainingConfig
::: probjax.nn.TransformerARConditionerConfig
::: probjax.nn.UniformContinuousTimeTrainingConfig
::: probjax.nn.UniformFlowTrainingConfig
::: probjax.nn.made

## Loss builders and data utilities

::: probjax.nn.build_denoising_loss
::: probjax.nn.build_denoising_score_matching_loss
::: probjax.nn.build_flow_matching_loss
::: probjax.nn.build_mean_flow_matching_loss
::: probjax.nn.build_mean_flow_matching_loss_from_schedule
::: probjax.nn.build_score_matching_loss
::: probjax.nn.build_sliced_score_matching_loss
::: probjax.nn.build_target_score_matching_loss
::: probjax.nn.build_time_dependent_denoising_loss
::: probjax.nn.build_time_dependent_denoising_score_matching_loss
::: probjax.nn.build_time_dependent_multinomial_diffusion_loss
::: probjax.nn.build_time_dependent_score_matching_loss
::: probjax.nn.build_time_dependent_sliced_score_matching_loss
::: probjax.nn.build_time_dependent_target_score_matching_loss
::: probjax.nn.DataLoader
::: probjax.nn.chunkify
