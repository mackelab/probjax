# Neural models

Flax NNX modules. The generative families share one interface — construct with
`nnx.Rngs`, `fit` to train, `as_dist()` for a frozen distribution — so they are
interchangeable. See [Density estimation](../guides/density-estimation.md).

This page covers the user-facing surface. `probjax.nn` also exports the building
blocks those models are assembled from (attention masks, block sizes, individual
bijector configs); read the source for those.

## Training and EMA

Generative models share `FitMixin`; other NNX modules can inherit it alongside
`nnx.Module` and implement `loss(rng, data, ...)`. The functional `probjax.nn.fit`
also works with plain parameter pytrees. Existing return values are unchanged:
module fitting returns losses, and functional fitting returns `(params, losses)`.

```python
import jax
import jax.numpy as jnp
from flax import nnx
from probjax.nn import FitCallback, FitMixin, MLP

class Autoencoder(MLP, FitMixin):
    def loss(self, rng, data):
        return jnp.mean((self(data) - data) ** 2)

model = Autoencoder([2, 16, 2], rngs=nnx.Rngs(0))
key = jax.random.key(1)
data = jax.random.normal(jax.random.key(2), (128, 2))

class Monitor(FitCallback):
    def on_step_end(self, state, info):
        print(int(state.step), float(info.loss))
        # state.params, state.ema_params, and state.model_state are device arrays.
        # Evaluate validation metrics or save a snapshot here.
        # Return False to stop before the next chunk.

result = model.fit(
    key, data,
    num_steps=1000,
    ema_decay=0.999,
    use_ema=True,
    callbacks=[Monitor()],
    callback_every=100,
    return_result=True,
)
```

EMA starts from the initial parameters and updates after every optimizer step:
`ema = decay * ema + (1 - decay) * params`. It is disabled by default.
`use_ema=True` installs the average into the model at completion;
`result.state.params` retains the raw optimizer parameters and
`result.state.ema_params` holds the average. Non-parameter state, including
BatchNorm statistics and RNG counters, follows the training updates and is not
averaged. Existing train/eval flags are respected; use `model.train()` when
needed. By default each `fit` starts a fresh optimizer and EMA. Passing
`initial_state=previous_result.state` preserves raw parameters, optimizer state,
RNG, EMA and module state; `num_steps` then requests additional updates.
Reuse the same kernel or optimizer and EMA configuration. A snapshot does not
serialize those settings or a data-loader position. Supplied `params` and `rng`
are ignored on resume; for module fitting the snapshot's variables take
precedence over subsequent edits to the model. The stopped flag is cleared.

Callbacks can override `on_fit_begin(state)`, `on_step_end(state, info)`, and
`on_fit_end(result)`. A plain callable `(state, info)` is also accepted for
step events. Begin and step hooks may return `False` to stop. Hooks run in
registration order, including when an earlier hook requests stopping; exceptions
propagate to the caller. End hooks run after successful completion, including
early stopping, but not after an exception. Snapshots are read-only; modify the
update rule through an Optax optimizer or a custom kernel rather than mutating
callback state.
The model itself receives final weights only after training; evaluate snapshots
rather than the original model inside callbacks.

`state.step` counts completed updates. Step callbacks receive post-update
parameters and the last pre-update `FitInfo` in their chunk. `info.loss` is the
training loss and `info.metrics` contains any auxiliary diagnostics. They run every
`callback_every` updates and after a final partial chunk. Larger chunks reduce
Python dispatch overhead. Without general callbacks, training remains a single
compiled scan. The existing `on_step(step, loss)` / `log_every` interface remains
available for lightweight host logging and retains its zero-based step index.

### State/info updates

`build_fit_kernel` exposes a pure interface suitable for JIT and `lax.scan`:
`kernel.init(params, key)` creates `FitState`, and
`kernel.step(key, state, batch)` returns `(new_state, FitInfo)`.
`fit(..., kernel=kernel)` runs the same update with batching and callbacks.
Custom `FitKernel` implementations can replace the complete update, while
preserving the state structure and advancing `state.step` once per update.

With `has_aux=True`, a loss returns `(loss, metrics)`. For a functional loss
that also updates `model_state`, return `(loss, (new_model_state, metrics))`.
Module losses only return `(loss, metrics)`; `FitMixin` handles NNX state.
Metrics are forwarded unchanged to callbacks and stacked by update in
`result.info.metrics`. `result.info.loss` aliases `result.losses`.

```python
import optax
from probjax.nn import build_fit_kernel, fit

def objective(params, key, batch):
    error = params["mean"] - batch
    return jnp.mean(error ** 2), {"mean_error": jnp.mean(error, axis=0)}

kernel = build_fit_kernel(
    objective, optax.adam(1e-3), has_aux=True, ema_decay=0.99,
)
state = kernel.init({"mean": jnp.zeros(2)}, key)
state, info = jax.jit(kernel.step)(state.rng, state, data)
result = fit(
    None, None, None, data,
    kernel=kernel, initial_state=state, num_steps=10,
    return_result=True,
)
assert int(result.state.step) == 11
assert result.info.metrics["mean_error"].shape == (10, 2)
```

The default kernel preserves the existing three-way key split (next key,
batch key, loss key), and records the next key in state. Resumed history contains
only the new updates; callback step counts remain absolute. The legacy
`on_step(step, loss)` hook continues to receive a scalar loss.

### JIT and explicit Python callbacks

The complete functional `fit` is JIT-compatible for array batches, including
state/info forwarding and resume. Keep loop configuration static, for example
by closing over it. `FitResult` is a pytree:

```python
@jax.jit
def train(params, key, batch):
    return fit(
        objective, params, key, batch,
        optimizer=optax.adam(1e-3), num_steps=10,
        has_aux=True, return_result=True,
    )

compiled = train({"mean": jnp.zeros(2)}, key, data)
assert compiled.info.loss.shape == (10,)
assert int(compiled.valid_steps) == 10
```

No callbacks means no host effects; the functional fit can also be
differentiated. Module fitting can run under `nnx.jit` when its preprocessing
hooks are traceable. Flow standardization supports this and only fits once.
Python data loaders and arbitrary host preprocessing belong outside JIT.

General callbacks default to host execution between chunks. To invoke them
inside JIT, explicitly pass `callback_mode="io"`:

```python
@jax.jit
def train_with_logging(params, key, batch):
    return fit(
        objective, params, key, batch,
        num_steps=10, has_aux=True, return_result=True,
        callbacks=[Monitor()], callback_mode="io", callback_every=5,
    )

logged = train_with_logging({"mean": jnp.zeros(2)}, key, data)
jax.effects_barrier()  # Wait for runtime callback effects when needed.
```

IO callbacks run at execution time, including repeated calls of the same
compiled function. They transfer snapshots to the host: ordinary leaves are
NumPy arrays, and typed PRNG keys are reconstructed on CPU. This is an explicit
cost; use the scalar `on_step` callback when only loss logging is needed. Ordered
IO callbacks do not support autodiff or `vmap`. Prefer ordinary host mode for
callbacks that launch substantial JAX computations.

Under JIT, and whenever `callback_mode="io"`, histories always have the requested
`num_steps` length. Early stopping leaves zero-filled padding. Use
`result.valid_steps` (updates performed by this call) or `result.valid` (a mask)
to exclude it. Ordinary eager host mode still trims histories. The JIT path does
not issue Python warnings or perform host value checks; inspect the valid losses
for non-finite values. Custom kernels must obey the fixed state/info structure
and one-step increment contract.

Subclass hooks let families customize training without overriding `fit`:

| Hook | Purpose |
| --- | --- |
| `_prepare_fit(data)` | Prepare model state and return the data to train on |
| `_fit_loss(rng, batch)` | Customize batch-to-loss dispatch |
| `_fit_param_filter()` | Select trainable NNX variables; defaults to `nnx.Param` |
| `_default_fit_kwargs()` | Supply defaults; explicit caller options win |
| `_fit_callbacks()` | Prepend model-specific callbacks |

Non-selected variables remain available to the model, and non-parameter updates
are carried between steps. Normalizing flows and autoregressive models use
`_prepare_fit` for their standardization step. Every fit snapshots the current
module graph, so replacing a submodule or changing non-parameter state between
calls is reflected in the next fit.

::: probjax.nn.fit
::: probjax.nn.FitMixin
::: probjax.nn.FitCallback
::: probjax.nn.FitState
::: probjax.nn.FitInfo
::: probjax.nn.FitKernel
::: probjax.nn.build_fit_kernel
::: probjax.nn.FitResult

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
categorical diffusion (`MultinomialDiffusion` and its presets) now require
`event_spec` at construction. It describes **one event**, excluding sample and
batch axes. An integer is shorthand for a feature vector; tuples describe image
or sequence events. Continuous diffusion and flow matching also accept pytrees of shapes and
`jax.ShapeDtypeStruct` leaves to specify dtypes.

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
Changing a default invalidates cached exports. Per-view overrides use separate
exports; each continuous sampler retains a polymorphic batch axis, so one export
handles different sample-batch sizes. Event dimensions themselves remain
concrete: changing them selects another export, rather than introducing symbolic
event axes. The backbone must already support the requested shape; changing
metadata does not resize learned projections or embeddings.

Categorical event shapes count token positions and exclude the one-hot class
axis: for example, `MultinomialCosineDM(net, num_classes=10, event_spec=(32,))`.
Their default event dtype is `int32`, and they require a single-array event spec.
Continuous defaults use `float32`; use a `ShapeDtypeStruct` or
`as_dist(dtype=...)` to select another floating dtype. `model.event_spec` exposes
the normalized default, and `model.event_shape` exposes just its shape(s).

**Migration:** replace `EDM(net)` with `EDM(net, event_spec=features)` (or an
appropriate shape/pytree). Existing explicit `as_dist(event_spec=...)` overrides
continue to work.

Flow matching and mean-flow use the same defaults and overrides:

```python
from probjax.nn import LinearFlow, LinearMeanFlow

class FlexibleVelocity(nnx.Module):
    def __call__(self, t, x, r=None, **kwargs):
        return 0.1 * x

flow = LinearFlow(FlexibleVelocity(), event_spec=3)
flow_view = flow.as_dist(num_steps=4)
sequence_flow = flow.as_dist(event_spec=(8, 3), num_steps=4)
flow.set_event_spec((16, 3))
mean_flow = LinearMeanFlow(FlexibleVelocity(), event_spec=3)
mean_flow_view = mean_flow.as_dist(num_steps=4)
```

Migrate existing `LinearFlow(net)` and `LinearMeanFlow(net)` constructors by
passing `event_spec=features` (or the event shape/pytree). Event metadata does
not resize network weights; the network must support each requested shape.

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

`DiffusionTransformer` adapts `Transformer` to `net(t, x, r=None, context=None)`.
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
weights. Images need an external patch/token embedding.

Subclass the existing Transformer factories or `projection_cls`, `fourier_cls`,
`time_mlp_cls`, and `position_cls` to customize the architecture. Explicit
constructor overrides take precedence. `position_cls=None` disables positional
encoding for permutation-equivariant inputs.

::: probjax.nn.DiffusionTransformer

### Continuous schedule audit

EDM, VE, VP and cosine are checked against their forward marginal/SDE identities,
inverse effective-noise maps, all three training targets and Gaussian-oracle
reverse ODE/SDE sampling. VP now includes the time-domain rescaling in its SDE
coefficients. VP/cosine low-noise calculations avoid float32 cancellation.
Generic schedules use `g² = d(std²)/dt - 2 (scale′/scale) std²`.

EDM preconditioning now normalizes by the signal scale for VP/cosine, including
the denoiser skip connection. Existing VP/cosine checkpoints used different
coefficients and may need retraining. Cosine's singular endpoint is bounded;
use the preset's finite effective-noise bounds for sampling. Numerical checks
with an exact Gaussian denoiser validate solver behavior, not learned sample
quality on arbitrary datasets.

`VSolverConfig` now uses the reverse SDE for `mode="sde"`; previously that path
had zero diffusion and was deterministic. Explicit DDIM remains deterministic.
Categorical linear, cosine, sigmoid and log-SNR schedules are also checked for
monotone retention and normalized, nonnegative transition probabilities.
The categorical linear/sigmoid factory defaults retain about 95% of the initial
signal at terminal time; choose stronger rates when a nearly independent base
prior is required. Their probability checks alone do not establish mixing.


Further sampling fixes: an overridden `t_max` now also selects the base-noise
scale. `sample_with_state` uses the supplied state for both initialization and
integration, and `model_state()` handles Python scalar leaves. Call
`dist.compile()` before wrapping sampling in an outer `jax.jit`; constructing
an export lazily during tracing is currently unsupported.

Exponential AB2 now handles nonuniform step sizes and uses the correct
phi-function weights and midpoint linear coefficient. Gaussian and nonlinear
mixture benchmarks, transformer scaling measurements, and remaining limitations
are recorded in `examples/nn/benchmark_results/diffusion_audit.md`.

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
