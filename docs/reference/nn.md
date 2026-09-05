# Neural models

Flax NNX modules. The generative families share one interface — construct with
`nnx.Rngs`, `fit` to train, `as_dist()` for a frozen distribution — so they are
interchangeable. See [Density estimation](../guides/density-estimation.md).

This page covers the user-facing surface. `probjax.nn` also exports the building
blocks those models are assembled from (attention masks, block sizes, individual
bijector configs); read the source for those.

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
