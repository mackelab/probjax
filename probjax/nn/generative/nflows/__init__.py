"""Normalizing flows and the bijective primitives they're built on.

Three layers:

* :mod:`probjax.stats.bijective` — the bijections themselves, as pure functions
  of ``x`` and *natural* parameters (knot positions, slopes, mixture
  log-weights), each with a registered custom inverse.
* :mod:`probjax.nn.generative.nflows.config` — configuration dataclasses along
  three axes: the **bijector** config maps an unconstrained parameter vector
  onto those natural parameters, the **conditioner** config builds the network
  that emits it, and the **mixing** config builds the layer between transforms.
* :mod:`probjax.nn.generative.nflows.models` — :class:`NFlow`, which assembles a
  config into a flow, plus named presets (``RealNVP``, ``MAF``, ``NSF``, …) that
  implement :class:`probjax.stats.base.DistributionAPI` and so plug into MCMC,
  SMC and filters as priors / proposals.

The parameter-producing networks live in
:mod:`probjax.nn.generative.nflows.autoregressive` and
:mod:`probjax.nn.generative.nflows.coupling`; the fixed and learnable primitives
(``Flip``, ``Permute``, ``Rotate``, …) live in
:mod:`probjax.nn.generative.nflows.bijective`.
"""

from probjax.nn.generative.nflows.autoregressive import (
    AutoregressiveMLP,
    AutoregressiveSSM,
    AutoregressiveTransformer,
)
from probjax.nn.generative.nflows.bijective import (
    Affine,
    ElementwiseMonotone,
    Flip,
    Permute,
    Rotate,
)
from probjax.nn.generative.nflows.config import (
    AffineBijectorConfig,
    AutoregressiveNFlowConfig,
    BernsteinBijectorConfig,
    BijectorConfigProtocol,
    ConditionerConfigProtocol,
    CouplingNFlowConfig,
    DeepSigmoidBijectorConfig,
    ElementwiseNFlowConfig,
    FlipMixingConfig,
    MixingConfigProtocol,
    MixtureCDFBijectorConfig,
    MLPConditionerConfig,
    MonotoneHermiteCubicSplineConfig,
    NFlowConfig,
    NoMixingConfig,
    PermuteMixingConfig,
    PiecewiseAffineSplineConfig,
    RationalLinearSplineConfig,
    RationalQuadraticSplineConfig,
    RotationMixingConfig,
    ShiftBijectorConfig,
    SSMConditionerConfig,
    SumOfSquaresBijectorConfig,
    TransformerConditionerConfig,
    UMNNBijectorConfig,
)
from probjax.nn.generative.nflows.coupling import CouplingMLP, CouplingTransformer
from probjax.nn.generative.nflows.models import (
    AdditiveAutoregressiveFlow,
    AdditiveCouplingFlow,
    AffineAutoregressiveFlow,
    AffineCouplingFlow,
    BernsteinPolynomialFlow,
    GaussianizationFlow,
    NeuralAutoregressiveFlow,
    NeuralSplineFlow,
    NFlow,
    NormalizingFlow,
    SplineAutoregressiveFlow,
    SplineCouplingFlow,
    SumOfSquaresPolynomialFlow,
    UnconstrainedNeuralAutoregressiveFlow,
    bpf,
    gf,
    maf,
    naf,
    nice,
    nsf,
    realnvp,
    sospf,
    unaf,
)

__all__ = [
    # bijective primitives
    "Affine",
    "ElementwiseMonotone",
    "Flip",
    "Permute",
    "Rotate",
    # parameter-producing networks
    "AutoregressiveMLP",
    "AutoregressiveSSM",
    "AutoregressiveTransformer",
    "CouplingMLP",
    "CouplingTransformer",
    # config protocols
    "BijectorConfigProtocol",
    "ConditionerConfigProtocol",
    "MixingConfigProtocol",
    # bijector configs
    "AffineBijectorConfig",
    "BernsteinBijectorConfig",
    "DeepSigmoidBijectorConfig",
    "MixtureCDFBijectorConfig",
    "MonotoneHermiteCubicSplineConfig",
    "PiecewiseAffineSplineConfig",
    "RationalLinearSplineConfig",
    "RationalQuadraticSplineConfig",
    "ShiftBijectorConfig",
    "SumOfSquaresBijectorConfig",
    "UMNNBijectorConfig",
    # conditioner configs
    "MLPConditionerConfig",
    "SSMConditionerConfig",
    "TransformerConditionerConfig",
    # mixing configs
    "FlipMixingConfig",
    "NoMixingConfig",
    "PermuteMixingConfig",
    "RotationMixingConfig",
    # flow configs
    "AutoregressiveNFlowConfig",
    "CouplingNFlowConfig",
    "ElementwiseNFlowConfig",
    "NFlowConfig",
    # composed flows
    "AdditiveAutoregressiveFlow",
    "AdditiveCouplingFlow",
    "AffineAutoregressiveFlow",
    "AffineCouplingFlow",
    "BernsteinPolynomialFlow",
    "GaussianizationFlow",
    "NeuralAutoregressiveFlow",
    "NeuralSplineFlow",
    "NFlow",
    "NormalizingFlow",
    "SplineAutoregressiveFlow",
    "SplineCouplingFlow",
    "SumOfSquaresPolynomialFlow",
    "UnconstrainedNeuralAutoregressiveFlow",
    # short aliases
    "bpf",
    "gf",
    "maf",
    "naf",
    "nice",
    "nsf",
    "realnvp",
    "sospf",
    "unaf",
]
