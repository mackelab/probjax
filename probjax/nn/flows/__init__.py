"""Flow-based generative models and the bijective primitives they're built on.

This sub-package consolidates everything needed to construct a normalizing
flow:

* :mod:`probjax.nn.flows.bijective` — bijective primitives (``Affine``,
  ``Flip``, ``Permute``, ``Rotate``).
* :mod:`probjax.nn.flows.autoregressive` /
  :mod:`probjax.nn.flows.coupling` — neural architectures that produce the
  parameters of those bijectors.
* :mod:`probjax.nn.flows.normalizing_flows` — composed flow distributions
  (``NormalizingFlow``, ``RealNVP``, ``MAF``, ``NSF``, …) that implement
  :class:`probjax.stats.base.DistributionAPI` and so plug into MCMC, SMC,
  filters as priors / proposals.
"""

from probjax.nn.flows.autoregressive import (
    AutoregressiveMLP,
    AutoregressiveTransformer,
)
from probjax.nn.flows.bijective import Affine, Flip, Permute, Rotate
from probjax.nn.flows.coupling import CouplingMLP, CouplingTransformer
from probjax.nn.flows.normalizing_flows import (
    AdditiveAutoregressiveFlow,
    AdditiveCouplingFlow,
    AffineAutoregressiveFlow,
    AffineCouplingFlow,
    BernsteinPolynomialFlow,
    GaussianizationFlow,
    NeuralAutoregressiveFlow,
    NeuralSplineFlow,
    NormalizingFlow,
    NormalizingFlowsOnToriAndSpheres,
    SplineAutoregressiveFlow,
    SplineCouplingFlow,
    SumOfSquaresPolynomialFlow,
    UnconstrainedNeuralAutoregressiveFlow,
    bpf,
    gf,
    maf,
    naf,
    ncsf,
    nice,
    nsf,
    realnvp,
    sospf,
    unaf,
)

__all__ = [
    # bijective primitives
    "Affine",
    "Flip",
    "Permute",
    "Rotate",
    # parameter-producing networks
    "AutoregressiveMLP",
    "AutoregressiveTransformer",
    "CouplingMLP",
    "CouplingTransformer",
    # composed flows
    "AdditiveAutoregressiveFlow",
    "AdditiveCouplingFlow",
    "AffineAutoregressiveFlow",
    "AffineCouplingFlow",
    "BernsteinPolynomialFlow",
    "GaussianizationFlow",
    "NeuralAutoregressiveFlow",
    "NeuralSplineFlow",
    "NormalizingFlow",
    "NormalizingFlowsOnToriAndSpheres",
    "SplineAutoregressiveFlow",
    "SplineCouplingFlow",
    "SumOfSquaresPolynomialFlow",
    "UnconstrainedNeuralAutoregressiveFlow",
    # short aliases
    "bpf",
    "gf",
    "maf",
    "naf",
    "ncsf",
    "nice",
    "nsf",
    "realnvp",
    "sospf",
    "unaf",
]
