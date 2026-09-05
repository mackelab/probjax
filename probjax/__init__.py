"""ProbJax: Probabilistic Computation in JAX.

ProbJax is a powerful library for probabilistic computation in JAX, designed to
simplify the development of probabilistic models and inference algorithms.

Key Modules
-----------
- :mod:`probjax.core`: Core functionality for tracing, inversion, and transformations
- :mod:`probjax.stats`: Probability distributions with SciPy-like API
- :mod:`probjax.nn`: Neural networks, normalizing flows, and diffusion models
- :mod:`probjax.inference`: MCMC, SMC, and filtering algorithms
- :mod:`probjax.utils`: ODE/SDE integration and utility functions

Example
-------
>>> import jax
>>> from probjax.stats import norm
>>>
>>> # Create a normal distribution
>>> normal = norm(loc=0.0, scale=1.0)
>>>
>>> # Sample from it
>>> key = jax.random.key(0)
>>> samples = normal.sample(key, shape=(1000,))
"""

__version__ = "0.1.0"

# Core imports
from probjax.core import (
    condition,
    custom_inverse,
    do,
    inverse,
    inverse_and_logabsdet,
    intervene,
    JaxprGraph,
    joint_sample,
    log_joint_fn,
    log_prob_fn,
    log_potential_fn,
    observe,
    scope,
    substitute,
    trace,
)

# Stats/distributions imports
from probjax.stats import (
    bernoulli,
    beta,
    binomial,
    bingham,
    cauchy,
    categorical,
    chi2,
    dirac,
    dirichlet,
    empirical,
    expon,
    gamma,
    gennorm,
    genpareto,
    geometric,
    indep,
    laplace,
    logistic,
    mixture,
    multivariate_normal,
    norm,
    pareto,
    poisson,
    rv_continuous,
    rv_discrete,
    rv_exponential_family,
    rv_generic,
    rv_multivariate,
    rv_spherical,
    skewnorm,
    t,
    transformed,
    truncnorm,
    uniform,
    vonmises,
    watson,
    wrapcauchy,
)

# Neural network imports
from probjax.nn import (
    AdditiveBinaryFuse,
    AdditiveCouplingFlow,
    AdditiveFuse,
    Affine,
    AffineCouplingFlow,
    BernsteinPolynomialFlow,
    AffineFuse,
    BinaryFuse,
    ConcatFuse,
    ContextFuse,
    ConvBlock,
    CouplingMLP,
    CouplingTransformer,
    DataLoader,
    DeepSet,
    DiffusionDenoiser,
    DropPath,
    EDM,
    Flip,
    FlowMatcher,
    GatedFuse,
    GaussianFourierEmbedding,
    GaussianizationFlow,
    InducedSelfAttention,
    LinearFlow,
    LRUCell,
    LRUModel,
    LearnablePosEncode,
    MambaCell,
    MaskedLinear,
    MaskedMLP,
    MLP,
    MeanFlowMatcher,
    MultiHeadAttention,
    MultinomialDiffusion,
    NeuralAutoregressiveFlow,
    NeuralSplineFlow,
    NFlow,
    NormalizingFlow,
    SumOfSquaresPolynomialFlow,
    UnconstrainedNeuralAutoregressiveFlow,
    OneHot,
    Permute,
    PosEncode,
    RecurrentCell,
    ResNet,
    RescaleConv,
    ResizeConv,
    ResnetBlock,
    RotaryPosEncode,
    Rotate,
    SSDCell,
    SSMModel,
    Sequential,
    SpatialSelfAttention,
    Transformer,
    UNet,
    VE,
    VP,
    build_denoising_loss,
    build_denoising_score_matching_loss,
    build_flow_matching_loss,
    build_score_matching_loss,
    build_sliced_score_matching_loss,
    build_target_score_matching_loss,
    build_time_dependent_denoising_loss,
    build_time_dependent_denoising_score_matching_loss,
    build_time_dependent_multinomial_diffusion_loss,
    build_time_dependent_score_matching_loss,
    build_time_dependent_sliced_score_matching_loss,
    build_time_dependent_target_score_matching_loss,
    chunkify,
)

# Import distributions as dist for convenience
from probjax import stats as distributions

# Re-export commonly used distributions at top level for convenience
Normal = norm
Gamma = gamma
Beta = beta
Uniform = uniform
Expon = expon
Laplace = laplace
Logistic = logistic
Cauchy = cauchy
MultivariateNormal = multivariate_normal
Dirichlet = dirichlet
Bernoulli = bernoulli
Binomial = binomial
Categorical = categorical
Poisson = poisson
Geometric = geometric

__all__ = [
    # Version
    "__version__",
    # Core
    "condition",
    "custom_inverse",
    "do",
    "inverse",
    "inverse_and_logabsdet",
    "intervene",
    "JaxprGraph",
    "joint_sample",
    "log_joint_fn",
    "log_prob_fn",
    "log_potential_fn",
    "observe",
    "scope",
    "substitute",
    "trace",
    # Stats module and base classes
    "distributions",
    "rv_generic",
    "rv_continuous",
    "rv_discrete",
    "rv_exponential_family",
    "rv_multivariate",
    "rv_spherical",
    # Continuous distributions
    "norm",
    "Normal",
    "gamma",
    "Gamma",
    "beta",
    "Beta",
    "uniform",
    "Uniform",
    "expon",
    "Expon",
    "laplace",
    "Laplace",
    "logistic",
    "Logistic",
    "cauchy",
    "Cauchy",
    "chi2",
    "t",
    "pareto",
    "skewnorm",
    "truncnorm",
    "gennorm",
    "genpareto",
    "vonmises",
    "watson",
    "bingham",
    "wrapcauchy",
    "multivariate_normal",
    "MultivariateNormal",
    "dirichlet",
    "Dirichlet",
    # Discrete distributions
    "bernoulli",
    "Bernoulli",
    "binomial",
    "Binomial",
    "categorical",
    "Categorical",
    "poisson",
    "Poisson",
    "geometric",
    "Geometric",
    "dirac",
    "empirical",
    # Higher-order distributions
    "transformed",
    "mixture",
    "indep",
    # Neural networks - architectures
    "MLP",
    "ResNet",
    "Transformer",
    "UNet",
    "DeepSet",
    "Sequential",
    "MaskedMLP",
    "CouplingMLP",
    "CouplingTransformer",
    "LRUModel",
    "SSMModel",
    # Neural networks - flows
    "NFlow",
    "NormalizingFlow",
    "AffineCouplingFlow",
    "AdditiveCouplingFlow",
    "NeuralSplineFlow",
    "BernsteinPolynomialFlow",
    "GaussianizationFlow",
    "NeuralAutoregressiveFlow",
    "SumOfSquaresPolynomialFlow",
    "UnconstrainedNeuralAutoregressiveFlow",
    "LinearFlow",
    "FlowMatcher",
    "MeanFlowMatcher",
    # Neural networks - diffusion
    "DiffusionDenoiser",
    "EDM",
    "VP",
    "VE",
    "MultinomialDiffusion",
    # Neural networks - layers
    "MultiHeadAttention",
    "MaskedLinear",
    "Affine",
    "ConcatFuse",
    "AdditiveFuse",
    "GatedFuse",
    "ResnetBlock",
    "ConvBlock",
    "GaussianFourierEmbedding",
    "PosEncode",
    "RotaryPosEncode",
    "LRUCell",
    "MambaCell",
    "SSDCell",
    "RecurrentCell",
    "OneHot",
    "Permute",
    "Flip",
    "Rotate",
    "DropPath",
    "InducedSelfAttention",
    "SpatialSelfAttention",
    "RescaleConv",
    "ResizeConv",
    "AdditiveBinaryFuse",
    "AffineFuse",
    "BinaryFuse",
    "ContextFuse",
    "LearnablePosEncode",
    # Neural networks - loss functions
    "build_flow_matching_loss",
    "build_denoising_loss",
    "build_score_matching_loss",
    "build_sliced_score_matching_loss",
    "build_target_score_matching_loss",
    "build_denoising_score_matching_loss",
    "build_time_dependent_denoising_loss",
    "build_time_dependent_score_matching_loss",
    "build_time_dependent_sliced_score_matching_loss",
    "build_time_dependent_target_score_matching_loss",
    "build_time_dependent_denoising_score_matching_loss",
    "build_time_dependent_multinomial_diffusion_loss",
    # Neural networks - utilities
    "DataLoader",
    "chunkify",
]
