Neural Networks Module
======================

The neural networks module is built on Flax NNX. It re-exports standard NNX
layers and provides ProbJax architectures, generative models, and functional
training-loss builders.

.. module:: probjax.nn

This module includes:

- Standard neural network architectures
- Specialized layers for normalizing flows
- Coupling and autoregressive layers
- Continuous and multinomial diffusion models
- Continuous and mean flow matching
- Custom layer implementations

Generative models are NNX modules with model-specific ``loss(rng, data)``
methods. ``GenerativeModel.as_dist(event_spec, ...)`` creates a distribution
view backed by cached, shape-polymorphic exported sampling code; normalizing
flows also provide tractable densities.

Architectures
-------------

.. autosummary::
   :toctree: generated

   MLP
   ResNet
   Transformer
   UNet
   DeepSet
   Sequential
   MaskedMLP
   CouplingMLP
   CouplingTransformer
   LRUModel
   AutoregressiveMLP
   AutoregressiveTransformer

Normalizing Flows
-----------------

.. autosummary::
   :toctree: generated

   NormalizingFlow
   NFlow
   AffineCouplingFlow
   AdditiveCouplingFlow
   SplineCouplingFlow
   AffineAutoregressiveFlow
   AdditiveAutoregressiveFlow
   SplineAutoregressiveFlow
   NeuralSplineFlow
   NeuralAutoregressiveFlow
   UnconstrainedNeuralAutoregressiveFlow
   SumOfSquaresPolynomialFlow
   BernsteinPolynomialFlow
   GaussianizationFlow

Normalizing Flow Configs
------------------------

Three orthogonal axes: the bijector config maps an unconstrained parameter
vector onto the natural parameters of a bijection in
:mod:`probjax.stats.bijective`, the conditioner config builds the network that
emits it, and the mixing config builds the layer between transforms.

.. autosummary::
   :toctree: generated

   BijectorConfigProtocol
   ConditionerConfigProtocol
   MixingConfigProtocol
   NFlowConfig
   CouplingNFlowConfig
   AutoregressiveNFlowConfig
   ElementwiseNFlowConfig
   ShiftBijectorConfig
   AffineBijectorConfig
   RationalQuadraticSplineConfig
   RationalLinearSplineConfig
   MonotoneHermiteCubicSplineConfig
   PiecewiseAffineSplineConfig
   DeepSigmoidBijectorConfig
   UMNNBijectorConfig
   SumOfSquaresBijectorConfig
   BernsteinBijectorConfig
   MixtureCDFBijectorConfig
   MLPConditionerConfig
   TransformerConditionerConfig
   FlipMixingConfig
   PermuteMixingConfig
   RotationMixingConfig
   NoMixingConfig

Autoregressive Density Models
-----------------------------

``p(x) = prod_i p(x_i | x_<i)`` with each conditional a univariate family from
:mod:`probjax.stats`; hand the model a distribution class and the parameter
count, constraints, density and sampler all follow from it.

.. autosummary::
   :toctree: generated

   AutoregressiveModel
   MADE
   MixtureAutoregressive
   SplineAutoregressive
   HistogramAutoregressive
   CategoricalAutoregressive
   ARFamily
   ARConditionerConfig
   MLPARConditionerConfig
   TransformerARConditionerConfig

Flow Matching
-------------

.. autosummary::
   :toctree: generated

   FlowMatcher
   MeanFlowMatcher
   LinearFlow
   LinearMeanFlow

Diffusion Models
----------------

.. autosummary::
   :toctree: generated

   DiffusionDenoiser
   EDM
   VP
   VE
   MultinomialDiffusion
   MultinomialCosineDM
   MultinomialLogSNRDM

Generative Model Interface
--------------------------

.. autosummary::
   :toctree: generated

   GenerativeModel
   GenerativeModelProtocol

Layers
------

.. autosummary::
   :toctree: generated

   MultiHeadAttention
   MaskedLinear
   Affine
   ConcatFuse
   AdditiveFuse
   GatedFuse
   ResnetBlock
   ConvBlock
   GaussianFourierEmbedding
   PosEncode
   RotaryPosEncode
   LearnablePosEncode
   OneHot
   Permute
   Flip
   Rotate
   DropPath
   LRUCell
   MambaCell
   SSDCell
   RecurrentCell
   InducedSelfAttention
   SpatialSelfAttention
   RescaleConv
   ResizeConv
   AdditiveBinaryFuse
   AffineFuse
   BinaryFuse
   ContextFuse

Loss Functions
--------------

.. autosummary::
   :toctree: generated

   build_flow_matching_loss
   build_mean_flow_matching_loss
   build_mean_flow_matching_loss_from_schedule
   build_denoising_loss
   build_score_matching_loss
   build_sliced_score_matching_loss
   build_target_score_matching_loss
   build_denoising_score_matching_loss
   build_time_dependent_denoising_loss
   build_time_dependent_score_matching_loss
   build_time_dependent_sliced_score_matching_loss
   build_time_dependent_target_score_matching_loss
   build_time_dependent_denoising_score_matching_loss
   build_time_dependent_multinomial_diffusion_loss

Utilities
---------

.. autosummary::
   :toctree: generated

   DataLoader
   chunkify

Protocols and Configs
---------------------

.. autosummary::
   :toctree: generated

   FlowPreconditioningProtocol
   FlowSolverConfigProtocol
   FlowTrainingConfigProtocol
   InterpolationScheduleProtocol
   GaussianFlowPreconditioning
   CosineInterpolationSchedule
   QuadraticInterpolationSchedule
   LinearInterpolationSchedule
   LogitNormalFlowTrainingConfig
   UniformFlowTrainingConfig
   CategoricalPreconditioningProtocol
   CategoricalScheduleProtocol
   CategoricalTrainingConfigProtocol


Detailed Documentation
----------------------

.. automodule:: probjax.nn.nets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.nn.layers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.nn.losses
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.nn.generative
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.nn.io_util
   :members:
   :undoc-members:
   :show-inheritance:
