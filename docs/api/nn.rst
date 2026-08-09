Neural Networks Module
======================

The neural networks module provides layers, architectures, and normalizing flows built on top of Flax.

.. module:: probjax.nn

This module includes:

- Standard neural network architectures
- Specialized layers for normalizing flows
- Coupling and autoregressive layers
- Diffusion models
- Custom layer implementations

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

.. automodule:: probjax.nn.loss_fn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.nn.io_util
   :members:
   :undoc-members:
   :show-inheritance:
