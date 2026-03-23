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

Key Classes
-----------

Architectures
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   nets.MLP
   nets.ResNet
   nets.UNet
   nets.Transformer
   nets.DeepSet
   nets.Sequential

Normalizing Flows
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   nets.NormalizingFlow
   nets.AffineCouplingFlow
   nets.AdditiveCouplingFlow
   nets.NeuralSplineFlow
   nets.BernsteinPolynomialFlow
   nets.NeuralAutoregressiveFlow
   nets.GaussianizationFlow
   nets.LinearFlow

Diffusion Models
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   nets.DiffusionDenoiser
   nets.EDM
   nets.VP
   nets.VE

Layers
~~~~~~

.. autosummary::
   :toctree: generated

   layers.MultiHeadAttention
   layers.MaskedLinear
   layers.Affine
   layers.ConcatFuse
   layers.AdditiveFuse
   layers.GatedFuse
   layers.ResnetBlock
   layers.ConvBlock
   layers.GaussianFourierEmbedding
   layers.PosEncode
   layers.RotaryPosEncode

Loss Functions
--------------

.. autosummary::
   :toctree: generated

   loss_fn.build_flow_matching_loss
   loss_fn.build_denoising_loss
   loss_fn.build_score_matching_loss
   loss_fn.build_sliced_score_matching_loss

Utilities
---------

.. autosummary::
   :toctree: generated

   io_util.DataLoader
   io_util.chunkify

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

.. automodule:: probjax.nn.pallas_kernels
   :members:
   :undoc-members:
   :show-inheritance:
