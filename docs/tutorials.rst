Tutorials
=========

These tutorials demonstrate various features of ProbJax through Jupyter notebooks.

Core Functionality
------------------

Core ProbJax features including tracing, graph operations, and probabilistic programming.

.. toctree::
   :maxdepth: 1

   /examples/core/graph
   /examples/core/ppl
   /examples/core/trace_random

Distributions (Stats)
---------------------

Working with probability distributions.

.. toctree::
   :maxdepth: 1

   /examples/stats/distributions_basic
   /examples/stats/distributions_highlevel

Neural Networks: Architectures
------------------------------

Attention, transformers, the rest of ``probjax.nn.nets``, and multi-device sharding.

.. toctree::
   :maxdepth: 1

   /examples/nn/nets/attention
   /examples/nn/nets/transformer
   /examples/nn/nets/architectures
   /examples/nn/nets/sharding

Neural Networks: Generative Models
----------------------------------

Diffusion, flow matching, normalizing flows and autoregressive models.

.. toctree::
   :maxdepth: 1

   /examples/nn/generative/normalizing_flows
   /examples/nn/generative/diffusion
   /examples/nn/generative/flow_matching
   /examples/nn/generative/autoregressive
   /examples/nn/generative/simformer
   /examples/nn/flows/mean_flow_matching_mnist
   /examples/nn/diffusion/discrete_diffusion

Inference
---------

MCMC, SMC, and filtering algorithms.

.. toctree::
   :maxdepth: 1

   /examples/inference/mcmc
   /examples/inference/smc
   /examples/inference/kalman_filter
   /examples/inference/filters
   /examples/inference/bnn

Utilities
---------

Utility functions for ODE/SDE integration and special functions.

.. toctree::
   :maxdepth: 1

   /examples/utils/odeint
   /examples/utils/sdeint
   /examples/utils/betaincinv
