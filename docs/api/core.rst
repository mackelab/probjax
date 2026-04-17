Core Module
===========

The core module provides fundamental functionality for probabilistic computation in JAX.

.. module:: probjax.core

This module includes:

- Function tracing and JAXPR manipulation
- Automatic function inversion
- Log-probability computation
- Custom primitives for probabilistic programming
- Interventions, observations, and conditioning

Key Functions
-------------

.. autosummary::
   :toctree: generated

   inverse
   inverse_and_logabsdet
   trace
   log_prob_fn
   log_joint_fn
   log_potential_fn
   condition
   observe
   intervene
   do
   substitute
   joint_sample
   scope

Classes
-------

.. autosummary::
   :toctree: generated

   JaxprGraph
   custom_inverse

Detailed Documentation
----------------------

.. automodule:: probjax.core.transformation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.core.jaxpr_propagation.graph
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.core.custom_primitives.custom_inverse
   :members:
   :undoc-members:
   :show-inheritance:
