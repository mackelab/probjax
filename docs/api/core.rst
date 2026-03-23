Core Module
===========

The core module provides fundamental functionality for probabilistic computation in JAX.

.. module:: probjax.core

This module includes:

- Function tracing and JAXPR manipulation
- Automatic function inversion
- Log-probability computation
- Custom primitives for probabilistic programming

Key Functions
-------------

.. autosummary::
   :toctree: generated

   transformation.inverse
   transformation.inverse_and_logabsdet
   transformation.trace
   transformation.log_prob_fn
   transformation.log_joint_fn
   transformation.condition
   transformation.observe
   transformation.intervene
   transformation.substitute
   transformation.do
   transformation.scope
   transformation.joint_sample
   transformation.log_potential_fn

Classes
-------

.. autosummary::
   :toctree: generated

   jaxpr_propagation.graph.JaxprGraph

Detailed Documentation
----------------------

.. automodule:: probjax.core.transformation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.core.jaxpr_propagation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.core.custom_primitives
   :members:
   :undoc-members:
   :show-inheritance:
