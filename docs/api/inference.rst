Inference Module
================

The inference module provides various inference algorithms for probabilistic models.

.. module:: probjax.inference

This module includes:

- MCMC (Markov Chain Monte Carlo) sampling
- SMC (Sequential Monte Carlo) methods
- Filtering and smoothing algorithms
- Rejection sampling

MCMC
----

.. autosummary::
   :toctree: generated

   mcmc.hmc.HMC
   mcmc.nuts.NUTS
   mcmc_runner.MCMCRunner

SMC
---

.. autosummary::
   :toctree: generated

   smc_runner.SMCRunner
   smc.SMC

Filtering and Smoothing
-----------------------

.. autosummary::
   :toctree: generated

   filtering.kalman_filter
   filtering.particle_filter

Detailed Documentation
----------------------

.. automodule:: probjax.inference.mcmc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.inference.mcmc_runner
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.inference.smc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.inference.smc_runner
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.inference.filtering
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.inference.rejection_sampling
   :members:
   :undoc-members:
   :show-inheritance:
