Inference Module
================

The inference module provides various inference algorithms for probabilistic models.

.. module:: probjax.inference

This module includes:

- MCMC (Markov Chain Monte Carlo) sampling
- SMC (Sequential Monte Carlo) methods
- Filtering and smoothing algorithms
- Rejection sampling

MCMC Kernels
------------

.. autosummary::
   :toctree: generated

   mcmc.hmc
   mcmc.nuts
   mcmc.mala
   mcmc.mh
   mcmc.gauss_rwmh
   mcmc.imh
   mcmc.gaussian_imh
   mcmc.slice
   mcmc.elliptical_slice
   mcmc.latent_slice
   mcmc.mclmc
   mcmc.adjusted_mclmc
   mcmc.adjusted_mclmc_dynamic
   mcmc.dynamic_hmc
   mcmc.arms
   mcmc.a2rms
   mcmc.pseudo_marginal
   mcmc.sgld
   mcmc.sghmc
   mcmc.sgnht

MCMC Runner
-----------

.. autosummary::
   :toctree: generated

   MCMC
   MarkovKernel
   State
   Params

SMC
---

.. autosummary::
   :toctree: generated

   smc.smc
   smc.adaptive_smc
   smc.adaptive_smc_kernel
   smc.persistent_smc
   smc.persistent_smc_kernel
   smc.adaptive_persistent_smc
   smc.adaptive_persistent_smc_kernel
   smc.path_smc
   smc.GeometricPath
   smc.PartialPosteriorsPath
   smc.tuning
   SMC

Filtering and Smoothing
-----------------------

.. autosummary::
   :toctree: generated

   kalman_filter
   extended_kalman_filter
   ukf
   sq_kalman_filter
   rank_reduced_kalman_filter
   ParticleFilter
   particle_smoother
   rauch_tung_stribel_smoother
   smooth
   FilterAPI
   FilterInfo
   FilterKernel
   FilterState
   make_filter_api

Rejection Sampling
------------------

.. autosummary::
   :toctree: generated

   ars
   ARSState
   init_ars_state
   update_ars_state
   RejectionSampler

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

.. automodule:: probjax.inference.rejection
   :members:
   :undoc-members:
   :show-inheritance:
