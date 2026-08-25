Inference Module
================

The inference module provides composable kernels, adaptation rules, and
compiled runners for probabilistic inference.

.. module:: probjax.inference

This module includes:

- MCMC (Markov Chain Monte Carlo) sampling
- SMC (Sequential Monte Carlo) methods
- Filtering and smoothing algorithms
- Rejection sampling

MCMC constructors return a kernel with ``init`` and ``init_params`` methods.
The ``MCMC`` runner scans that kernel and can collect samples or selected
diagnostics. SMC follows the same separation between kernels, parameters, and
the ``SMC`` runner. Filtering constructors return ``FilterKernel`` values with
``init`` and ``step`` callables.

MCMC Kernels
------------

.. autosummary::
   :toctree: generated

   hmc
   nuts
   mala
   mh
   gauss_rwmh
   imh
   gaussian_imh
   slice
   elliptical_slice
   latent_slice
   mclmc
   adjusted_mclmc
   adjusted_mclmc_dynamic
   dynamic_hmc
   arms
   a2rms
   pseudo_marginal
   sgld
   sghmc
   sgnht

MCMC Runner
-----------

.. autosummary::
   :toctree: generated

   MCMC
   MarkovKernel
   State
   Params

Adaptation and Warmup
---------------------

Adaptors are local ``init/update/finalize`` rules. They can be applied after
individual transitions with ``MCMC.adapt_step`` or scanned through finite
warmup. Warmup procedures are separate because they own a finite schedule or
replace the chain state.

.. autosummary::
   :toctree: generated

   Adaptor
   AdaptationResult
   AdaptationTrace
   Warmup
   WarmupResult
   adapt_step
   compose_adaptors
   step_size_adaptor
   mass_matrix_adaptor
   covariance_adaptor
   slice_step_size_adaptor
   window_warmup
   pathfinder_warmup
   mclmc_warmup

Variational Inference
---------------------

``flow_vi`` fits a normalizing flow to an unnormalized target by reparameterised
reverse KL, following the shape of ``blackjax.vi.meanfield_vi``. ``neutra``
reparameterises a target through such a flow so that any kernel above can sample
it in better-conditioned coordinates -- a change of variables, so MCMC stays
asymptotically exact however imperfect the flow is.

.. autosummary::
   :toctree: generated

   flow_vi
   neutra
   FlowVIState
   FlowVIInfo
   NeuTraTransform

SMC
---

.. autosummary::
   :toctree: generated

   smc
   adaptive_smc
   adaptive_smc_kernel
   persistent_smc
   persistent_smc_kernel
   adaptive_persistent_smc
   adaptive_persistent_smc_kernel
   path_smc
   GeometricPath
   PartialPosteriorsPath
   tuning
   particle_adaptor
   acceptance_rate_adaptor
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
