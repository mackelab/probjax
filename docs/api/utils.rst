Utils Module
============

The utils module exposes numerical helpers used throughout ProbJax.

.. module:: probjax.utils

This module includes:

- ODE/SDE integration
- Linear algebra utilities
- Interpolation functions
- Graph utilities
- Special mathematical functions
- Root finding

The top-level module exports the functions listed below. Lower-level ODE and
SDE solver configuration lives in ``probjax.utils.odeutil`` and
``probjax.utils.sdeutil``.

ODE/SDE Integration
-------------------

.. autosummary::
   :toctree: generated

   odeint
   sdeint

Linear Algebra
--------------

.. autosummary::
   :toctree: generated

   cholesky_update
   mv_diag_or_dense

Interpolation
-------------

.. autosummary::
   :toctree: generated

   linear_interpolation
   polynomial_interpolation

Graph Utilities
---------------

.. autosummary::
   :toctree: generated

   find_ancestors_jax
   faithfull_mask

Special Functions
-----------------

.. autosummary::
   :toctree: generated

   betaincinv
   digammainv
   gammaincinv

Root Finding
------------

.. autosummary::
   :toctree: generated

   newton_raphson
   root

Function Utilities
------------------

.. autosummary::
   :toctree: generated

   split_drift

JAX Utilities
-------------

.. autosummary::
   :toctree: generated

   ravel_args

Detailed Documentation
----------------------

.. automodule:: probjax.utils.odeint
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.utils.sdeint
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.utils.linalg
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.utils.interpolation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.utils.graph
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.utils.solver
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.utils.functions
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.utils.jaxutils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.utils.containers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.utils.special
   :members:
   :undoc-members:
   :show-inheritance:
