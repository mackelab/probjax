Tutorials
=========

The repository contains notebooks and scripts covering ProbJax's major
subpackages. Some older notebooks are still being migrated after API refactors;
the links below are limited to files currently present in the repository.

Core Functionality
------------------

* `Tracing random variables <https://github.com/mackelab/probjax/blob/main/examples/core/trace_random.ipynb>`_
* `Probabilistic programs <https://github.com/mackelab/probjax/blob/main/examples/core/ppl.ipynb>`_
* `JAXPR graphs <https://github.com/mackelab/probjax/blob/main/examples/core/graph.ipynb>`_

Statistics
----------

* `Distribution basics <https://github.com/mackelab/probjax/blob/main/examples/stats/distributions_basic.ipynb>`_
* `High-level distributions <https://github.com/mackelab/probjax/blob/main/examples/stats/distributions_highlevel.ipynb>`_

Inference
---------

* `MCMC <https://github.com/mackelab/probjax/blob/main/examples/inference/mcmc.ipynb>`_
* `SMC <https://github.com/mackelab/probjax/blob/main/examples/inference/smc.ipynb>`_
* `Kalman filtering <https://github.com/mackelab/probjax/blob/main/examples/inference/kalman_filter.ipynb>`_
* `Filtering <https://github.com/mackelab/probjax/blob/main/examples/inference/filters.ipynb>`_
* `Bayesian neural networks <https://github.com/mackelab/probjax/blob/main/examples/inference/bnn.ipynb>`_

Neural And Generative Models
----------------------------

The ``examples/nn`` tree includes normalizing-flow, diffusion, flow-matching,
attention, transformer, and sharding examples. These examples exercise rapidly
evolving research APIs, so check imports against :mod:`probjax.nn` before using
an older notebook as a template.

Numerical Utilities
-------------------

* `ODE integration <https://github.com/mackelab/probjax/blob/main/examples/utils/odeint.ipynb>`_
* `SDE integration <https://github.com/mackelab/probjax/blob/main/examples/utils/sdeint.ipynb>`_
* `Inverse incomplete beta <https://github.com/mackelab/probjax/blob/main/examples/utils/betaincinv.ipynb>`_

Notebooks are kept outside the Sphinx source tree and are therefore referenced
as repository paths rather than included in the documentation build.
