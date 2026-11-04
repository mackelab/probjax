"""Memory-scalable optimal transport and kernel statistics (:mod:`probjax.nn.ot`).

Built on the streaming log-sum-exp reduction
:func:`probjax.nn.pallas_kernels.kernel_logsumexp` (Pallas on GPU, blocked XLA
on CPU), so no N x M matrix is ever materialized.

- :func:`sinkhorn`, :func:`sinkhorn_divergence`, :func:`entropic_map`,
  :func:`sample_conditional` - log-domain entropic OT.
- :func:`mmd`, :func:`gaussian_kde_logpdf` - kernel two-sample statistic and KDE.
- :func:`ot_pairing`, :func:`apply_ot_pairing` - minibatch OT couplings (OT-CFM).
"""

from probjax.nn.ot.coupling import apply_ot_pairing, exact_assignment, ot_pairing
from probjax.nn.ot.mmd import (
    gaussian_kde_logpdf,
    gaussian_kde_logpdf_naive,
    kernel_sum_log,
    mmd,
    mmd_naive,
)
from probjax.nn.ot.sinkhorn import (
    SinkhornOutput,
    entropic_map,
    mean_sqeuclidean_cost,
    sample_conditional,
    sinkhorn,
    sinkhorn_divergence,
    transport_plan_dense,
)

__all__ = [
    "SinkhornOutput",
    "apply_ot_pairing",
    "entropic_map",
    "exact_assignment",
    "gaussian_kde_logpdf",
    "gaussian_kde_logpdf_naive",
    "kernel_sum_log",
    "mean_sqeuclidean_cost",
    "mmd",
    "mmd_naive",
    "ot_pairing",
    "sample_conditional",
    "sinkhorn",
    "sinkhorn_divergence",
    "transport_plan_dense",
]
