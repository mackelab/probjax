"""Frozen pre-primitive pallas implementation, kept for A/B validation.

NOT imported by the parent package: importing this module registers the
legacy global batching rule for ``custom_partitioning_p`` (cp_utils), which
the live implementation no longer needs. Import explicitly for comparisons::

    from probjax.nn.pallas_kernels.old_pallas import mha as mha_old

Delete this package once the new primitive-based kernels are validated on a
multi-GPU machine (see tests/test_pallas_old_vs_new.py).
"""

from probjax.nn.pallas_kernels.old_pallas.kernels import (  # noqa: F401
    BlockSizes,
    mha,
    attention,
    mha_flash,
    compute_mamba_scan,
    ssd,
)
from probjax.nn.pallas_kernels.old_pallas.kernels.kernel_mv import (  # noqa: F401
    kernel_mv,
    kernel_mv_naive,
    rbf_kde_density,
    rbf_kernel_mv,
)

# Importing the legacy kernels registered their heuristic batching rule for
# custom_partitioning_p globally; restore the live general rule so the new
# primitive-based kernels keep working correctly under vmap. (Legacy kernels
# should be compared un-vmapped; their vmap path relied on the heuristic.)
from probjax.nn.pallas_kernels.kernel_utils.kernel_primitive import (
    register_general_cp_batching as _register_general_cp_batching,
)

_register_general_cp_batching()
