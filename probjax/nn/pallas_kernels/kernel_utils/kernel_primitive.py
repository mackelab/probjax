"""Re-export: the declarative primitive machinery lives in probjax.core.

Kept as an import location for the pallas kernels; the implementation moved
to :mod:`probjax.core.custom_primitives.sharded_primitive` so that core
primitives (custom_inverse, random_variable) can share it.
"""

from probjax.core.custom_primitives.sharded_primitive import *  # noqa: F401,F403
from probjax.core.custom_primitives.sharded_primitive import (  # noqa: F401
    _cp_general_batching,
    _validate_shardings,
    batch_shard,
)
