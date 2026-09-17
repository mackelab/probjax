"""Single entry point for parameter-dict helpers in inference.

Historically these lived in two dialects: :mod:`probjax.inference.adaptation`
(``get_param``/``replace_params`` for Mapping/NamedTuple/dataclass updates)
and :mod:`probjax.inference.smc.base` (``_filter_kwargs``/
``_params_to_dict``/``_wrap_params``/``_ensure_param_batch`` for
convert/batch/wrap). Import from here to avoid depending on either dialect
module directly.
"""

from probjax.inference.adaptation import get_param as get
from probjax.inference.adaptation import replace_params as replace
from probjax.inference.smc.base import _ensure_param_batch as ensure_batch
from probjax.inference.smc.base import _filter_kwargs as filter_kwargs
from probjax.inference.smc.base import _params_to_dict as to_dict
from probjax.inference.smc.base import _wrap_params as wrap_params

__all__ = [
    "ensure_batch",
    "filter_kwargs",
    "get",
    "replace",
    "to_dict",
    "wrap_params",
]
