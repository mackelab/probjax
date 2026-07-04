"""Cross-family training losses.

Family-specific losses (denoising, flow-matching, mean-flow-matching,
multinomial-diffusion) live alongside their model class in
:mod:`probjax.nn.diffusion.<family>`.
"""

from probjax.nn.losses.score_matching import (
    build_score_matching_loss,
    build_time_dependent_score_matching_loss,
)
from probjax.nn.losses.sliced_score_matching import (
    build_sliced_score_matching_loss,
    build_time_dependent_sliced_score_matching_loss,
)
from probjax.nn.losses.target_score_matching import (
    build_target_score_matching_loss,
    build_time_dependent_target_score_matching_loss,
)

__all__ = [
    "build_score_matching_loss",
    "build_sliced_score_matching_loss",
    "build_target_score_matching_loss",
    "build_time_dependent_score_matching_loss",
    "build_time_dependent_sliced_score_matching_loss",
    "build_time_dependent_target_score_matching_loss",
]
