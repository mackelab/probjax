"""Bijections in natural parameterization.

Every bijection here is a pure function taking ``x`` first and *already
constrained* natural parameters afterwards — knot positions, slopes, mixture
log-weights, and so on — never a raw unconstrained network output::

    rational_quadratic_spline(x, x_pos, y_pos, knot_slopes)
    mixture_cdf(x, log_weights, locs, scales)

Mapping a well-behaved parameter vector onto those natural parameters is the
job of the bijector configs in :mod:`probjax.nn.generative.nflows.config`.

Each family registers its analytic inverse (and log-determinant) with
:class:`~probjax.core.custom_inverse`, so ``inverse`` / ``inverse_and_logabsdet``
recover it exactly instead of re-deriving it from the jaxpr. Splines take a
**scalar** ``x`` with 1-D natural params; the monotone families broadcast over
leading batch dimensions. Callers that need batching should wrap with
``jnp.vectorize`` and an explicit signature.
"""

from probjax.stats.bijective.affine import (
    affine,
    rotate,
    shift,
)
from probjax.stats.bijective.monotone import (
    bernstein,
    deep_sigmoid,
    inv_bernstein,
    inv_deep_sigmoid,
    inv_mixture_cdf,
    inv_sos_polynomial,
    inv_unconstrained_monotone,
    mixture_cdf,
    sos_polynomial,
    unconstrained_monotone,
)
from probjax.stats.bijective.monotone_hermite_cubic import (
    inv_monotone_hermite_cubic_spline,
    monotone_hermite_cubic_spline,
    monotone_hermite_cubic_spline_and_logdet,
)
from probjax.stats.bijective.piecewise_affine import (
    inv_piecewise_affine_spline,
    piecewise_affine_spline,
    piecewise_affine_spline_and_logdet,
)
from probjax.stats.bijective.protocols import (
    InvertibleTransformProtocol,
    TransformedDistribution,
    TransformProtocol,
    ensure_invertible,
    forward_and_logdet,
)
from probjax.stats.bijective.rational_linear import (
    inv_rational_linear_spline,
    rational_linear_spline,
    rational_linear_spline_and_logdet,
)
from probjax.stats.bijective.rational_quadratic import (
    inv_rational_quadratic_spline,
    rational_quadratic_spline,
    rational_quadratic_spline_and_logdet,
)

__all__ = [
    # protocols
    "TransformProtocol",
    "InvertibleTransformProtocol",
    "TransformedDistribution",
    "ensure_invertible",
    "forward_and_logdet",
    # affine
    "affine",
    "rotate",
    "shift",
    # splines
    "monotone_hermite_cubic_spline",
    "monotone_hermite_cubic_spline_and_logdet",
    "inv_monotone_hermite_cubic_spline",
    "piecewise_affine_spline",
    "piecewise_affine_spline_and_logdet",
    "inv_piecewise_affine_spline",
    "rational_linear_spline",
    "rational_linear_spline_and_logdet",
    "inv_rational_linear_spline",
    "rational_quadratic_spline",
    "rational_quadratic_spline_and_logdet",
    "inv_rational_quadratic_spline",
    # monotone networks
    "bernstein",
    "inv_bernstein",
    "deep_sigmoid",
    "inv_deep_sigmoid",
    "mixture_cdf",
    "inv_mixture_cdf",
    "sos_polynomial",
    "inv_sos_polynomial",
    "unconstrained_monotone",
    "inv_unconstrained_monotone",
]
