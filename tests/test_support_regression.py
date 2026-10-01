"""Regression tests for distribution ``support`` correctness.

P0-1: ``pareto.support`` used to return the whole-real-line constraint
(``real``) instead of the ``[b, inf)`` interval. The same copy-paste bug
existed in ``chi2`` (true support ``[loc, inf)``) and ``truncnorm``
(true support ``[a, b]``). All other ``rv_continuous`` subclasses with a
whole-real-line support now inherit the default from ``rv_continuous``.
"""

import jax.numpy as jnp

from probjax.stats import chi2, norm, pareto, truncnorm
from probjax.stats.constraints import Interval, Real


def _contains(support, x):
    return jnp.asarray(x) in support


def test_pareto_support_is_bounded_below_by_b():
    s = pareto.support(b=2.0)
    assert isinstance(s, Interval), f"expected Interval, got {type(s).__name__}"
    assert float(s.lower) == 2.0
    assert float(s.upper) == float("inf")
    assert not _contains(s, 1.0), "1.0 must be outside Pareto(b=2.0) support"
    assert _contains(s, 2.0), "lower bound b must be inside the support"
    assert _contains(s, 3.0), "3.0 must be inside Pareto(b=2.0) support"


def test_pareto_support_tracks_b():
    s = pareto.support(b=0.5, alpha=3.0)
    assert isinstance(s, Interval)
    assert float(s.lower) == 0.5
    assert not _contains(s, 0.25)
    assert _contains(s, 0.5)


def test_chi2_support_is_bounded_below_by_loc():
    s = chi2.support(df=2.0, loc=1.5)
    assert isinstance(s, Interval), f"expected Interval, got {type(s).__name__}"
    assert float(s.lower) == 1.5
    assert float(s.upper) == float("inf")
    assert not _contains(s, 0.5), "0.5 < loc must be outside chi2 support"
    assert _contains(s, 1.5), "loc must be inside the support"
    assert _contains(s, 5.0)


def test_truncnorm_support_respects_a_b():
    s = truncnorm.support(a=-1.0, b=1.0)
    assert isinstance(s, Interval), f"expected Interval, got {type(s).__name__}"
    assert float(s.lower) == -1.0
    assert float(s.upper) == 1.0
    assert not _contains(s, -2.0)
    assert not _contains(s, 2.0)
    assert _contains(s, 0.0)


def test_truncnorm_default_support_is_real_line():
    s = truncnorm.support()
    assert isinstance(s, Interval)
    assert float(s.lower) == float("-inf")
    assert float(s.upper) == float("inf")


def test_whole_real_line_support_default():
    # Distributions whose support is genuinely the whole real line inherit
    # the default from rv_continuous instead of overriding it.
    for dist in (norm,):
        s = dist.support()
        assert isinstance(s, Real), f"{dist.name}: expected Real, got {type(s)}"
        assert _contains(s, -1e6) and _contains(s, 1e6)
