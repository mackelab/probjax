"""Adaptive Euler-Maruyama SDE integrator.

Selected with ``sdeint(..., solver="adaptive")`` and configured with a
:class:`~probjax.utils.sdeutil.adaptive.SDEStepSizeAdaptor`. The integrator
uses step doubling on an Euler-Maruyama step, so it has no step function of
its own: the adaptive machinery in
:mod:`probjax.utils.sdeutil.integrate_adaptive` drives it directly.
"""

from probjax.utils.sdeutil.base import SDESolverAPI, register_method

info = {
    "order": 1,
    "strong_order": 0.5,
    "weak_order": 1,
    "adaptive": True,
}

_UNSUPPORTED = (
    "The adaptive SDE integrator has no step function; it is selected with "
    "sdeint(..., solver='adaptive') and driven by a step-size adaptor."
)


class adaptive(SDESolverAPI):
    """Adaptive Euler-Maruyama SDE integrator (step doubling)."""

    @staticmethod
    def init(*args, **kwargs):
        raise NotImplementedError(_UNSUPPORTED)

    @staticmethod
    def build_step(*args, **kwargs):
        raise NotImplementedError(_UNSUPPORTED)


register_method("adaptive", adaptive, info=info)
