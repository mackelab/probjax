from probjax.distributions.utils import Match
from probjax.distributions import Distribution
from probjax import distributions as dist

import jax
import warnings

__all__ = ["register_kl", "kl_divergence"]

_KL_REGISTRY = {}
_KL_MEMOIZE = {}


def register_kl(type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised. For example to
    resolve the ambiguous situation::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(p, q): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(p, q): ...

    you should register a third most-specific implementation, e.g.::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
        type_q (type): A subclass of :class:`~torch.distributions.Distribution`.
    """
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError(
            "Expected type_p to be a Distribution subclass but got {}".format(type_p)
        )
    if not isinstance(type_q, type) and issubclass(type_q, Distribution):
        raise TypeError(
            "Expected type_q to be a Distribution subclass but got {}".format(type_q)
        )

    def decorator(fun):
        _KL_REGISTRY[type_p, type_q] = fun
        _KL_MEMOIZE.clear()  # reset since lookup order may have changed
        return fun

    return decorator


def _dispatch_kl(type_p, type_q):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [
        (super_p, super_q)
        for super_p, super_q in _KL_REGISTRY
        if issubclass(type_p, super_p) and issubclass(type_q, super_q)
    ]
    if not matches:
        return NotImplemented
    # Check that the left- and right- lexicographic orders agree.
    # mypy isn't smart enough to know that _Match implements __lt__
    # see: https://github.com/python/typing/issues/760#issuecomment-710670503
    left_p, left_q = min(Match(*m) for m in matches).types  # type: ignore[type-var]
    right_q, right_p = min(Match(*reversed(m)) for m in matches).types  # type: ignore[type-var]
    left_fun = _KL_REGISTRY[left_p, left_q]
    right_fun = _KL_REGISTRY[right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn(
            "Ambiguous kl_divergence({}, {}). Please register_kl({}, {})".format(
                type_p.__name__, type_q.__name__, left_p.__name__, right_q.__name__
            ),
            RuntimeWarning,
        )
    return left_fun


def kl_divergence(
    p: Distribution, q: Distribution, mc_samples=0, key=None
) -> jax.Array:
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

    .. math::

        KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

    Args:
        p (Distribution): A :class:`~torch.distributions.Distribution` object.
        q (Distribution): A :class:`~torch.distributions.Distribution` object.
        mc_samples (int): Number of samples to use for Monte Carlo approximation of KL divergence. Defaults to 0. Then only analytic expressions.
        key (jax.random.PRNGKey): Key for random number generation. Defaults to None. Only required if mc_samples > 0.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    try:
        fun = _KL_MEMOIZE[type(p), type(q)]
    except KeyError:
        fun = _dispatch_kl(type(p), type(q))
        _KL_MEMOIZE[type(p), type(q)] = fun
    if fun is NotImplemented:
        raise NotImplementedError(
            "No KL(p || q) is implemented for p type {} and q type {}".format(
                p.__class__.__name__, q.__class__.__name__
            )
        )
    return fun(p, q)


@register_kl(Distribution, Distribution)
def _kl_generic(p, q, mc_samples=0, key=None):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL divergence between distributions with different event shapes not supported"
        )

    assert (
        mc_samples >= 0
    ), "For general distirbutions we require mc_samples >= 0, to evaluate a Monte Carlo approximation of the KL divergence."
    assert key is not None, "Key must be provided if mc_samples > 0"

    if p.has_rsample:
        samples = p.rsample(key, (mc_samples,))
    else:
        samples = p.sample(key, (mc_samples,))
    log_prob_p = p.log_prob(samples)
    log_prob_q = q.log_prob(samples)
    return (log_prob_p - log_prob_q).mean(0)


@register_kl(dist.Bernoulli, dist.Bernoulli)
def _kl_bernoulli_bernoulli(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    probs_q = q.probs
    t1 = probs_p * (probs_p / probs_q).log()
    t2 = (1 - probs_p) * ((1 - probs_p) / (1 - probs_q)).log()
    return t1 + t2


@register_kl(dist.Normal, dist.Normal)
def _kl_normal_normal(p, q, mc_samples=0, key=None):
    loc_p, scale_p = p.loc, p.scale
    loc_q, scale_q = q.loc, q.scale
    t1 = (scale_p / scale_q).log()
    t2 = ((scale_p / scale_q) ** 2 + ((loc_p - loc_q) / scale_q) ** 2 - 1) / 2
    return t1 + t2
