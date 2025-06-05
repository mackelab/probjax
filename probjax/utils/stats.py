import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, logsumexp

from probjax.utils.special import digammainv

# MLE for dirichlet distribution


def mle_dirichlet(xs, alpha0=None, maxiter=100):
    if alpha0 is None:
        alpha0 = jnp.ones(xs.shape[1])

    # Ensure that log(xs) is finite
    log_xs = jnp.log(xs)
    log_xs_is_finite = jnp.isfinite(log_xs)
    log_xs = jnp.where(log_xs_is_finite, log_xs, 0.0)
    suff_stat = jnp.sum(log_xs, axis=0) / jnp.sum(log_xs_is_finite, axis=0)

    def fixed_point_iteration(alpha, _):
        dialpha = digamma(alpha.sum())
        new_dialpha = dialpha + suff_stat
        new_alpha = digammainv(new_dialpha)
        return new_alpha, None

    return jax.lax.scan(fixed_point_iteration, alpha0, None, length=maxiter)[0]


# Estimate the differential entropy of a continuous random variable. -------------------
# Main function to compute differential entropy using various methods


# Main function to compute differential entropy using various methods
def differential_entropy(values, window_length=None, base=None, axis=0, method="auto"):
    """
    Computes the differential entropy of a dataset using one of several methods.

    Args:
        values (array-like): Input array.
        window_length (int, optional): Window length for entropy estimation.
        base (float, optional): Logarithm base for entropy. Defaults to natural log is
            None.
        axis (int, optional): Axis along which to compute entropy.
        method (str, optional): Entropy estimation method ('vasicek', 'van es', 'correa'
            ,'ebrahimi', or 'auto').

    Returns:
        float: Estimated differential entropy.
    """
    values = jnp.asarray(values)
    values = jnp.moveaxis(values, axis, -1)
    n = values.shape[-1]

    if window_length is None:
        window_length = int(jnp.sqrt(n) + 0.5)

    if not (2 <= 2 * window_length < n):
        raise ValueError(
            f"Window length ({window_length}) must be at least 2 and less than half"
            "the sample size ({n})."
        )

    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or None.")

    # Sort the data along the last axis
    sorted_data = jnp.sort(values, axis=-1)

    method = method.lower()
    if method == "auto":
        method = _select_auto_method(n)

    if method not in methods:
        raise ValueError(
            f"`method` must be one of {set(methods.keys())}, but got '{method}'."
        )

    # Compute entropy using the selected method
    entropy = methods[method](sorted_data, window_length, n)

    if base is not None:
        entropy /= jnp.log(base)

    return entropy


def _select_auto_method(n):
    """
    Selects an entropy estimation method based on sample size.
    """
    if n <= 10:
        return "van es"
    elif n <= 1000:
        return "ebrahimi"
    else:
        return "vasicek"


# Helper function to pad data along the last axis
def _pad_along_last_axis(X, m):
    shape = X.shape[:-1] + (m,)
    X_left = jnp.broadcast_to(X[..., :1], shape)
    X_right = jnp.broadcast_to(X[..., -1:], shape)
    return jnp.concatenate([X_left, X, X_right], axis=-1)


# Vasicek entropy estimation
def _vasicek_entropy(X, m, n):
    X = _pad_along_last_axis(X, m)
    differences = X[..., 2 * m :] - X[..., : -2 * m]
    logs = jnp.log(n / (2 * m) * differences)
    return jnp.mean(logs, axis=-1)


# Van Es entropy estimation
def _van_es_entropy(X, m, n):
    differences = X[..., m:] - X[..., :-m]
    term1 = jnp.mean(jnp.log((n + 1) / m * differences), axis=-1)
    harmonic_sum = logsumexp(-jnp.log(jnp.arange(m, n + 1)))
    return term1 + harmonic_sum + jnp.log(m) - jnp.log(n + 1)


# Corrected Ebrahimi entropy estimation
def _ebrahimi_entropy(X, m, n):
    """
    Ebrahimi entropy estimator based on differences between order statistics.
    """
    differences = X[..., 1:] - X[..., :-1]  # Consecutive differences
    ci = jnp.where(
        jnp.arange(1, n) <= m,
        1 + (jnp.arange(1, n) - 1) / m,
        1 + (n - jnp.arange(1, n)) / m,
    )
    logs = jnp.log(n * differences / (ci * m))
    return jnp.mean(logs, axis=-1)


# Correa entropy estimation
def _correa_entropy(X, m, n):
    i = jnp.arange(1, n + 1, dtype=jnp.int32)
    dj = jnp.arange(-m, m + 1)[:, None]
    j = i + dj
    j0 = j + m - 1
    Xibar = jnp.mean(X[..., j0], axis=-2, keepdims=True)
    difference = X[..., j0] - Xibar
    num = jnp.sum(difference * dj, axis=-2)
    den = n * jnp.sum(difference**2, axis=-2)
    return -jnp.mean(jnp.log(num / den), axis=-1)


# Mapping methods to functions
methods = {
    "vasicek": _vasicek_entropy,
    "van es": _van_es_entropy,
    "correa": _correa_entropy,
    "ebrahimi": _ebrahimi_entropy,
}


def mutual_information(x, y, method="kraskov", **kwargs):
    """
    Compute the mutual information between two continuous random variables.

    Args:
        x (array-like): First variable.
        y (array-like): Second variable.
        method (str, optional): Estimation method ('knn' or 'kraskov'). Defaults to
            'knn'.

    Returns:
        float: Estimated mutual information.
    """
    x, y = jnp.asarray(x), jnp.asarray(y)
    n = x.shape[0]

    if method == "kraskov":
        return mutual_information_knn_jax(x, y)
    else:
        raise ValueError(f"Unknown method '{method}'.")


# Function to compute the mutual information using k-NN
def mutual_information_knn_jax(x, y, k=3):
    # Reshape if needed
    x = jnp.atleast_2d(x).T if x.ndim == 1 else x
    y = jnp.atleast_2d(y).T if y.ndim == 1 else y

    # Combine x and y into joint space
    xy = jnp.concatenate([x, y], axis=1)

    # Compute pairwise distances in the joint space
    d_xy = jax.vmap(lambda row: jnp.linalg.norm(xy - row, axis=1))(xy)
    kth_distance = jnp.sort(d_xy, axis=1)[:, k]

    # Compute marginal distances
    d_x = jax.vmap(lambda row: jnp.linalg.norm(x - row, axis=1))(x)
    d_y = jax.vmap(lambda row: jnp.linalg.norm(y - row, axis=1))(y)

    # Count neighbors within the k-th distance
    nx = jnp.sum(d_x < kth_distance[:, None], axis=1)
    ny = jnp.sum(d_y < kth_distance[:, None], axis=1)

    # Kraskov's MI estimator
    n = x.shape[0]
    mi = digamma(k) + digamma(n) - (1 / n) * jnp.sum(digamma(nx) + digamma(ny))
    return mi
