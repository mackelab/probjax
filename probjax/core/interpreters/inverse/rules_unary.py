import jax
import jax.numpy as jnp


def integer_pow_inverse(x, **params):
    y = params.pop("y")
    return jax.lax.pow_p.bind(x, 1 / y, **params)


def logit(x, **params):
    return jax.lax.log_p.bind(x) - jax.lax.log1p_p.bind(-x)  # type: ignore


def sqrt_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.pow_p.bind(x, 2.0, **params)


def rsqrt_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return 1.0 / jax.lax.pow_p.bind(x, 2.0, **params)


def cbrt_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.pow_p.bind(x, 3.0, **params)


def asin_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.asin_p.bind(x, **params)


def acos_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.acos_p.bind(x, **params)


def atan_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.atan_p.bind(x, **params)


def sin_inverse(x, **params):
    params = dict(params)
    params.setdefault("accuracy", None)
    return jax.lax.sin_p.bind(x, **params)


def cos_inverse(x, **params):
    params = dict(params)
    params.setdefault("accuracy", None)
    return jax.lax.cos_p.bind(x, **params)


def tan_inverse(x, **params):
    params = dict(params)
    params.setdefault("accuracy", None)
    return jax.lax.tan_p.bind(x, **params)


UNIVARIATE_INVERSE_REGISTRY = {
    jax.lax.sin_p: asin_inverse,
    jax.lax.asin_p: sin_inverse,
    jax.lax.cos_p: acos_inverse,
    jax.lax.acos_p: cos_inverse,
    jax.lax.tan_p: atan_inverse,
    jax.lax.atan_p: tan_inverse,
    jax.lax.tanh_p: jax.lax.atanh_p,
    jax.lax.atanh_p: jax.lax.tanh_p,
    jax.lax.sinh_p: jax.lax.asinh_p,
    jax.lax.asinh_p: jax.lax.sinh_p,
    jax.lax.cosh_p: jax.lax.acosh_p,
    jax.lax.acosh_p: jax.lax.cosh_p,
    jax.lax.exp_p: jax.lax.log_p,
    jax.lax.exp2_p: lambda x, **params: jnp.log2(x),
    jax.lax.log_p: jax.lax.exp_p,
    jax.lax.sqrt_p: sqrt_inverse,
    jax.lax.rsqrt_p: rsqrt_inverse,
    jax.lax.cbrt_p: cbrt_inverse,
    jax.lax.neg_p: jax.lax.neg_p,
    jax.lax.copy_p: jax.lax.copy_p,
    jax.lax.log1p_p: jax.lax.expm1_p,
    jax.lax.expm1_p: jax.lax.log1p_p,
    jax.lax.erf_p: jax.lax.erf_inv_p,
    jax.lax.erf_inv_p: jax.lax.erf_p,
    jax.lax.conj_p: jax.lax.conj_p,
    jax.lax.real_p: jax.lax.real_p,
    jax.lax.imag_p: jax.lax.imag_p,
    jax.lax.logistic_p: logit,
    jax.lax.integer_pow_p: integer_pow_inverse,
}
