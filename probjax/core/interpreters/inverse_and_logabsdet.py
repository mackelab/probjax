import functools

import jax
import jax.numpy as jnp
from jax._src.util import safe_map
from jax.extend.core import Literal, Primitive

try:
    from jax.experimental.pjit import pjit_p
except ImportError:
    # JaX 0.7
    from jax._src.pjit import jit_p as pjit_p

from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p
from probjax.core.interpreters.inverse import (
    _BIVARIATE_INVERSE_REGISTRY,
    _CUSTOM_INVERSE_PROCESSING_RULES,
    _UNIVARIATE_INVERSE_REGISTRY,
    InverseProcessingRule,
    is_bivariate,
    is_univariate,
)

_CUSTOM_INVErSE_AND_LOG_DET_RULES = {}


def register_inverse_and_log_det_rule(key):
    def decorator(func):
        nonlocal key
        _CUSTOM_INVErSE_AND_LOG_DET_RULES[key] = func
        return func

    return decorator


def value_and_log_det_diagonal(f):
    # This assumes that the jacobian is diagonal!
    grad_fn = jax.value_and_grad(f)

    def log_det_fn(*args, **kwargs):
        # Handle scalar inputs by wrapping them in arrays
        args_arrays = [jnp.array(arg) if jnp.ndim(arg) == 0 else arg for arg in args]
        args_arrays = jnp.broadcast_arrays(*args_arrays)
        n_dim = args_arrays[0].ndim
        vmaped_grad_fn = grad_fn
        for _ in range(n_dim):
            vmaped_grad_fn = jax.vmap(vmaped_grad_fn)
        value, det = vmaped_grad_fn(*args_arrays, **kwargs)

        log_det = jnp.log(jnp.abs(det) + 1e-10)
        while log_det.ndim > 0:
            log_det = jnp.sum(log_det, axis=-1)
        return value, log_det

    return log_det_fn


def value_and_jacfwd(f, x):
    pushfwd = functools.partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac


def value_and_jacrev(f, x):
    y, pullback = jax.vjp(f, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)
    return y, jac


def log_det_multivariate(f):
    # This is expensive!
    def log_det_fn(*args, **kwargs):
        args = [jnp.atleast_1d(arg) for arg in args]
        value, jac = value_and_jacfwd(f, *args)
        sign, log_det = jnp.linalg.slogdet(jac)
        return value, log_det

    return log_det_fn


class InverseAndLogAbsDetProcessingRule(InverseProcessingRule):
    log_dets = {}

    def __call__(self, eqn, known_invars, known_outvars):
        # print(self.log_dets)
        is_known_invars = safe_map(lambda x: x is not None, known_invars)
        is_known_outvars = safe_map(lambda x: x is not None, known_outvars)

        # print(eqn.primitive, is_known_invars, is_known_outvars)

        if eqn.primitive is custom_inverse_call_p and all(is_known_outvars):
            return self._default_custom_inverse_call_apply(
                eqn, known_invars, known_outvars
            )
        elif (
            all(is_known_outvars) and eqn.primitive in _CUSTOM_INVERSE_PROCESSING_RULES
        ):
            return self._default_custom_rule_apply(eqn, known_invars, known_outvars)
        elif (
            not all(is_known_invars) and eqn.primitive is pjit_p
        ):  # or eqn.primitive is custom_jvp_call_p:
            return self._default_pjit(eqn, known_invars, known_outvars)

        elif is_univariate(eqn) and all(is_known_outvars):
            return self._default_univariate_inverse(eqn, known_invars, known_outvars)
        elif is_bivariate(eqn) and all(is_known_outvars) and any(is_known_invars):
            return self._default_bivariate_inverse(eqn, known_invars, known_outvars)
        elif all(is_known_invars):
            return self._default_forward_processing(eqn, known_invars, known_outvars)
        else:
            raise NotImplementedError(f"Cannot invert {eqn}")

    def _default_univariate_inverse(self, eqn, known_invars, known_outvars):
        primitive = eqn.primitive
        if primitive not in _UNIVARIATE_INVERSE_REGISTRY:
            raise NotImplementedError(f"{primitive} is not invertible!")

        inv_primitive = _UNIVARIATE_INVERSE_REGISTRY[primitive]
        if isinstance(inv_primitive, Primitive):

            def f(*args):
                subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
                invars = inv_primitive.bind(*subfuns, *args, **bind_params)

                return jnp.sum(invars)

            eval_fn = value_and_log_det_diagonal(f)
            invars, log_abs_det = eval_fn(*known_outvars)
        else:
            eval_fn = value_and_log_det_diagonal(
                lambda *args: jnp.sum(inv_primitive(*args, **eqn.params))
            )
            invars, log_abs_det = eval_fn(*known_outvars)

        if not isinstance(invars, list):
            invars = [
                invars,
            ]

        log_det_previous = self.log_dets.get(eqn.outvars[0], 0.0)
        self.log_dets[eqn.invars[0]] = log_det_previous + log_abs_det

        return eqn.invars, invars

    def _default_bivariate_inverse(self, eqn, known_invars, known_outvars):
        primitive = eqn.primitive
        input1 = known_outvars[0]
        left_inverse = known_invars[0] is None
        # Left or right inverses
        input2 = known_invars[1] if left_inverse else known_invars[0]

        (left_inverse_fn, right_inverse_fn) = _BIVARIATE_INVERSE_REGISTRY[primitive]

        inv_primitive = left_inverse_fn if left_inverse else right_inverse_fn

        if isinstance(inv_primitive, Primitive):

            def f(*args):
                subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
                invars = inv_primitive.bind(*subfuns, *args, **bind_params)

                return invars

            eval_fn = value_and_log_det_diagonal(f)
            invars, log_abs_det = eval_fn(input1, input2)
        else:
            eval_fn = value_and_log_det_diagonal(
                lambda *args: jnp.sum(inv_primitive(*args, **eqn.params))
            )
            invars, log_abs_det = eval_fn(input1, input2)

        log_det_previous = self.log_dets.get(eqn.outvars[0], 0.0)
        if left_inverse:
            self.log_dets[eqn.invars[0]] = log_det_previous + log_abs_det
            return [eqn.invars[0]], [invars]
        else:
            self.log_dets[eqn.invars[1]] = log_det_previous + log_abs_det
            return [eqn.invars[1]], [invars]

    def _default_pjit(self, eqn, known_invars, outvars):
        if "jaxpr" in eqn.params:
            jaxpr = eqn.params["jaxpr"]
        else:
            jaxpr = eqn.params["call_jaxpr"]

        sub_invars = jaxpr.jaxpr.invars
        sub_outvars = jaxpr.jaxpr.outvars

        subvars = sub_invars + sub_outvars
        vars = eqn.invars + eqn.outvars

        # print(subvars)
        # print(vars)
        for v_sub, v in zip(subvars, vars, strict=False):
            if v_sub in self.log_dets and not isinstance(v, Literal):
                self.log_dets[v] = self.log_dets[v_sub]

        log_det_previous = sum([
            self.log_dets.get(v, 0.0) for v in eqn.outvars if not isinstance(v, Literal)
        ])

        for v in eqn.invars:
            if not isinstance(v, Literal):
                self.log_dets[v] = log_det_previous

        # Pass logdet to outer scope

    def _default_custom_rule_apply(self, eqn, known_invars, known_outvars):
        primitive = eqn.primitive
        if primitive not in _CUSTOM_INVERSE_PROCESSING_RULES:
            raise NotImplementedError(f"{primitive} is not invertible!")

        outvars, outs = _CUSTOM_INVERSE_PROCESSING_RULES[primitive](
            eqn, known_invars, known_outvars
        )
        # vars = eqn.invars + eqn.outvars
        log_det_previous = sum([self.log_dets.get(v, 0.0) for v in eqn.outvars])
        for v in outvars:
            self.log_dets[v] = log_det_previous

        return outvars, outs

    def _default_custom_inverse_call_apply(self, eqn, known_invars, known_outvars):
        inverse_jaxpr = eqn.params["inverse_jaxpr_thunk"]()
        jaxpr = inverse_jaxpr.jaxpr
        consts = inverse_jaxpr.literals
        inputs = [v if v is not None else known_outvars[0] for v in known_invars]
        out = jax.core.eval_jaxpr(
            jaxpr,
            consts,
            *inputs,
        )
        invars = [
            eqn.invars[i] for i in range(len(eqn.invars)) if known_invars[i] is None
        ]
        inputs = [out[0] for i in range(len(eqn.invars)) if known_invars[i] is None]
        log_abs_det = out[-1]
        log_abs_det = jnp.sum(log_abs_det)
        log_det_previous = sum([self.log_dets.get(v, 0.0) for v in eqn.outvars])
        for v in eqn.invars:
            self.log_dets[v] = log_det_previous + log_abs_det
        out = out[:-1]
        return invars, out
