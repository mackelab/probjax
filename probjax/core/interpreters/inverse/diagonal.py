"""Symbolic elementwise affine coefficients, without Jacobians or matrices."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax.extend.core import Literal, Var

from probjax.core.jaxpr_propagation.utils import rebind_primitive


_ELEMENTWISE = frozenset({
    "add", "add_any", "sub", "neg", "mul", "div", "copy",
    "convert_element_type", "select_n",
})
_ORDER_PRESERVING = frozenset({"reshape", "squeeze", "broadcast_in_dim"})


def _static_scalar(value):
    return isinstance(value, (int, float, np.number)) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    )


def _is(value, scalar):
    # Never inspect a runtime array or tracer to simplify an expression.
    return _static_scalar(value) and value == scalar


def _add(a, b):
    return b if _is(a, 0) else a if _is(b, 0) else a + b


def _sub(a, b):
    return a if _is(b, 0) else -b if _is(a, 0) else a - b


def _mul(a, b):
    if _is(a, 0) or _is(b, 0):
        return 0.0
    return b if _is(a, 1) else a if _is(b, 1) else a * b


def _div(a, b):
    return 0.0 if _is(a, 0) else a if _is(b, 1) else a / b


class DiagonalAffineSystem:
    """A cached coefficient program for ``y = scale * x + offset``.

    Reshapes may change the boundary shape, but must preserve flattened element
    order. All shape-dependent operations are proven before tracing coefficients.
    Scalar/broadcast coefficients stay compact until their layout requires an
    expansion. Known parameters remain explicit inputs to the cached program.
    """

    @classmethod
    def try_build(cls, jaxpr, targets):
        if len(targets) != 1 or len(jaxpr.outvars) != 1 or jaxpr.effects:
            return None
        source, sink = targets[0], jaxpr.outvars[0]
        size = math.prod(source.aval.shape)
        if (
            size == 0
            or size != math.prod(sink.aval.shape)
            or not jnp.issubdtype(source.aval.dtype, jnp.floating)
        ):
            return None
        dependent = {source}
        for eqn in jaxpr.eqns:
            if not any(isinstance(v, Var) and v in dependent for v in eqn.invars):
                continue
            if len(eqn.outvars) != 1:
                return None
            name = eqn.primitive.name
            if name not in _ELEMENTWISE | _ORDER_PRESERVING:
                return None
            if math.prod(eqn.outvars[0].aval.shape) != size:
                return None
            if name == "reshape":
                dimensions = eqn.params.get("dimensions")
                if dimensions is not None and tuple(dimensions) != tuple(
                    range(len(eqn.invars[0].aval.shape))
                ):
                    return None
            dependent.update(eqn.outvars)
        if sink not in dependent:
            return None
        return cls(jaxpr, source)

    def __init__(self, jaxpr, source):
        self.shape = source.aval.shape
        self.size = math.prod(self.shape)
        self.known_vars = tuple(v for v in jaxpr.invars if v != source)
        inputs = (*jaxpr.constvars, *self.known_vars)

        def coefficients(*known_values):
            env = {v: (None, value) for v, value in zip(inputs, known_values, strict=True)}
            env[source] = (1.0, 0.0)

            def read(v):
                return (None, v.val) if isinstance(v, Literal) else env[v]

            for eqn in jaxpr.eqns:
                pairs = [read(v) for v in eqn.invars]
                if all(a is None for a, _ in pairs):
                    result = rebind_primitive(
                        eqn.primitive, eqn.params, *[b for _, b in pairs]
                    )
                    values = result if eqn.primitive.multiple_results else [result]
                    env.update((v, (None, b)) for v, b in zip(eqn.outvars, values, strict=True))
                    continue
                name = eqn.primitive.name
                a, b = pairs[0]
                if name in {"add", "add_any", "sub"}:
                    c, d = pairs[1]
                    op = _sub if name == "sub" else _add
                    pair = (op(0.0 if a is None else a, 0.0 if c is None else c), op(b, d))
                elif name == "mul":
                    c, d = pairs[1]
                    pair = (_mul(a, d) if a is not None else _mul(c, b), _mul(b, d))
                elif name == "div":
                    d = pairs[1][1]
                    pair = (_div(a, d), _div(b, d))
                elif name == "neg":
                    pair = (-a, -b)
                elif name == "select_n":
                    which = b
                    slopes, offsets = [], []
                    for v, (c, d) in zip(eqn.invars[1:], pairs[1:], strict=True):
                        slopes.append(jnp.broadcast_to(0.0 if c is None else c, v.aval.shape))
                        offsets.append(jnp.broadcast_to(d, v.aval.shape))
                    pair = (
                        rebind_primitive(eqn.primitive, eqn.params, which, *slopes),
                        rebind_primitive(eqn.primitive, eqn.params, which, *offsets),
                    )
                elif name in _ORDER_PRESERVING:
                    def rearrange(value):
                        if jnp.ndim(value) == 0:
                            return value
                        value = jnp.broadcast_to(value, eqn.invars[0].aval.shape)
                        return rebind_primitive(eqn.primitive, eqn.params, value)

                    pair = (rearrange(a), rearrange(b))
                elif name == "convert_element_type":
                    pair = tuple(rebind_primitive(eqn.primitive, eqn.params, v) for v in (a, b))
                else:  # copy
                    pair = (a, b)
                env[eqn.outvars[0]] = pair
            return env[jaxpr.outvars[0]]

        # Trace once from abstract parameters, never from captured runtime values.
        self.program = jax.make_jaxpr(coefficients)(
            *[jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in inputs]
        )

    def solve(self, consts, known, outputs, *, compute_logdet):
        scale, offset = jax_core.eval_jaxpr(
            self.program.jaxpr, self.program.consts,
            *consts, *[known[v] for v in self.known_vars],
        )
        y = jnp.asarray(outputs[0])
        if _is(scale, 0):
            value = jnp.full(self.shape, jnp.nan, dtype=y.dtype)
        else:
            value = _div(_sub(y, offset), scale)
            if not _static_scalar(scale):
                value = jnp.where(scale == 0, jnp.nan, value)
            value = value.reshape(self.shape)
        logdet = None
        if compute_logdet:
            if _static_scalar(scale):
                total = -self.size * math.log(abs(float(scale))) if scale != 0 else math.inf
                logdet = jnp.asarray(total, dtype=y.dtype)
            else:
                multiplicity = self.size // jnp.size(scale)
                logdet = -multiplicity * jnp.sum(jnp.log(jnp.abs(scale)))
        return [value], logdet
