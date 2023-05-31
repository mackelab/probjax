import jax
from functools import wraps

from jax import core
import jax.random as jrandom
from jax.tree_util import tree_flatten, tree_unflatten
from jax.core import ClosedJaxpr, Jaxpr, Primitive, JaxprEqn
from typing import Iterable
from jax import lax
from jax._src.util import safe_map, curry
from jax._src import util

from typing import Any, Callable, Iterable, Union


from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow import (
    _initial_style_open_jaxpr,
    _initial_style_jaxprs_with_common_consts,
)
from jax._src.lax.control_flow.common import (
    _abstractify,
    _avals_short,
    _check_tree_and_avals,
    _initial_style_jaxprs_with_common_consts,
    _make_closed_jaxpr,
    _prune_zeros,
    _typecheck_param,
    allowed_effects,
)

ABSTRACT_RANDOM_KEY = _abstractify(jrandom.PRNGKey(0))


class BaseRules(dict):
    """A dictionary of rules for primitives."""

    def _default_rule(self, prim: Primitive, *args, **kwargs) -> Any:
        """Default implementation of a rule for a primitive.

        Args:
            prim (Primitive): Primitive

        Returns:
            Any: Rule to apply
        """
        return prim.bind(*args, **kwargs)

    def __getitem__(self, key: Primitive) -> Any:
        """Returns a custom rule for a primitive.
            If no rule is found, returns the default rule.

        Args:
            key (str): Name of primitive

        Returns:
            Any: Rule to apply
        """
        return self.__dict__.get(key, self._default_rule)  # type: ignore

    def __setitem__(self, key: Primitive, value: Any) -> None:
        """Sets a custom rule for a primitive.

        Args:
            key (str): Name of primitive
            value (Any): Rule to apply
        """
        self.__dict__[key] = value  # type: ignore


class BaseInterpreter:
    rules: BaseRules = BaseRules()
    env: dict = {}

    def write(self, var: str, val: Any):
        """Writes a value to the environment."""
        self.env[var] = val

    def read(self, var: Union[str, core.Literal]):
        """Reads a value from the environment."""
        if type(var) is core.Literal:
            return var.val
        else:
            return self.env[var]

    def _init_environment(
        self, jaxpr: Jaxpr, consts: Iterable, *args
    ) -> Iterable[JaxprEqn]:
        """Initializes the environment for the Jaxpr.

        Args:
            write (Callable): Function that writes to the environment
            jaxpr (Jaxpr): Jaxpr
            consts (Iterable): Constants required in the Jaxpr

        Returns:
            Iterable[Equation]: Iterable of equations to evaluate.
        """
        self.env = {}
        # Bind args and consts to environment
        safe_map(self.write, jaxpr.invars, args)
        safe_map(self.write, jaxpr.constvars, consts)

        # Equations in some order
        eqns = jaxpr.eqns

        return eqns

    def _get_output(self, jaxpr: Jaxpr) -> Any:
        """Reads the output of the Jaxpr from the environment."""
        out = safe_map(self.read, jaxpr.outvars)
        return out

    def _get_invals(self, eqn: JaxprEqn) -> Iterable:
        """Returns the input variables of an equation."""
        return safe_map(self.read, eqn.invars)

    def _write_outvals(self, eqn: JaxprEqn, outvals) -> Iterable:
        """Returns the output variables of an equation."""
        return safe_map(self.write, eqn.outvars, outvals)


    def eval_jaxpr(self, jaxpr: Jaxpr, consts: Iterable, *args, **kwargs) -> Any:
        # Mapping from variable -> value
        eqns = self._init_environment(jaxpr, consts, *args, **kwargs)

        # Loop through equations and evaluate primitives using `bind`
        for eqn in eqns:
            # Read inputs to equation from environmen
            invals = self._get_invals(eqn)
            prim = eqn.primitive
            subfuns, bind_params = prim.get_bind_params(eqn.params)
            rule = self.rules[prim]
            # `bind` is how a primitive is called
            outvals = rule(prim, *subfuns, *invals, **bind_params)
            # Primitives may return multiple outputs or not
            if not eqn.primitive.multiple_results:
                outvals = [outvals]

            # Write the results of the primitive into the environment
            self._write_outvals(eqn, outvals)
        # Read the final result of the Jaxpr from the environment

        return self._get_output(jaxpr)

def remove_closed_jaxpr_vars_with_suffix(closed_jaxpr, suffix="_"):
    jaxpr = closed_jaxpr.jaxpr 
    new_jaxpr = remove_jaxpr_vars_with_suffix(jaxpr, suffix=suffix)
    return ClosedJaxpr(new_jaxpr, closed_jaxpr.literals)

def remove_jaxpr_vars_with_suffix(jaxpr, suffix="_"):
    return jaxpr.replace(invars=[v for v in jaxpr.invars if v.suffix!=suffix])

def jaxpr_returning_const(*consts, invars=[]):
    consts, const_tree = tree_flatten(consts)
    const_avals = tuple(map(_abstractify, consts))
    const_vars = [jax.core.Var(0, "_obs", c_aval) for c_aval in const_avals]
    new_jaxpr = Jaxpr(const_vars, invars, const_vars, [])
    new_closed_jaxpr = ClosedJaxpr(new_jaxpr, consts)
    return new_closed_jaxpr, const_tree

@util.cache()
def _sampling_logprobs_jaxprs_with_common_consts(sampling_fn, log_prob_fn):
    operands = (
            ABSTRACT_RANDOM_KEY,
    )
    sampling_ops_avals, sampling_ops_tree = tree_flatten(operands)
    sampling_jaxpr, sampling_consts, sampling_out_trees = _initial_style_open_jaxpr(
        sampling_fn, sampling_ops_tree, tuple(sampling_ops_avals)
    )
    sampling_fn_closed_jaxpr = ClosedJaxpr(pe.convert_constvars_jaxpr(sampling_jaxpr), ())
    sampling_out_avals = sampling_fn_closed_jaxpr.out_avals

    log_prob_operands = sampling_out_avals
    log_prob_ops, log_prob_ops_tree = tree_flatten(log_prob_operands)
    log_prob_ops_avals = tuple(log_prob_ops)  # Is already abstract
    log_prob_jaxpr, log_prob_consts, log_prob_out_trees = _initial_style_open_jaxpr(
        log_prob_fn, log_prob_ops_tree, log_prob_ops_avals
    )

    jaxprs = [sampling_jaxpr, log_prob_jaxpr]
    consts = [sampling_consts ,log_prob_consts]
    out_trees = [sampling_out_trees, log_prob_out_trees]

    newvar = core.gensym(jaxprs, suffix='_')
    all_const_avals = [map(_abstractify, consts) for consts in consts]
    unused_const_vars = [map(newvar, const_avals)
                       for const_avals in all_const_avals]
    def pad_jaxpr_constvars(i, jaxpr):
        prefix = util.concatenate(unused_const_vars[:i])
        suffix = util.concatenate(unused_const_vars[i + 1:])
        constvars = [*prefix, *jaxpr.constvars, *suffix]
        return jaxpr.replace(constvars=constvars)
    
    consts = util.concatenate(consts)
    jaxprs = tuple(pad_jaxpr_constvars(i, jaxpr) for i, jaxpr in enumerate(jaxprs))
    closed_jaxprs = [core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
                    for jaxpr in jaxprs]
    
    return closed_jaxprs, consts, out_trees

