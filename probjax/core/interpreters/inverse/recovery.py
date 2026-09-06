"""Lazy, structural affine recovery for the inverse propagation engine.

Analysis is cached per forward trace. Only the recovery callback and its
environment are per invocation; cached evaluators never retain dynamic values.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.extend.core import Jaxpr, Literal, Var

from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p
from probjax.core.interpreters.inverse.affine import (
    AffineSystem,
    _flat_size,
    affine_equation,
)
from probjax.core.interpreters.inverse.logabsdet_rules import (
    inverse_and_logabsdet_state_reducer,
)
from probjax.core.interpreters.inverse.registry import inverse_cost_fn
from probjax.core.jaxpr_propagation.engine import StallRecoveryResult, propagate
from probjax.core.jaxpr_propagation.utils import KnownessLevel


def _parameter_key(value):
    if isinstance(value, dict):
        return tuple((k, _parameter_key(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_parameter_key(v) for v in value)
    hash(value)  # Unhashable parameters are deliberately excluded from CSE.
    return value


def _atom_key(var):
    if isinstance(var, Literal):
        return (var.aval, str(var.val))
    return var


def normalize_inverse_graph(closed):
    """Inline pure jit calls, share pure expressions, and remove dead code.

    Custom inverses and control flow remain opaque. Fresh variables keep two
    calls to the same cached jit body distinct until proven identical by CSE.
    """
    original = closed.jaxpr
    constvars, consts = list(original.constvars), list(closed.consts)
    equations, expressions = [], {}
    changed = False

    def visit(jaxpr, substitutions):
        nonlocal changed

        def resolve(var):
            return var if isinstance(var, Literal) else substitutions.get(var, var)

        for eqn in jaxpr.eqns:
            inputs = [resolve(v) for v in eqn.invars]
            body = eqn.params.get("jaxpr")
            if (
                eqn.primitive.name in {"jit", "pjit"}
                and body is not None
                and not eqn.effects
                and not body.jaxpr.effects
            ):
                changed = True
                inner = dict(zip(body.jaxpr.invars, inputs, strict=True))
                for var, value in zip(body.jaxpr.constvars, body.consts, strict=True):
                    fresh = Var(var.aval)
                    constvars.append(fresh)
                    consts.append(value)
                    inner[var] = fresh
                for child in body.jaxpr.eqns:
                    for var in child.outvars:
                        inner[var] = Var(var.aval)
                outputs = visit(body.jaxpr, inner)
                substitutions.update(zip(eqn.outvars, outputs, strict=True))
                continue

            outputs = [resolve(v) for v in eqn.outvars]
            key = None
            # A custom inverse's registered contract must survive normalization.
            if not eqn.effects and eqn.primitive is not custom_inverse_call_p:
                try:
                    key = (
                        eqn.primitive,
                        tuple(_atom_key(v) for v in inputs),
                        _parameter_key(eqn.params),
                        tuple(v.aval for v in outputs),
                    )
                    hash(key)
                except TypeError:
                    key = None
            if key is not None and key in expressions:
                changed = True
                substitutions.update(zip(eqn.outvars, expressions[key], strict=True))
                continue
            if key is not None:
                expressions[key] = outputs
            if inputs != eqn.invars or outputs != eqn.outvars:
                changed = True
                eqn = eqn.replace(invars=inputs, outvars=outputs)
            equations.append(eqn)
        return [resolve(v) for v in jaxpr.outvars]

    outputs = visit(original, {})
    live = {v for v in outputs if isinstance(v, Var)}
    retained = []
    for eqn in reversed(equations):
        if eqn.effects or any(v in live for v in eqn.outvars):
            retained.append(eqn)
            live.update(v for v in eqn.invars if isinstance(v, Var))
        else:
            changed = True
    if not changed:
        return original, tuple(consts)
    return original.replace(
        constvars=constvars, outvars=outputs, eqns=list(reversed(retained))
    ), tuple(consts)


@dataclass(frozen=True)
class AffineSection:
    source: Var
    sink: Var
    equation_ids: tuple
    system: AffineSystem


class AffineRecoveryPlan:
    """Indexed disjoint sections with a single unknown array at each boundary."""

    def __init__(self, closed, targets):
        self.jaxpr, self.consts = normalize_inverse_graph(closed)
        self.sections = []
        self.by_trigger = {}
        if self.jaxpr.effects:
            return
        roots = {v: v for v in targets}
        groups, consumers = {}, {}
        for index, eqn in enumerate(self.jaxpr.eqns):
            for var in eqn.invars:
                if isinstance(var, Var):
                    consumers.setdefault(var, set()).add(index)
            dependent = [isinstance(v, Var) and v in roots for v in eqn.invars]
            if not any(dependent):
                continue
            sources = {
                roots[v] for v, dep in zip(eqn.invars, dependent, strict=True) if dep
            }
            if len(sources) == 1 and affine_equation(eqn, dependent):
                root = next(iter(sources))
                groups.setdefault(root, []).append(index)
                roots.update((v, root) for v in eqn.outvars)
            else:
                roots.update((v, v) for v in eqn.outvars)

        outputs = {v for v in self.jaxpr.outvars if isinstance(v, Var)}
        for source, indices in groups.items():
            index_set = set(indices)
            equations = [self.jaxpr.eqns[i] for i in indices]
            members = {v for eqn in equations for v in eqn.outvars}
            sinks = {
                v
                for v in members
                if v in outputs or consumers.get(v, set()) - index_set
            }
            if len(sinks) != 1 or consumers.get(source, set()) - index_set:
                continue
            sink = next(iter(sinks))
            if (
                _flat_size(source) != _flat_size(sink)
                or not jnp.issubdtype(source.aval.dtype, jnp.floating)
                or not jnp.issubdtype(sink.aval.dtype, jnp.floating)
            ):
                continue
            side_inputs = list(
                dict.fromkeys(
                    v
                    for eqn in equations
                    for v in eqn.invars
                    if isinstance(v, Var) and v != source and v not in members
                )
            )
            debug_info = self.jaxpr.debug_info._replace(
                traced_for="affine inverse",
                arg_names=tuple(f"arg{i}" for i in range(1 + len(side_inputs))),
                result_paths=("result",),
            )
            section_jaxpr = Jaxpr(
                [], [source, *side_inputs], [sink], equations, debug_info=debug_info
            )
            section = AffineSection(
                source,
                sink,
                tuple((i,) for i in indices),
                AffineSystem(section_jaxpr, [source]),
            )
            section_index = len(self.sections)
            self.sections.append(section)
            for var in [sink, *side_inputs]:
                self.by_trigger.setdefault(var, []).append(section_index)

    def recover(self, context, targets, changed, *, with_logdet):
        del targets
        env = context.env
        candidates = {i for v in changed for i in self.by_trigger.get(v, ())}
        variables, values, consumed, updates = [], [], [], {}
        state = context.read_run_state(namespace=context.state_namespace) or {}
        for index in sorted(candidates, reverse=True):
            section = self.sections[index]
            if env.get_knowness_level(section.source) == KnownessLevel.COMPLETE:
                continue
            if any(
                env.get_knowness_level(v) != KnownessLevel.COMPLETE
                for v in (section.sink, *section.system.known_vars)
            ):
                continue
            recovered, logdet = section.system.solve(
                (), env, [env.read(section.sink)], compute_logdet=with_logdet
            )
            variables.append(section.source)
            values.extend(recovered)
            consumed.extend(section.equation_ids)
            if with_logdet:
                updates[section.source] = state.get(section.sink, 0.0) + logdet
        if not variables:
            return None
        return StallRecoveryResult(variables, values, consumed, updates)


def make_affine_recovery(
    closed, known_inputs, inputs, targets, processing_rule, cache, *, with_logdet=False
):
    """Build a cheap per-call callback; perform/cache analysis only on a stall."""

    def recover(context, requested, changed):
        if closed.jaxpr.effects:
            return None
        key = (closed.jaxpr, tuple(targets))
        if key not in cache:
            cache[key] = AffineRecoveryPlan(closed, targets)
        plan = cache[key]
        if plan.jaxpr is closed.jaxpr:
            return plan.recover(context, requested, changed, with_logdet=with_logdet)

        # Normalization may replace intermediate variables. Restart against the
        # original boundary seeds, then commit only newly recovered targets to
        # the outer environment. Never mix state from the two graph versions.
        def recover_normalized(inner_context, inner_targets, inner_changed):
            return plan.recover(
                inner_context, inner_targets, inner_changed, with_logdet=with_logdet
            )

        options = {}
        if with_logdet:
            options.update(
                reducer=inverse_and_logabsdet_state_reducer,
                initial_state={},
                state_namespace=context.state_namespace,
            )
        result, state, env = propagate(
            plan.jaxpr,
            plan.consts,
            list(known_inputs) + list(plan.jaxpr.outvars),
            inputs,
            targets,
            process_eqn=processing_rule,
            cost_fn=inverse_cost_fn,
            process_all_eqns=True,
            return_state=True,
            return_env=True,
            stall_recovery=recover_normalized,
            **options,
        )
        variables, values, updates = [], [], {}
        for var, value in zip(targets, result, strict=True):
            if (
                env.get_knowness_level(var) == KnownessLevel.COMPLETE
                and context.env.get_knowness_level(var) != KnownessLevel.COMPLETE
            ):
                variables.append(var)
                values.append(value)
                if with_logdet:
                    updates[var] = state.get(var, 0.0)
        if not variables:
            return None
        return StallRecoveryResult(
            variables,
            values,
            tuple(e.eqn_id for e in context.extended_jaxpr.equations),
            updates,
        )

    return recover
