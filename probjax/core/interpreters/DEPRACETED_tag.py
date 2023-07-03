
from jax._src import effects
from jax import core
from jax import tree_util

from jax.interpreters import ad, batching, mlir
from typing import Hashable




tag_p = core.Primitive('tag')
tag_p.multiple_results = True

class TagEffect(effects.Effect):
  __repr__ = lambda _: 'tag'


sow_effect = TagEffect()

effects.remat_allowed_effects.add_type(TagEffect)
effects.control_flow_allowed_effects.add_type(TagEffect)
effects.lowerable_effects.add_type(TagEffect)


@tag_p.def_impl
def _tag_impl(*args, **_):
  return args


@tag_p.def_effectful_abstract_eval
def _tag_abstract_eval(*avals, **_):
  return avals, {sow_effect}


def _sow_jvp(primals, tangents, **kwargs):
  out_primals = tag_p.bind(*primals, **kwargs)
  return out_primals, tangents


ad.primitive_jvps[tag_p] = _sow_jvp


def _sow_transpose(cts_in, *args, **kwargs):
  del args, kwargs
  return cts_in


ad.primitive_transposes[tag_p] = _sow_transpose


def _sow_batch_rule(batched_args, batch_dims, **params):
  outs = tag_p.bind(*batched_args, **params)
  return outs, batch_dims


batching.primitive_batchers[tag_p] = _sow_batch_rule
mlir.register_lowering(tag_p, lambda c, *args, **kw: args)


def tag(value, *, tag: Hashable, name: str):
  """Marks a value with a name and a tag.

  Args:
    value: A JAX value to be tagged and named.
    tag: a string representing the tag of the sown value.
    name: a string representing the name to sow the value with.
    mode: The mode by which to sow the value. There are three options: 1.
      `'strict'` - if another value is sown with the same name and tag in the
      same context, harvest will throw an error. 2. `'clobber'` - if another is
      value is sown with the same name and tag, it will replace this value 3.
      `'append'` - sown values of the same name and tag are appended to a
      growing list. Append mode assumes some ordering on the values being sown
      defined by data-dependence.
    key: an optional JAX value that will be tied into the sown value.

  Returns:
    The original `value` that was passed in.
  """
  value = tree_util.tree_map(core.raise_as_much_as_possible, value)
  flat_args, in_tree = tree_util.tree_flatten(value)
  out_flat = tag_p.bind(*flat_args, name=name, tag=tag,tree=in_tree)
  return tree_util.tree_unflatten(in_tree, out_flat)
