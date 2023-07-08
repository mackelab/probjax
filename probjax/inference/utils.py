

# def flatten_potential_fn(potential_fn: Callable, in_vals: PyTree):
#     """Process the potential function to return a function that takes in a single argument."""
#     leaves, in_tree = jax.tree_util.tree_flatten(in_vals)

#     # Casting to tuple to make them hashable -> static_argnums
#     shapes = tuple(jax.tree_map(lambda x: jnp.shape(x), leaves))
#     lengths = jax.tree_map(lambda x: jnp.size(x), leaves)
#     cum_lengths = tuple(accumulate(lengths))[:-1]

#     def _flatten(x):
#         leaves, _ = jax.tree_util.tree_flatten(x)
#         flatten_leaves = jax.tree_map(lambda x: jnp.ravel(x), leaves)
#         return jnp.concatenate(flatten_leaves)

#     @partial(jax.jit, static_argnums=(1, 2))
#     def _unflatten(x, cum_lengths, shapes):
#         flattened_leaves = jnp.split(x, cum_lengths)
#         leaves = jax.tree_map(
#             lambda x, s: jnp.reshape(x, s), tuple(flattened_leaves), shapes
#         )
#         return jax.tree_util.tree_unflatten(in_tree, leaves)

#     def _potential_fn(x):
#         x = _unflatten(x, cum_lengths, shapes)
#         print(x)
#         return potential_fn(*x)

#     return _flatten, _unflatten, _potential_fn


# def sliced_potential_fn(flatten_potential_fn, loc, direction):
#     """Returns a function that slices the potential function in a given direction."""

#     def _sliced_potential_fn(t):
#         return flatten_potential_fn(loc + t * direction)

#     return _sliced_potential_fn


# def conditional_potential_fn(flatten_potential_fn, x, indices):
#     """Returns a function that slices the potential function in a given direction."""

#     def _conditional_potential_fn(sub_x):
#         return flatten_potential_fn(x.at[indices].set(sub_x))

#     return _conditional_potential_fn