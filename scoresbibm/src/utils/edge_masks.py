
import jax
from probjax.utils.graph import faithfull_mask, min_faithfull_mask, moralize


def get_edge_mask_fn(name, task):
    base_mask_fn = task.get_base_mask_fn()
    if name.lower() == "faithfull":

        def faithfull_edge_mask(node_id, condition_mask, meta_data=None):
            base_mask = base_mask_fn(node_id, meta_data)
            return jax.vmap(faithfull_mask, in_axes=(None, 0))(
                base_mask, condition_mask
            )

        return faithfull_edge_mask
    elif name.lower() == "min_faithfull":
        def min_faithfull_edge_mask(node_id, condition_mask,meta_data=None):
            base_mask = base_mask_fn(node_id, meta_data)

            return jax.vmap(min_faithfull_mask, in_axes=(None, 0))(
                base_mask, condition_mask
            )

        return min_faithfull_edge_mask
    elif name.lower() == "undirected":
        
        def undirected_edge_mask(node_id, condition_mask, meta_data=None):
            base_mask = base_mask_fn(node_id, meta_data)
            return moralize(base_mask)
        
        return undirected_edge_mask
    elif name.lower() == "none":
        return lambda node_id, condition_mask, *args: None
    else:
        raise NotImplementedError()