
import jax 
import jax.numpy as jnp

import networkx as nx


@jax.jit
def find_ancestors_jax(mask, node):
    """Find ancestors of a node in a graph.

    Args:
        mask (Array): Adjacency matrix of a directed graph.
        node (int): Node of interest.

    Returns:
        _type_: _description_
    """
    num_nodes = mask.shape[0]
    is_ancestor = jnp.zeros(num_nodes, dtype=jnp.bool_)
    stack = jnp.empty(num_nodes, dtype=jnp.int32)
    stack = stack.at[0].set(node)
    
    def body_fn(carry, i):
        is_ancestor, stack = carry
        current_node = stack[i]
        current_parents = mask[current_node, :]
        
        def inner_body_fn(carry, j):
            is_ancestor, stack = carry
            value = current_parents[j]
            cond = value & (j != current_node) & (~is_ancestor[j])
            
            def true_fn(is_ancestor, stack):
                is_ancestor = is_ancestor.at[j].set(True)
                stack = stack.at[i+1].set(j)
                return is_ancestor, stack
            def false_fn(is_ancestor, stack):
                return is_ancestor, stack
            
            is_ancestor, stack = jax.lax.cond(cond, true_fn, false_fn, is_ancestor, stack)
            return (is_ancestor, stack), None
        
        (is_ancestor, stack), _ = jax.lax.scan(inner_body_fn, (is_ancestor, stack), jnp.arange(num_nodes))
        return (is_ancestor, stack), None
    
    (is_ancestor, stack), _ = jax.lax.scan(body_fn, (is_ancestor, stack), jnp.arange(num_nodes))
    

    return is_ancestor



@jax.jit
def faithfull_mask(base_mask, condition_mask):
    """ Faithfull mask update for conditioning"""
    
    graph = base_mask.astype(jnp.bool_).copy()
    base_mask = base_mask.astype(jnp.bool_) # Rows are paraents, columns are children
    condition_mask = condition_mask.astype(jnp.bool_)
    num_nodes = base_mask.shape[0]
    
    def body_fn(carry, i):
        base_mask, condition_mask = carry
        
        def condition_case(base_mask, condition_mask):
            # We need to update all ancestors of i
            is_ancestor = find_ancestors_jax(graph, i)
            is_ancestor = is_ancestor & (~condition_mask)
            all_ancestors = jnp.nonzero(is_ancestor, size=num_nodes, fill_value=i)[0]
            # They will now depend on i
            base_mask = base_mask.at[all_ancestors,i].set(True)
            # They will now depend on each other!
            base_mask = base_mask | (is_ancestor[:,None] & is_ancestor[None,:])
            # Zero out row of conditioned index
            base_mask = base_mask.at[i,:].set(False)
            return base_mask, condition_mask
        
        def uncondition_case(base_mask, condition_mask):
            return base_mask, condition_mask
        
        base_mask, condition_mask = jax.lax.cond(condition_mask[i], condition_case, uncondition_case, base_mask, condition_mask)
        

        return (base_mask, condition_mask), None

    (base_mask, condition_mask), _ = jax.lax.scan(body_fn, (base_mask, condition_mask), jnp.arange(num_nodes))
        
    
    return base_mask

def convert_to_networkx(mask):
    """Converts a mask to a networkx graph"""
    return nx.from_numpy_array(mask - jnp.eye(mask.shape[0]), create_using=nx.DiGraph).reverse()

@jax.jit
def moralize(adj_matrix):
    adj_matrix = adj_matrix.astype(jnp.bool_)
    
    # Make the graph undirected
    undirected_graph = adj_matrix | adj_matrix.T
    
    # Add edges between parents
    undirected_graph = undirected_graph | (adj_matrix.T @ adj_matrix)

    return undirected_graph

def moralize_networkx(adj_matrix):
    return nx.to_numpy_array(nx.moral_graph(convert_to_networkx(adj_matrix))) != 0


def minimally_faithfull_mask(mask, condition_mask):
    """ Minimally faithfull mask update for conditioning"""
    I = moralize(mask)
    H = jnp.zeros_like(mask, dtype=jnp.bool_)
    