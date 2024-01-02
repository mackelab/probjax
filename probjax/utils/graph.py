
import jax 
import jax.numpy as jnp

import networkx as nx
from functools import partial


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
def faithfull_mask(base_mask, condition_mask, conditioned_nodes="unchanged"):
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
            # The parents of all children of i will now depend on each other
            children_of_i = base_mask[:,i]
            parents_of_children_of_i = base_mask & (children_of_i[:,None] & ~children_of_i[None,:])
            parents_of_children_of_i = jnp.any(parents_of_children_of_i, axis=0)

            return base_mask, condition_mask
        
        def uncondition_case(base_mask, condition_mask):
            return base_mask, condition_mask
        
        base_mask, condition_mask = jax.lax.cond(condition_mask[i], condition_case, uncondition_case, base_mask, condition_mask)
        

        return (base_mask, condition_mask), None

    (base_mask, condition_mask), _ = jax.lax.scan(body_fn, (base_mask, condition_mask), jnp.arange(num_nodes))
        
    
    return base_mask



@jax.jit
def min_faithfull_mask(mask, condition_mask, top_mode=0, conditioned_nodes="unchanged"):
    """ Minimally faithfull mask update for conditioning"""
    num_nodes = mask.shape[0]
    I = moralize(mask)
    H = jnp.zeros_like(mask, dtype=jnp.bool_)
    # 0 is child, 1 is parent
    UPSTREAM = top_mode
    DOWNSTREAM = 1 - top_mode
    num_parents_or_childs = jnp.sum(mask & (~condition_mask[None, :] & ~condition_mask[:, None]), axis=UPSTREAM)
    #print(num_parents_or_childs)
    S = (num_parents_or_childs == 1) & (~condition_mask) # Frontier set
    M = jnp.zeros((num_nodes), dtype=jnp.bool_) # Marked nodes
    
    def cond_fn(val):
        S, _, _, _ = val
        return jnp.any(S) 
    
    def body_fn(val):
        S, M, I, H = val
        
        #print(S)
        # Find the node with the fewest edges added
        v = min_fill_heuristic(mask, I, S, top_mode)
        # Add edge in I between unmarked neighbours in I 
        neighbours_v = I[v,:] & (~M)
        I = I | (neighbours_v[:,None] & neighbours_v[None,:])
        # Make unmarked neighbours of v, the parents of v in H
        H = H.at[v,:].set(neighbours_v)
        # Remove v from S and mark it
        S = S.at [v].set(False)
        M = M.at[v].set(True)
        
        if top_mode == 1:
            u = mask[:,v] & (~M) # Not marked children
            upstream_u = mask & (u[:, None]  & ~u[None,:]) # Parents of not marked children
            all_upstream_u_marked = ~jnp.any(upstream_u & ~M, axis=1)
        else:
            u = mask[v,:] & (~M) # Not marked parents
            upstream_u = mask & (u[None,:]  & ~u[:, None]) # Children of not marked parents
            all_upstream_u_marked = ~jnp.any(upstream_u & ~M, axis=1)
        
        

        S = S | (u & all_upstream_u_marked)
        S = S & (~condition_mask)

        
        return S, M, I, H
    
    _,_,_, H = jax.lax.while_loop(cond_fn, body_fn, (S, M, I, H))
    H = H | jnp.eye(num_nodes, dtype=jnp.bool_)
    
    # Conditioned nodes will keep the unconditional edges, hence each row of H where condition_mask is true should be equal to "mask"
    if conditioned_nodes == "unchanged":
        H = H & ~condition_mask[:, None] | mask & condition_mask[:, None]
    elif conditioned_nodes == "removed":
        H = H & ~condition_mask[:, None]
    elif conditioned_nodes == "added":
        H = H | condition_mask[:, None]
    
    return H
    
                
        
    
    
@partial(jax.jit, static_argnums=(3,))
def min_fill_heuristic(G, I, S, top_mode=0):
    """ Min-fill heuristic for finding a node to eliminate"""
    
    # 0 is child, 1 is parent
    UPSTREAM = top_mode
    DOWNSTREAM = 1 - top_mode
    
    # Find the number of edges that would be added if we eliminated each node
    num_edges_added = I.sum(axis=DOWNSTREAM)
    num_edges_added = S * num_edges_added + (~S) * (I.shape[0] + 1)
    # Find the node that would add the fewest edges
    # Additional constraint: Given the above, find the node with the fewest parents
    min_val = jnp.min(num_edges_added)
    num_parents = G.sum(axis=UPSTREAM) * (num_edges_added == min_val) + (I.shape[0] + 1) * (num_edges_added != min_val)
    #print(num_edges_added + num_parents)
    node_to_eliminate = jnp.argmin(num_edges_added + num_parents)

    return node_to_eliminate 



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
    