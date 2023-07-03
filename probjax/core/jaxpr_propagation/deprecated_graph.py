from string import digits
from IPython.display import display, SVG
import io
import matplotlib.pyplot as plt
import networkx as nx

import jax.numpy as jnp

from jax.core import Literal
from probjax.core.custom_primitives.random_variable import rv_p


NODE_STYLES = {
    "const": dict(
        style="",
        color="goldenrod1",
        shape="circle",
        fontsize=9,
        margin=0.01,
        width=0.2,
        height=0.2,
        regular=True,
    ),
    "invar": dict(
        fillcolor="mediumspringgreen",
        style="filled",
        shape="circle",
        fontsize=9,
        width=0.2,
        height=0.2,
        margin=0.01,
        regular=True,
    ),
    "outvar": dict(
        style="filled",
        fillcolor="indianred1",
        color="black",
        shape="circle",
        width=0.2,
        height=0.2,
        fontsize=9,
        margin=0.01,
        regular=True,
    ),
    "operation": dict(
        shape="box",
        color="black",
        fontsize=9,
        regular=True,
        margin=0.01,
        width=0.2,
        height=0.2,
    ),
    "random_variable": dict(shape="circle", style="bold", width=0.2, height=0.2),
    "latent": dict(
        style="filled",
        color="cornflowerblue",
        shape="circle",
        width=0.2,
        height=0.2,
        fontsize=8,
        margin=0.01,
        regular=True,
    ),
}


def eqn_pretty_print(eqn):
    name = eqn.primitive.name
    if name == "integer_pow":
        y = eqn.params["y"]
        return f"pow({y})"
    elif name == "convert_element_type":
        return str(eqn.params["new_dtype"])
    elif name == "broadcast_in_dim":
        return f"broadcast"
    elif name == "concatenate":
        return "concat"
    else:
        return name


def _round_name_to_str(n):
    if hasattr(n, "val"):
        val = float(n.val)
        val = round(val, 2)
        name = str(val)
    else:
        name = str(n)
    return name


from networkx.algorithms import bipartite


class JaxprGraph:
    def __init__(self, invars, constvars, outvars, eqns) -> None:
        self.graph = nx.DiGraph()
        self.invars = invars
        self.outvars = outvars
        self.constvars = constvars
        self.eqns = eqns

        # Adding constants
        for n in self.constvars:
            self.graph.add_node(
                _round_name_to_str(n),
                tag="const",
                val=n,
            )

        # Adding invars
        for n in self.invars:
            self.graph.add_node(_round_name_to_str(n), tag="invar", val=n)

        # Adding outvars
        for n in self.outvars:
            self.graph.add_node(_round_name_to_str(n), tag="outvar", val=n)

        # Adding equations
        for i, eqn in enumerate(self.eqns):
            # Add function node
            self.graph.add_node(
                f"f{i}",
                tag="operation",
                index=i,
                xlabel=eqn_pretty_print(eqn),
            )

            # Add invars
            in_vars = eqn.invars
            for n in in_vars:
                name = _round_name_to_str(n)
                if name not in self.graph.nodes:
                    if isinstance(n, Literal):
                        val = n.val
                        self.graph.add_node(name, tag="const", val=val)
                    else:
                        self.graph.add_node(
                            name,
                            tag="latent",
                            val=n,
                        )

                self.graph.add_edge(_round_name_to_str(n), f"f{i}")
            # Add outvars edges
            out_vars = eqn.outvars
            for n in out_vars:
                self.graph.add_edge(f"f{i}", _round_name_to_str(n))

    def invert(self):
        new_graph = JaxprGraph(
            outvars=self.invars,
            invars=self.outvars,
            constvars=self.constvars,
            eqns=self.eqns,
        )
        new_graph.graph = self.graph.reverse()
        for n in new_graph.graph.nodes():
            if new_graph.graph.nodes(data="tag")[n] == "invar":
                new_graph.graph.nodes(data=True)[n]["tag"] = "outvar"
            elif new_graph.graph.nodes(data="tag")[n] == "outvar":
                new_graph.graph.nodes(data=True)[n]["tag"] = "invar"
        return new_graph

    def compute_graph(self):
        nodes = self.compute_nodes()
        subgraph = bipartite.projected_graph(self.graph, nodes)
        graph = JaxprGraph(
            invars=self.invars,
            outvars=self.outvars,
            constvars=self.constvars,
            eqns=self.eqns,
        )
        graph.graph = subgraph
        return graph

    def variable_graph(self):
        nodes = self.variable_nodes()
        subgraph = bipartite.projected_graph(self.graph, nodes)
        graph = JaxprGraph(
            invars=self.invars,
            outvars=self.outvars,
            constvars=self.constvars,
            eqns=self.eqns,
        )
        graph.graph = subgraph
        return graph

    def compute_nodes(self):
        return [n for n, tag in self.graph.nodes(data="tag") if tag == "operation"]

    def variable_nodes(self):
        return [n for n, tag in self.graph.nodes(data="tag") if tag != "operation"]

    def const_nodes(self):
        return [n for n, tag in self.graph.nodes(data="tag") if tag == "const"]

    def input_nodes(self):
        return [n for n, tag in self.graph.nodes(data="tag") if tag == "invar"]

    def output_nodes(self):
        return [n for n, tag in self.graph.nodes(data="tag") if tag == "outvar"]

    def children(self, node):
        return list(self.graph.successors(node))

    def parents(self, node):
        return list(self.graph.predecessors(node))

    def subgraph(self, nodes):
        # Todo adjust vars and outvars
        new_graph = JaxprGraph(
            outvars=self.invars,
            invars=self.outvars,
            constvars=self.constvars,
            eqns=self.eqns,
        )
        new_graph.graph = self.graph.subgraph(nodes)
        return new_graph

    def __repr__(self):
        AGraph = nx.nx_agraph.to_agraph(self.graph)
        # Node styles by tag
        nodes = AGraph.nodes()
        for n in nodes:
            attributes = dict(n.attr)
            n.attr.update(NODE_STYLES[attributes.get("tag", "latent")])

        # for e in AGraph.edges():
        #     weight = float(e.attr.get("weight", 1))
        #     e.attr["label"] = f"{weight:.1f}"
        #     e.attr["fontsize"] = 6

        # Left to right in topological order
        AGraph.graph_attr["rankdir"] = "LR"
        AGraph.layout("dot")

        # Render for jupyter
        svg = io.BytesIO()
        AGraph.draw(svg, format="svg")
        svg.seek(0)
        display(SVG(svg.read()))
        return ""



import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = 0

    __repr__ = __str__ = lambda self: str([(x[-1], x[0]) for x in self.heap if x[-1] is not None])

    def insert(self, element, cost):
        if element in self.entry_finder:
            self.remove(element)
        entry = [cost, self.counter, element]
        self.entry_finder[element] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def __contains__(self, element):
        return element in self.entry_finder

    def remove(self, element):
        entry = self.entry_finder.pop(element)
        entry[-1] = None

    def update_cost(self, element, new_cost):
        self.remove(element)
        self.insert(element, new_cost)

    def pop(self):
        while self.heap:
            _, _, element = heapq.heappop(self.heap)
            if element is not None:
                del self.entry_finder[element]
                return element
        raise IndexError("Priority queue is empty.")

    def is_empty(self):
        return len(self.entry_finder) == 0





def propagate(graph, cost_fn, process_fn):
    # Propagates in a topological order, but prioritizes lower cost nodes
    queue = PriorityQueue()
    known_variables = set()
    processed_functions = set()


    outs = []
    leaf_nodes = [n for n, tag in graph.nodes(data="tag") if tag == "const" or tag == "invar"]
    # Add all initially known variables.
    # Add all processing nodes that can be evaluated with these variables
    for node in leaf_nodes:
        known_variables.add(node)
        neighbors = list(graph.neighbors(node)) +  list(graph.predecessors(node))
        for neighbor in neighbors:
            children = list(graph.neighbors(neighbor))   # Outputs
            parents = list(graph.predecessors(neighbor)) # Inputs
            known_parents = [parent in known_variables for parent in parents]
            known_children = [child in known_variables for child in children]
            queue.insert(neighbor, cost_fn(known_parents, neighbor, known_children))
    
    while not queue.is_empty():
        print(queue)
        node = queue.pop()
        children = list(graph.neighbors(node))   # Outputs
        parents = list(graph.predecessors(node)) # Inputs
        known_parents = [parent in known_variables for parent in parents]
        known_children = [child in known_variables for child in children]

        # Process node
        processed_functions.add(node)
        out = process_fn(known_parents, node,known_children)
        outs.append(out)

        neighbors = children + parents
        # This are all output variables
        for out in neighbors:
            known_variables.add(out)
            # Go over all operations in which the output is an input
            neighbors = list(graph.neighbors(out)) +  list(graph.predecessors(out))
            # Go over all operations in which the output is an input
            for operation in neighbors:
                if operation in processed_functions:
                    continue
                # The parents of all processing nodes are all the input variables to these operations
                parents = graph.predecessors(operation)
                children = graph.neighbors(operation)
                known_parents = [parent in known_variables for parent in parents]
                known_children = [child in known_variables for child in children]
                queue.insert(operation, cost_fn(known_parents, operation, known_children))
        
    return outs


def inverse_cost_fn(known_parents,node, known_children):
    print(known_parents, known_children, node)
    num_parents = len(known_parents)
    num_children = len(known_children)
    # Univariate fn
    if num_parents == 1 and num_children == 1:
        if known_parents[0] and not known_children[0]:
            return 0.5
        elif not known_parents[0] and known_children[0]:
            return 0. 
        else:
            return 1
    else:
        known = sum(known_parents) + sum(known_children)
        if known == num_parents + num_children - 1:
            return 0
        else:
            return 2
    
def inverse_process_fn(known_parents,node, known_children):
    assert sum(known_parents) + sum(known_children) == len(known_parents) + len(known_children) - 1


def default_cost_fn(known_parents, known_children, node):
    if all(known_parents):
        return 0
    else:
        return 1

def default_process_fn(known_parents, known_children, node):
    assert all(known_parents)
    return (known_parents, node, known_children)