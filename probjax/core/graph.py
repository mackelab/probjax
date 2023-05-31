from IPython.display import display, SVG
import io
import matplotlib.pyplot as plt
import networkx as nx



NODE_STYLES = {
  'const': dict(style='', color='goldenrod1', shape="circle", fontsize=9, margin=0.01,width=0.2, height=0.2, regular=True),
  'invar': dict(fillcolor='mediumspringgreen', style='filled', shape="circle", fontsize=9,width=0.2, height=0.2, margin=0.01, regular=True),
  'outvar': dict(style='filled', fillcolor='indianred1', color='black', shape="circle",width=0.2, height=0.2, fontsize=9, margin=0.01, regular=True),
  'operation': dict(shape='box', color='black', fontsize=9, regular=True, margin=0.01,width=0.2, height=0.2),
  'random_variable': dict(shape='circle', style='bold',width=0.2, height=0.2),
  'latent': dict(style='filled', color='cornflowerblue', shape="circle", width=0.2, height=0.2, fontsize=8, margin=0.01, regular=True),
}

def eqn_pretty_print(eqn):
    name = eqn.primitive.name
    if name == "integer_pow":
        y = eqn.params["y"]
        return f"pow(2)"
    elif name == "convert_element_type":
        return str(eqn.params["new_dtype"])
    elif name == "broadcast_in_dim":
        return f"broadcast"
    elif name == "concatenate":
        return "concat"
    else:
        return name


class FactorGraph():
    def __init__(self, jaxpr) -> None:
        self.graph = nx.DiGraph()
        
        # Adding constants
        for n in jaxpr.constvars:
            self.graph.add_node(str(n), tag='const', value=n, **NODE_STYLES["const"])

        # Adding invars
        for n in jaxpr.invars:
            self.graph.add_node(str(n), tag='invar', value=n, **NODE_STYLES["invar"])

        # Adding outvars
        for n in jaxpr.outvars:
            self.graph.add_node(str(n), tag='outvar', value=n, **NODE_STYLES["outvar"])

        # Adding equations
        for i,eqn in enumerate(jaxpr.eqns):

            # Add function node
            if eqn.primitive is rv_p:
                self.graph.add_node(f"f{i}", tag='operation', value=eqn, xlabel=eqn_pretty_print(eqn), **NODE_STYLES["operation"])
            else:
                self.graph.add_node(f"f{i}", tag='operation', value=eqn, xlabel=eqn_pretty_print(eqn), **NODE_STYLES["operation"])

            # Add invars
            in_vars = eqn.invars
            for n in in_vars:
                if str(n) not in self.graph.nodes:
                    self.graph.add_node(str(n), tag='latent', value=n, **NODE_STYLES["latent"])

                self.graph.add_edge(str(n), f"f{i}")
            # Add outvars
            out_vars = eqn.outvars
            for n in out_vars:
                if str(n) not in self.graph.name:
                    self.graph.add_node(str(n), tag='latent', value=n, **NODE_STYLES["latent"])
                
                self.graph.add_edge(f"f{i}", str(n))

    def nodes(self, data=False, filters=None):
        out = self.graph.nodes(data=data)
        if filters is None:
            return out
        else:
            return filter(filters, out)

    def __repr__(self):
        AGraph = nx.nx_agraph.to_agraph(self.graph)
        AGraph.graph_attr["rankdir"] = 'LR'
        AGraph.layout('dot')
        svg  = io.BytesIO()
        AGraph.draw(svg, format='svg')
        svg.seek(0)
        display(SVG(svg.read()))
        return ''
            
