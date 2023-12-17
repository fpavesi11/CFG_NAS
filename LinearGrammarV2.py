import numpy as np
from typing import Union

class LinearProductionRules:
    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers

    def linear_rules(self, previous_node):
        previous_node_type = previous_node[0]
        # START
        if previous_node_type == '<start>':
            node_type = 'linear'
        elif previous_node_type == 'linear':
            node_sampled = np.random.randint(low=0, high=2)  # <-- two possible layers
            if node_sampled == 0:
                node_type = 'linear'
            else:
                node_type = 'dropout'
        # DROPOUT
        elif previous_node_type == 'dropout':
            node_type = 'linear'
        else:
            raise ValueError('Unexpected ' + previous_node[0])

        return node_type

    def grow_tree(self):
        tree = [['<start>', None, None]]
        for node in range(self.n_layers):
            new_node = [self.linear_rules(tree[-1]), None, None]
            tree.append(new_node)
        tree.append(['<end>',None,None])
        # restore parameters
        return tree


class LinearGrammar:
    def __init__(self, input_dim: tuple, output_dim: int, n_layers: int, hidden_in: int, hidden_out: int) -> None:

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_in = hidden_in
        self.hidden_out = hidden_out
        self.production_rules = LinearProductionRules(n_layers=n_layers)

    @staticmethod
    def linear_convergence(initial_point, end_point, n_steps):
        steps = n_steps - 1
        m = (initial_point - end_point)/(0 - steps)
        q = (0 - steps*initial_point)/(0-steps)
        dimensions = []
        pos = 0
        for step in range(n_steps):
            new_dim = round(m * pos + q)
            dimensions.append(new_dim)
            pos += 1
        return dimensions

    def produceNetwork(self):
        tree = self.production_rules.grow_tree()
        n_layers = len(tree)-2

        # <START> NODE----
        tree[0][1] = self.input_dim
        tree[0] = tuple(tree[0])

        # LINEAR LAYERS---
        linear_params_seq = self.linear_convergence(self.hidden_in,
                                                    self.hidden_out,
                                                    n_layers)
        for j, node in enumerate(range(1, len(tree)-1)): #notice node is just a position and not a node
            refactored_node = tree[node]
            if refactored_node[0] == 'dropout':
                P = np.random.beta(a=2, b=5)  # Sample from a Beta with most of the mass on values lower than 0.5
                P = np.clip(P, a_min=0.05,
                            a_max=0.9)
                refactored_node[1] = P
            else:
                refactored_node[1] = linear_params_seq[j] #<--- assing neurons
            tree[node] = tuple(refactored_node)

        # END
        tree[-1][1] = self.output_dim
        tree[-1] = tuple(tree[-1])

        return tree

