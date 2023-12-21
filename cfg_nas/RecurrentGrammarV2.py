import numpy as np

class RecurrentProductionRules:
    def __init__(self, n_layers: int, max_recurrent_layers:int = 1) -> None:
        assert max_recurrent_layers > 0 and max_recurrent_layers <= n_layers, 'Invalid maximum number of recurrent layers'
        self.n_layers = n_layers
        self.max_recurrent_layers = max_recurrent_layers
        self.n_recurrent_layers = 0

    def recurrent_rules(self, previous_node):
        previous_node_type = previous_node[0]
        # START
        if previous_node_type == '<start>':
            node_sampled = np.random.randint(low=0, high=2) #0 GRU, 1 LSTM
            if node_sampled == 0:
                node_type = 'GRU'
                self.n_recurrent_layers += 1
            else:
                node_type = 'LSTM'
                self.n_recurrent_layers += 1
        # RECURRENT
        elif previous_node_type in ('GRU', 'LSTM'):
            if self.n_recurrent_layers >= self.max_recurrent_layers:
                node_type = 'linear'
            else:
                node_sampled = np.random.randint(low=0, high=3)  # 0 GRU, 1 LSTM, linear
                if node_sampled == 0:
                    node_type = 'GRU'
                    self.n_recurrent_layers += 1
                elif node_sampled == 1:
                    node_type = 'LSTM'
                    self.n_recurrent_layers += 1
                else:
                    node_type = 'linear'
        # LINEAR
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
            new_node = [self.recurrent_rules(tree[-1]), None, None]
            tree.append(new_node)
        tree.append(['<end>', None, None])
        # restore parameters
        self.n_recurrent_layers = 0
        return tree

class RecurrentGrammar:
    def __init__(self, input_dim: int, output_dim: int, hidden_in: int, hidden_out: int,
                 n_layers: int, max_recurrent_layers: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_in = hidden_in
        self.hidden_out = hidden_out
        self.production_rules = RecurrentProductionRules(n_layers=n_layers,
                                                         max_recurrent_layers=max_recurrent_layers)

    @staticmethod
    def linear_convergence(initial_point, end_point, n_steps):
        steps = n_steps - 1
        m = (initial_point - end_point) / (0 - steps)
        q = (0 - steps * initial_point) / (0 - steps)
        dimensions = []
        pos = 0
        for step in range(n_steps):
            new_dim = round(m * pos + q)
            dimensions.append(new_dim)
            pos += 1
        return dimensions

    def produceNetwork(self):
        tree = self.production_rules.grow_tree()

        # <START> NODE----
        tree[0][1] = self.input_dim
        tree[0] = tuple(tree[0])

        # RECURRENT LAYERS----
        recurrent_positions = []
        for j, node in enumerate(tree):
            if node[0] in ('LSTM', 'GRU'):
                node[1] = self.hidden_in
                tree[j] = tuple(node)
                recurrent_positions.append(j)

        # LINEAR LAYERS----
        num_linear_layers = len(tree) - len(recurrent_positions) - 2 + 1
        linear_params_seq = self.linear_convergence(self.hidden_in,
                                                    self.hidden_out,
                                                    num_linear_layers)
        linear_params_seq = linear_params_seq[1:]

        for j, node in enumerate(range(recurrent_positions[-1]+1, len(tree)-1)):
            refactored_node = tree[node]
            if refactored_node[0] == 'dropout':
                P = np.random.beta(a=2, b=5)  # Sample from a Beta with most of the mass on values lower than 0.5
                P = np.clip(P, a_min=0.05,
                            a_max=0.9)
                refactored_node[1] = P
            else:
                refactored_node[1] = linear_params_seq[j]  # <--- assing neurons
            tree[node] = tuple(refactored_node)

        # END
        tree[-1][1] = self.output_dim
        tree[-1] = tuple(tree[-1])

        return tree