import numpy as np
from typing import Union

class ImageProductionRules:
    def __init__(self, n_layers: int, min_spatial_layers: int) -> None:
        assert n_layers >= min_spatial_layers, 'number of layers must be >= than minimun number of spatial layers'
        self.n_layers = n_layers
        self.min_spatial_layers = min_spatial_layers
        self.remaining_spatial_layers = min_spatial_layers

    def convolutional_rules(self, previous_node):
        previous_node_type = previous_node[0]
        # START
        if previous_node_type == '<start>':
            node_type = 'conv2d'
            self.remaining_spatial_layers -= 1
        # CONV2D
        elif previous_node_type == 'conv2d':
            # Objective spatial layers reached
            if self.remaining_spatial_layers <= 0:
                node_sampled = np.random.randint(low=0, high=2) # conv, flatten
                if node_sampled == 0: #conv
                    node_type = 'conv2d'
                else: #flatten
                    node_type = 'flatten'
            # Objective spatial layers not reached yet
            else:
                node_type = 'conv2d'
                self.remaining_spatial_layers -= 1

        # FLATTEN
        elif previous_node_type == 'flatten':
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

    @staticmethod
    def check_tree_validity(tree):
        flatten_found = False
        for node in tree:
            if node[0] == 'flatten':
                flatten_found = True
                break
        return flatten_found

    def grow_tree(self):
        tree = [['<start>', None, None]]
        for node in range(self.n_layers):
            new_node = [self.convolutional_rules(tree[-1]), None, None]
            tree.append(new_node)
        if not self.check_tree_validity(tree):
            tree.append(['flatten',None,None])
        tree.append(['<end>',None,None])
        # restore parameters
        self.remaining_spatial_layers = self.min_spatial_layers
        return tree


class ImageGrammar:
    def __init__(self, input_dim: tuple, channels: int, output_dim: int, n_layers: int, min_spatial_layers: int,
                 hidden_in: int, hidden_out: int, shrinkage_objective: Union[tuple, list, float]) -> None:

        assert input_dim[0] % 2 == 0 and input_dim[1] % 2 == 0, 'Input must have even dimensions'
        if isinstance(shrinkage_objective, float) is False:
            assert shrinkage_objective[0] % 2 == 0 and shrinkage_objective[1] % 2 == 0, 'Shrinkage must be even'
        self.input_dim = input_dim
        self.channels = channels
        self.output_dim = output_dim
        self.hidden_in = hidden_in
        self.hidden_out = hidden_out
        self.shrinkage_objective = shrinkage_objective
        self.production_rules = ImageProductionRules(n_layers=n_layers,
                                                     min_spatial_layers=min_spatial_layers)

        if isinstance(shrinkage_objective, float):
            self.shrinkage_objective = self.calculate_shrinkage_obj()

    def calculate_shrinkage_obj(self):
        height = int(self.input_dim[0] * self.shrinkage_objective)
        width = int(self.input_dim[1] * self.shrinkage_objective)
        if height % 2 == 1: # odd dimensions not allowed
            height += 1
        if width % 2 == 1: #odd dimensions not allowed
            width += 1
        return (height, width)

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

    @staticmethod
    def calculate_kernel(input_dim, objective_dim, steps):
        # HEIGHT
        difference_height = input_dim[0] - objective_dim[0]
        step_diff = difference_height / steps
        kernel_height = int(step_diff)
        if kernel_height % 2 == 0:  # even number
            kernel_height += 1
        kernel_heights_seq = [kernel_height] * steps
        residual_height = (input_dim[0] - steps*(kernel_height-1)) - objective_dim[0]
        if residual_height != 0:
            n_to_increase = int(residual_height/2)
            for node in range(n_to_increase):
                kernel_heights_seq[node] += 2
        # WIDTH
        difference_width = input_dim[1] - objective_dim[1]
        step_diff = difference_width / steps
        kernel_width = int(step_diff)
        if kernel_width % 2 == 0:  # even number
            kernel_width += 1
        kernel_widths_seq = [kernel_width] * steps
        residual_width = (input_dim[1] - steps * (kernel_width - 1)) - objective_dim[1]
        if residual_width != 0:
            n_to_increase = int(residual_width / 2)
            for node in range(n_to_increase):
                kernel_widths_seq[node] += 2
        # build sequence
        kernel_seq = []
        for step in range(steps):
            kernel_seq.append((kernel_heights_seq[step], kernel_widths_seq[step]))
        return kernel_seq

    @staticmethod
    def check_output_dim(input_dim, kernel_seq):
        out_dim = list(input_dim)
        for kernel in kernel_seq:
            out_dim[0] -= kernel[0] - 1
            out_dim[1] -= kernel[0] - 1
        return out_dim

    def produceNetwork(self):
        tree = self.production_rules.grow_tree()

        # <START> NODE----
        tree[0][1] = self.channels
        tree[0][2] = self.input_dim
        tree[0] = tuple(tree[0])

        # SPATIAL LAYERS-----
        # count number of spatial layers
        spatial_count = 0
        for node in tree:
            if node[0] == 'conv2d':
                spatial_count += 1

        spatial_params_seq = self.linear_convergence(self.hidden_in,
                                                     self.hidden_out,
                                                     spatial_count)

        kernel_seq = self.calculate_kernel(self.input_dim,
                                           self.shrinkage_objective,
                                           spatial_count)

        for node, (params, kernel) in enumerate(zip(spatial_params_seq, kernel_seq)):
            # node+1 because the first node is <start> node
            tree[node+1][1] = params
            tree[node+1][2] = kernel
            tree[node+1] = tuple(tree[node+1]) #<-- transform into tuple

        # FLATTENING LAYER---
        flatten_position = spatial_count + 1 #<-- +1 because first node is <start>
        out_dim = self.check_output_dim(self.input_dim, kernel_seq) #<--- THIS IS REMOVABLE (use self.shrinkage_dim)
        flattened_dim = out_dim[0]*out_dim[1] * spatial_params_seq[-1]
        tree[flatten_position][1] = flattened_dim #<-- store the flattened dimension to simplify network encoding
        tree[flatten_position] = tuple(tree[flatten_position])

        # LINEAR LAYERS---
        remaining_positions = len(tree[flatten_position:]) # DO NOT EXCLUDE LAST NODE
        # NOTE: remaining layers is computed so because we want a line going from flattened dimension
        # to input dimension, then we remove first and last as we do not need them
        linear_params_seq = self.linear_convergence(flattened_dim,
                                                    self.output_dim,
                                                    remaining_positions)
        linear_params_seq = linear_params_seq[1:-1] #EXCLUDE FIRST (flattened dim) AND LAST (end)
        for j, node in enumerate(range(flatten_position+1, len(tree)-1)): #notice node is just a position and not a node
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




