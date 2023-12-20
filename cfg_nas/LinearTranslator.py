from torch import nn

"""
NOTES: Actually ConvTranslator is just an extension of this translator. In future implementations it could be useful to
harmonize in only one translator
"""


class GrammarToTorch:
    def __init__(self, default_activation=nn.ReLU(), default_final_activation=nn.Sigmoid()):
        self.default_activation = default_activation
        self.default_final_activation = default_final_activation

    def torch_encode(self, network_tree):
        model = nn.Sequential()
        for depth in range(len(network_tree)):
            token = network_tree[depth]

            # START token sets the initial dimension, no module is created
            if token[0] == '<start>':
                pass #<-- no action is needed

            # LINEAR token
            elif token[0] == 'linear':
                if network_tree[depth - 1][0] in ('linear', '<start>'):
                    input_dimension = network_tree[depth - 1][1]
                elif network_tree[depth - 1][0] == 'dropout':
                    input_dimension = network_tree[depth - 2][1]
                else:
                    raise ValueError('Unexpected ' + network_tree[depth-1][0]
                                     + ' before ' + network_tree[depth][0])

                module = nn.Linear(input_dimension, token[1])
                model.append(module)
                model.append(self.default_activation)

            # DROPOUT token
            elif token[0] == 'dropout':
                module = nn.Dropout(token[1])
                model.append(module)

            # END token
            elif token[0] == '<end>':
                if network_tree[depth - 1][0] in ('linear', '<start>'):
                    input_dimension = network_tree[depth - 1][1]
                elif network_tree[depth - 1][0] == 'dropout':
                    input_dimension = network_tree[depth - 2][1]
                else:
                    raise ValueError('Unexpected ' + network_tree[depth-1][0]
                                     + ' before ' + network_tree[depth][0])

                module = nn.Linear(input_dimension, token[1])
                model.append(module)
                model.append(self.default_final_activation)

            # UNKNOWN TOKEN
            else:
                raise ValueError('Token ' + token[0] + ' is an unknown type')

        return model


class TranslatedNetwork(nn.Module):
    def __init__(self, network_tree, default_activation=nn.ReLU, default_final_activation=nn.Sigmoid):
        super(TranslatedNetwork, self).__init__()
        self.default_activation = default_activation
        self.default_final_activation = default_final_activation
        self.network_tree = network_tree

        self.model_encoder = GrammarToTorch(default_activation, default_final_activation)
        self.model = self.model_encoder.torch_encode(network_tree)

    def forward(self, x):
        output = self.model(x)
        return output