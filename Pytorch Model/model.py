import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, input_size, config):
        super(MLPRegressor, self).__init__()

        n_layers     = config.get("n_layers", 2)
        first_size   = config.get("first_layer_size", 100)
        activation   = config.get("activation", "relu")
        dropout_rate = config.get("dropout_rate", 0.0)
        batch_norm   = config.get("batch_norm", False)

        # Funnel layer sizes
        sizes = [first_size]
        for i in range(1, n_layers):
            prev = sizes[-1]
            next_size = max(8, prev // 2)
            sizes.append(next_size)

        def get_activation():
            if activation == "relu":
                return nn.ReLU()
            elif activation == "leaky_relu":
                return nn.LeakyReLU(0.01)
            elif activation == "elu":
                return nn.ELU()

        layers = []
        in_size = input_size
        for out_size in sizes:
            layers.append(nn.Linear(in_size, out_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_size))
            layers.append(get_activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_size = out_size

        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)
