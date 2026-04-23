import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.0):
        super(MLPRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)
