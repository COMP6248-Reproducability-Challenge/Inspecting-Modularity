import torch as T
import torch.nn as nn
from typing import List

class FNN(nn.Module):
    def __init__(self, layer_dims:List[int]):
        super().__init__()
        num_layers = len(layer_dims) # number of layers (including input and output layers)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.ReLU
            ) if i != (num_layers-1)
            else nn.Linear(layer_dims[i], layer_dims[i+1])
            for i in range(num_layers)
        ])

    def forward(self, input: T.Tensor) -> T.Tensor:
        for layer in self.layers:
            input = layer(input)
        return input