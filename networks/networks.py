import torch as T
import torch.nn as nn
from typing import List

class FNN(nn.Module):
    def __init__(self, layer_dims:List[int], dir, lr:float=0.02):
        super().__init__()
        num_layers = len(layer_dims) # number of layers (including input and output layers)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.ReLU()
            ) if i != (num_layers-2)
            else nn.Linear(layer_dims[i], layer_dims[i+1])
            for i in range(num_layers-1)
        ])
        self.sig = nn.Sigmoid()

        self.optimiser = T.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'... FNN Network training on {self.device} ...')
        self.to(self.device)

        print(f'Accessing : {dir}')
        self.save_file = dir

    def forward(self, input: T.Tensor) -> T.Tensor:
        for layer in self.layers:
            input = layer(input)
        return input

    def forward_test(self, input: T.Tensor) -> T.Tensor:
        for layer in self.layers:
            input = layer(input)
        input = self.sig(input)
        return input

    def save_(self):
        print('Saving network ...')
        T.save(self.state_dict(), self.save_file)

    def load_save(self):  # file
        print('Load saves ...')
        self.load_state_dict(T.load(self.save_file))