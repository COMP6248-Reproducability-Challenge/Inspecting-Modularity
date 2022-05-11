"""Reproduction of the FNN as described in Appendix C.4

    FNN : 20k steps
          5 layers deep
            2000 units in reach layers
          ReLU between each layer
          Adam optimiser
          lr = 0.001
          gradient clipping = 1

    Notes : Default 5 layers 2000 units is considered 'big' (Figure 14)
            If we want to test other sizes (to reproduce decreased in shared % of nodes as size increases)

"""
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, input_dims, output_dims, dir, lr:float=0.001, use_optimiser:bool=True):
        super(FNN, self).__init__()

        self.layers = nn.ModuleList([nn.Sequential(
                nn.Linear(*input_dims, 2000), # Input Layer
                nn.ReLU(),
                nn.Linear(2000, 2000), # L1
                nn.ReLU(),
                nn.Linear(2000, 2000), # L2
                nn.ReLU(),
                nn.Linear(2000, 2000), # L3
                nn.ReLU(),
                nn.Linear(2000, 2000), # L4
                nn.ReLU(),
                nn.Linear(2000, 2000), #L5
                nn.ReLU(),
                nn.Linear(2000, *output_dims) # Output Layer
            )])

        if use_optimiser:
            self.optimiser = T.optim.Adam(self.parameters(), lr=lr) # Adam optimiser
            self.loss = nn.CrossEntropyLoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'... FNN Network training on {self.device} ...')
        self.to(self.device)

        print(f'Accessing : {dir}')
        self.save_file = dir

    def forward(self, input: T.Tensor) -> T.Tensor:
        for layer in self.layers:
            input = layer(input)
        return input

    def save_(self):
        print('Saving network ...')
        T.save(self.state_dict(), self.save_file)

    def load_save(self):  # file
        self.load_state_dict(T.load(self.save_file))

    def forward_mask_layer(self, x, mask, weight, bias):
        weight = weight * mask
        return F.linear(x, weight, bias)
