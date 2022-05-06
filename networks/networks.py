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

class FNN(nn.Module):
    def __init__(self, input_dims, output_dims, dir, lr:float=0.001):
        super().__init__()

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


        self.optimiser = T.optim.Adam(self.parameters(), lr=lr) # Adam optimiser
        self.loss = nn.MSELoss()
        self.sig = nn.Sigmoid()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'... FNN Network training on {self.device} ...')
        self.to(self.device)

        print(f'Accessing : {dir}')
        self.save_file = dir

    def forward(self, input: T.Tensor) -> T.Tensor:
        for layer in self.layers:
            # print(layer._get_name())
            input = layer(input)
        return input

    def forward_mask(self, input: T.Tensor) -> T.Tensor:
        for layer in self.layers:
            # print(layer._get_name())
            input = layer(input)
        return input

    def forward_init_mask(self, input: T.Tensor):

        probability_mat = []
        for layer in self.layers:
            i = input
            for name, l in layer.named_children():
                i = l.forward(i)
                if int(name) % 2 == 0:
                    #prob = nn.functional.softmax(i, dim=0).requires_grad_(False)
                    prob = i
                    probability_mat.append(prob)
            input = i

        mask_ = []
        for idx, layer in enumerate(probability_mat):
            mask = layer.ge(0.9)
            vmask, _ = torch.max(mask, dim=0)
            mask_selected = torch.masked_select(layer, vmask)
            print(mask_selected)
            mask_.append(mask_selected)
        input = self.sig(input)
        return input, mask_

    def save_(self):
        print('Saving network ...')
        T.save(self.state_dict(), self.save_file)

    def load_save(self):  # file
        print('Load saves ...')
        self.load_state_dict(T.load(self.save_file))