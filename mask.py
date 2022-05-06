import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Linear).__init__()
        self.indims = input_dims
        self.outdims = output_dims

        self.weight = torch.nn.Parameter(torch.Tensor(output_dims, input_dims))
        self.bias = torch.nn.Parameter(torch.Tensor(output_dims))


class Layer:
    def __init__(self, name, param):
        self.name = name
        self.param = param
        h,w = self.param.size()
        self.layer = Linear(h,w)

        self.mask = param.ge(0.3)
        print(torch.masked_select(param, self.mask))




class MaskBase():
    def __init__(self, network:nn.Module):
        self.layers = []

        lay = network.layers[0]
        names = []
        params = []
        param_dict = {}
        for name, param in network.named_parameters():
            if 'weight' in name:
                a = name.split('.')
                name_ = a[0]+a[1]+a[2]+a[3]
                print(name_)
                self.layers.append(Layer(name, param.detach()))
                param_dict[name_] = nn.Parameter(torch.full_like(param, 1))
        print(param_dict)

        P_dict = nn.ParameterDict(param_dict)
        print(P_dict)
        n_mask_sets = 2
        # self.masks = torch.nn.ModuleList([torch.nn.ParameterDict({k: torch.nn.Parameter(torch.full_like(v, 1))
        #                                                           for k, v in network.named_parameters() if
        #                                                           'weight' in k})
        #                                   for _ in range(n_mask_sets)])

        # print(self.masks)
        # for layer in self.layers:
        #     print(layer.name)
        #     for i, p in enumerate(layer.param):
        #         for j, p_i in enumerate(p):
        #             print(p_i.item())
        #             if p_i.item() < 0.9:
        #                 pass
        #             else:
        #                 layer.mask[i][j] = True
        #                 print(layer.p_i)


