import torch as T
import torch.nn.functional as F
import numpy as np

from networks.networks import FNN

class HandleAddMul():

    def __init__(self, input_layer:list, output_layer:list, lr=0.01, dir='', checkpoint=False):
        self.inp = []
        self.otp = []

        self.step_cntr = 0
        self.network = FNN(input_layer, output_layer, lr=lr, dir=dir)

        if checkpoint == True:
            print(dir)
            self.network.load_save()


    def get_digits(self, inputs: list, output: int, operator:int):
        """Set the integers inputs and outputs to corresponding digits

        :param inputs: list, list of the two (two-digit) integers

        :notes  The inputs come as two, two-digit, number whereby we need to separate the tens and unit digits, then
                determine their one-hot values (requires a tensor). e.g [12, 83] -> [[1, 2], [8, 3]] -> T.Tensor(.)
                -> F.one_hot( . , 10) = [[0100000000, 0010000000], [0000000010, 0001000000]]
        """
        digits = T.stack([T.stack([inp/10, inp%10]) for inp in inputs]) # digits turned into tensor of individual digits
        inp = F.one_hot(digits.to(T.int64), 10)

        digits = T.stack([output/10, output%10])
        otp = F.one_hot(digits.to(T.int64), 10)
        # digits = T.IntTensor([operator])
        digits = T.stack([operator])
        oper = F.one_hot(digits.to(T.int64), 2)
        return inp.float(), T.flatten(otp).float(), oper.float()


    def set_batched_digits(self, inputs:list, outputs:list, oper:list):
        """Return a batch on inputs/outputs in their 2-digit forms

        :param inputs: list, list of list of 2, 2-digit, input integers
        :param outputs: list, list of 2-digit output integers
        :return:
        """
        inputs_ = []
        outputs_ = []
        operations_ = []
        for inp, otp, op in zip(inputs, outputs, oper):
            inp_, otp_, op_ = self.get_digits(inp, otp, op)
            inputs_.append(inp_)
            outputs_.append(otp_)
            operations_.append(op_)

        reformat_inp = [] # we want a list (batch) of input tensors, (dim=0)
        for inp, op in zip(inputs_, operations_):
            inp = T.flatten(inp)
            inp_flat = T.cat((inp, op[0]))
            reformat_inp.append(inp_flat)

        return T.stack(reformat_inp), T.stack(outputs_)

    def learn(self, batch):
        # Split data into respective representations
        inp = [[b[0], b[1]] for b in batch]
        otp = [b[2] for b in batch]
        ops = [b[3] for b in batch]
        # Encode for one-hot vectors
        inp_, otp_ = self.set_batched_digits(inp, otp, ops)
        inp_ = inp_.to(self.network.device) # pass into GPU if avaliable
        otp_target = otp_.to(self.network.device)

        otp_pred = self.network(inp_)

        self.network.optimiser.zero_grad()
        loss = self.network.loss(otp_pred, otp_target).to(self.network.device)
        loss.backward()
        self.network.optimiser.step()
        return loss.item()