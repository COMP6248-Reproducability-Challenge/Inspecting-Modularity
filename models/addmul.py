import torch as T
import torch.nn.functional as F
import numpy as np

from mask import MaskBase
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
        digits = T.Tensor([ [int(inp/10), int(inp%10)] for inp in inputs],) # digits turned into tensor of individual digits
        inp = F.one_hot(digits.to(T.int64), 10).numpy()

        digits = T.Tensor([int(output/10), output%10])
        otp = F.one_hot(digits.to(T.int64), 10).numpy()

        digits = T.IntTensor([operator])
        oper = F.one_hot(digits.to(T.int64), 2).numpy()
        return inp, otp.flatten('C'), oper


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
            inp = inp.flatten('C')
            inp_flat = np.concatenate((inp, op[0]))
            reformat_inp.append(inp_flat)


        return reformat_inp, outputs_

    def learn(self, batch):
        self.network.optimiser.zero_grad()

        inp = [[b[0].item(), b[1].item()] for b in batch]
        otp = [b[2].item() for b in batch]
        ops = [b[3].item() for b in batch]
        inp, otp = self.set_batched_digits(inp, otp, ops)

        inp_ = T.Tensor(np.array(inp)).to(self.network.device)
        otp_ = T.Tensor(np.array(otp)).to(self.network.device)

        otp_pred = self.network.forward(inp_)

        loss = self.network.loss(otp_, otp_pred).to(self.network.device)

        loss.backward()
        self.network.optimiser.step()

        return loss.item()

    def steps(self, batch):
        with T.no_grad():

            inp = [[b[0].item(), b[1].item()] for b in batch]

            otp = [int(b[2].item()) for b in batch]
            ops = [b[3].item() for b in batch]
            inp, otp__ = self.set_batched_digits(inp, otp, ops)

            inp_ = T.Tensor(np.array(inp)).to(self.network.device)

            otp_pred = self.network.forward(inp_)
            pred = []
            for o in otp_pred:
                x = [i%10 for i, t in enumerate(o) if t > 0.7]
                if len(x) == 2: # so if we determine a 2 digit number
                    val_ = x[0]*10 +x[1]
                    pred.append(val_)
                else:
                    pred.append(100)

            correct = 0
            for i in range(len(otp)):
                if otp[i] == pred[i]:
                    correct += 1
            acc = correct/len(otp)
            return acc

    def init_mask(self, batch):
        """Method for Mask training (Section 2)

        Notes
        -----
            We take learned logits (initialised probabilities of node being chose as 0.9) `l_i` (i for index of each
            node).
        """
        # Freeze layers
        # msk = MaskBase(self.network)
        with T.no_grad():
            for param in self.network.parameters():
                param.requires_grad = False

            inp = [[b[0].item(), b[1].item()] for b in batch]

            otp = [int(b[2].item()) for b in batch]
            ops = [b[3].item() for b in batch]
            inp, otp__ = self.set_batched_digits(inp, otp, ops)

            inp_ = T.Tensor(np.array(inp)).to(self.network.device)

            otp_pred, mask = self.network.forward_init_mask(inp_)

            return mask

    def learn_mask(self, batch, masks):
        self.network.optimiser.zero_grad()
        with T.no_grad():
            inp = [[b[0].item(), b[1].item()] for b in batch]
            otp = [b[2].item() for b in batch]
            ops = [b[3].item() for b in batch]
            inp, otp = self.set_batched_digits(inp, otp, ops)

            inp_ = T.Tensor(np.array(inp)).to(self.network.device)
            otp_ = T.Tensor(np.array(otp)).to(self.network.device)

            otp_pred = self.network.forward_mask(inp_)

            loss = self.network.loss(otp_, otp_pred).to(self.network.device)

            loss.backward()
            self.network.optimiser.step()

            return loss.item()

