from models.addmul import HandleAddMul
import torch
import numpy as np

network_cache_dir = "networks/cache-networks/"
network_name = "lyr256-split0.8-lr0.01-mul.data"

checkpoint = True
test_flag = 1

input_dims = [42]
output_dims = [20]
batchsize = 128
num_epochs = 1

handler = HandleAddMul(input_dims, output_dims, dir=network_cache_dir + network_name, checkpoint=checkpoint, lr=0.001)

logits = []
for layer in handler.network.layers[0]:
    if isinstance(layer, torch.nn.Linear):
        logits.append(torch.full_like(layer.weight.data.clone(), 0.9, requires_grad=True))

for name, param in handler.network.named_parameters():
    param.requires_grad = False


train_split = 0.8
test_split = 1 - train_split

data_fp = "generate_datasets/tmp/digit-data/simple_add.npy"
data = np.load(data_fp, allow_pickle=True)

data_len = len(data)
train_split_idx = int(data_len * train_split)
train_data = data[:train_split_idx]
test_data = data[train_split_idx:]

train_loader = torch.utils.data.DataLoader(dataset=torch.tensor(train_data), batch_size=batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=torch.Tensor(test_data), batch_size=batchsize, shuffle=True)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

iterator_train = iter(cycle(train_loader))
iterator_test = iter(test_loader)

criterion = torch.nn.MSELoss()

optimiser = torch.optim.Adam(logits, lr=0.01)

NUM_EPOCHS = 20000  # NB: check for number of training epochs in paper
tau = 1  # temperature parameter, NB: check for value in paper
alpha = 0.0001/128  # regularisation parameter, NB: check for value in paper

import matplotlib.pyplot as plt
loss_hist = []
for e in range(NUM_EPOCHS):
    print(f'Starting epoch {e}...')

    '''Sampling and generating masks.'''

    U1 = torch.rand(1, requires_grad=True).to(handler.network.device)
    U2 = torch.rand(1, requires_grad=True).to(handler.network.device)

    samples = []

    for layer in logits:
        layer.requires_grad_(requires_grad=True)

        #         if layer.grad is not None:
        #             layer.grad.detach_()
        #             layer.grad.zero_()

        samples.append(torch.sigmoid((layer - torch.log(torch.log(U1) / torch.log(U2))) / tau))

    binaries_stop = []

    for layer in samples:
        with torch.no_grad():
            binaries_stop.append((layer > 0.5).float() - layer)

    binaries = []
    iterator_samples = iter(samples)

    for layer in binaries_stop:
        binaries.append(layer + next(iterator_samples))

    iterator_binaries = iter(binaries)

    for layer in handler.network.layers[0]:
        if isinstance(layer, torch.nn.Linear):
            with torch.no_grad():
                layer.weight.data * next(iterator_binaries)

    '''Inference with masked network and backpropagation.'''

    batch = next(iterator_train)

    with torch.no_grad():
        inp = torch.stack([torch.stack([b[0], b[1]]) for b in batch])
        otp = torch.stack([b[2] for b in batch])
        ops = torch.stack([b[3] for b in batch])
        inp, otp_ = handler.set_batched_digits(inp, otp, ops)
        inp_ = inp.to(handler.network.device)
        otp_ = otp_.to(handler.network.device)

        otp_pred = handler.network(inp_)

        pred = torch.stack((otp_pred[:,0:10], otp_pred[:,10:]))
        pred = torch.argmax(pred, dim=2)
        pred[0,:] = pred[0,:]*10
        pred = torch.sum(pred, dim=0).to(handler.network.device)

        otp_target = otp.float().to(handler.network.device)
    # otp = torch.Tensor(np.array(otp_)).to(handler.network.device)
    # pred = torch.tensor()).to(handler.network.device)

    all_logits = alpha*torch.cat([layer.view(-1) for layer in logits]).to(handler.network.device)
    optimiser.zero_grad()
    loss = criterion(pred, otp_target).to(handler.network.device) + torch.sum(all_logits)
    #loss.requires_grad = True
    loss.backward()
    optimiser.step()

    loss_hist.append(loss.item())

    if e % 200 == 0:
        plt.cla()
        plt.clf()
        plt.plot(loss_hist)
        plt.savefig('liveplot.png')
        plt.cla()
        plt.clf()
        plt.close()
        torch.save(logits, 'trained_logits.pt')

