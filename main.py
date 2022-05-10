import torch.nn

from models.addmul import HandleAddMul
import torch as T
import torch.nn as nn
import numpy as np

training_split = 0.8
testing_split = 1 - training_split

network_cache_dir = "networks/cache-networks/"
netwrok_name = "lyr256-split0.8-lr0.01-add-mul.data"

checkpoint = False # shall we load network or create new one
test_flag = 0 # are we trianing or shall we go straight to testing?

input_dims = [42]
output_dims = [20]
batchsize = 128
num_epochs = 20000

# For running the add data
data_fp = ["generate_datasets/tmp/digit-data/simple_mul.npy",
           "generate_datasets/tmp/digit-data/simple_add.npy"]
data_mul = np.load(data_fp[0], allow_pickle=True)
data_add = np.load(data_fp[1], allow_pickle=True)
np.random.shuffle(data_add)
data_add = data_add[:len(data_mul)]
assert len(data_mul) == len(data_add)

data = np.concatenate([data_add,  data_mul])
np.random.shuffle(data)
data_len = len(data)
train_split_idx = int(data_len * training_split)
train_data = data[:train_split_idx]
test_data = data[train_split_idx:]

admu = HandleAddMul(input_dims, output_dims, dir=network_cache_dir+netwrok_name, checkpoint=checkpoint, lr=0.001)

train_loader = T.utils.data.DataLoader(dataset=T.tensor(train_data),batch_size=batchsize,
                                          shuffle=True)
test_loader = T.utils.data.DataLoader(dataset=T.Tensor(test_data),batch_size=batchsize,
                                          shuffle=True)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

iterator_train = iter(cycle(train_loader))
iterator_test = iter(cycle(test_loader))

running_loss = 0.0
if not test_flag:
    for e in range(num_epochs):
        # for idx, batch in enumerate(train_loader):
        #     running_loss += admu.learn(batch)
        batch = next(iterator_train)
        loss = admu.learn(batch)
        print(f'Epoch {e} : Loss {loss}')
        if e % 2000 == 0 and e > 2000:
            admu.network.save_()