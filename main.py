from models.addmul import HandleAddMul
import torch as T
import torch.nn as nn
import numpy as np

training_split = 0.8
testing_split = 1 - training_split

network_cache_dir = "networks/cache-networks/"
netwrok_name = "lyr256-split0.8-lr0.01-mul.data"

checkpoint = True # shall we load network or create new one
test_flag = False # are we trianing or shall we go straight to testing?

input_dims = [42]
output_dims = [20]
batchsize = 128
num_epochs = 1

# For running the add data
data_fp = "generate_datasets/tmp/digit-data/simple_mul.npy"
data = np.load(data_fp, allow_pickle=True)
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

if not test_flag:
    for e in range(num_epochs):
        running_loss = 0.0
        for idx, batch in enumerate(train_loader):
            running_loss += admu.learn(batch)

        print(f'Epoch {e} : Loss {running_loss/idx }')

        if e % 20 == 0 and e > 20:
            admu.network.save_()

    admu.train_mask()
acc = 0.0
steps = 0
for idx, batch in enumerate(test_loader):
    acc += admu.steps(batch)
    steps += 1
print(f'Accuracy {acc/steps}')
