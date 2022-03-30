from models.addmul import HandleAddMul
import torch as T
import torch.nn as nn
import numpy as np

training_split = 0.8
testing_split = 1 - training_split

layer_dims = [42,256,256,20]
batchsize = 64
num_epochs = 50

# For running the add data
data_fp = "generate_datasets/tmp/digit-data/simple_add.npy"
data = np.load(data_fp, allow_pickle=True)
np.random.shuffle(data)
data_len = len(data)
train_split_idx = int(data_len * training_split)
train_data = data[:train_split_idx]
test_data = data[train_split_idx:]

admu = HandleAddMul(layer_dims)
train_loader = T.utils.data.DataLoader(dataset=T.tensor(train_data),batch_size=batchsize,
                                          shuffle=True)
test_loader = T.utils.data.DataLoader(dataset=T.Tensor(test_data),batch_size=batchsize,
                                          shuffle=True)

for e in range(num_epochs):
    running_loss = 0.0
    for idx, batch in enumerate(train_loader):
        running_loss += admu.learn(batch)

    print(f'Epoch {e} : Loss {running_loss/idx }')


acc = 0.0
steps = 0
for idx, batch in enumerate(test_loader):
    acc += admu.steps(batch)
    steps += 1
print(f'Accuracy {acc/steps}')
