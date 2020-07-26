import os
import time
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm_notebook

# Define data root directory
name = "custom prior less steps" # To change to match data

data_dir = "./run_outputs/regression_rollout/"
data_file = name + ".result_stream.pickle"

#### Creating the Dataset

with open(data_dir + 'ys' + data_file, 'rb') as f:
    ys = pickle.load(f)
with open(data_dir + 'ds' + data_file, 'rb') as f:
    ds = pickle.load(f)
n_data, seq_len, n = ys.shape
output_dim = ds.shape[-1]
input_dim = n
p = output_dim//n

# ys and ds are inputs and outputs

class Net(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, writer_dir):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.counter = 0
        self.s_epoch = 1
        self.writer = SummaryWriter("./run_outputs/regression_board/" + writer_dir)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc1(x))
        # If the size is a square you can only specify a single number
        x = F.relu(self.fc2(x))
        return x


def train(model, batch_size, learn_rate=.001, EPOCHS=5, counter_write = 50, ytrain = ys, dtrain = ds,
          yval = None, dval = None):
    validate = (yval != None)
    # Defining loss function and optimizer
    n_train = ytrain.shape[0]
    if validate:
        n_val = yval.shape[0]
        assert n_train == dtrain.shape[0] and n_val == dval.shape[0],\
            'Sizes of inputs and outputs must match'
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    writer = model.writer
    model.train()
    print("Starting Training")
    epoch_times = []
    epoch_loss = []
    batch_loss = []
    total_val_loss = []
    # Start training loop
    s_epoch = model.s_epoch
    for epoch in range(s_epoch, EPOCHS + s_epoch):
        permutation = torch.randperm(n_train)
        start_time = time.clock()
        for i in range(0, n_train, batch_size):
            if i+batch_size > n_train:
                continue
            indices = permutation[i:i + batch_size]
            y_batch = ytrain[indices]
            d_batch = dtrain[indices]
            model.counter += 1
            model.zero_grad()
            if epoch ==1 and i==0:
                print("y", y_batch.shape, "d", d_batch.shape)
            out = model(y_batch.to(device).float())
            loss = criterion(out, d_batch.to(device).float())
            loss.backward()
            optimizer.step()
            if model.counter % counter_write == 0:
                batch_loss.append(loss)
                writer.add_scalar('Loss/Train/Batch', loss, model.counter)
#                print("Epoch {}...Step: {}... Batch Loss: {}".format(epoch, model.counter, loss))
        current_time = time.clock()
        train_out = model(ytrain.to(device).float())
        train_loss = criterion(train_out, dtrain.to(device).float())
        writer.add_scalar('Loss/Train/Total', train_loss, epoch)
        print("Epoch {}/{} Done, Total Loss: {:.5f}".format(epoch, EPOCHS, train_loss))
        epoch_loss.append(train_loss)
        if validate:
            with torch.no_grad():
                val_out = model(yval.to(device).float())
                val_loss = criterion(val_out, dval.to(device).float()).item()
                writer.add_scalar('Loss/Val/Total', val_loss, epoch)
                total_val_loss.append(val_loss)
                print("Epoch {}/{} Done, Val Loss: {:.5f}".format(epoch, EPOCHS, val_loss))

        print("Total Time Elapsed: {:.3f} seconds".format(current_time - start_time))
        epoch_times.append(current_time - start_time)
        writer.add_scalar('Time_per_epoch', current_time - start_time, epoch)
        model.s_epoch += 1
    print("Total Training Time: {:.3f} seconds".format(sum(epoch_times)))
    writer.close()
    return epoch_loss, epoch_times, batch_loss, total_val_loss

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#b20 l27.7 b10 l26.7 b5 l26.2 with 2 layers: b10 l20.3
hidden_dim = 32
num_layers = 2

batch_size = 16
lr = .005
EPOCHS = 40
drop = 0
counter_write = 30

val_prop = .15
test_prop = .0

i_test = int(test_prop*n_data)
i_val = int((val_prop+test_prop)*n_data)

permutation = torch.randperm(n_data)
test_ind = permutation[:i_test]
val_ind = permutation[i_test:i_val]
train_ind = permutation[i_val:]

ytest, dtest = ys[test_ind], ds[test_ind]
yval, dval = ys[val_ind], ds[val_ind]
ytrain, dtrain = ys[train_ind], ds[train_ind]
ntrain, nval, ntest = ytrain.shape[0], yval.shape[0], ytest.shape[0]

writer = "LR{} B{} L{} H{} T{}".format(lr, batch_size, num_layers, hidden_dim,
                                           int(time.time()) % 10000)

writers = ["MN"+str(i)+" "+writer for i in range(1, seq_len+1)]

nets = [Net(input_dim*i, output_dim, hidden_dim, writer_dir)
        for i, writer_dir in zip(range(1, seq_len+1), writers)]

epoch_loss, total_val_loss = [], []
for i, model in enumerate(nets):
    print("TRAINING ########", i)
    ytrain_loc = ytrain[:,:(i+1),].reshape(ntrain, -1)
    dtrain_loc = dtrain[:,i,]
    yval_loc = yval[:,:(i+1),].reshape(nval, -1)
    dval_loc = dval[:,i,]
    if test_prop != 0:
        ytest_loc = ytest[:,:(i+1),]
        dtest_loc = dtest[:,[i],]
    loc_epoch_loss, _, _, loc_total_val_loss = \
        train(model, batch_size, lr, EPOCHS, counter_write,
              ytrain_loc, dtrain_loc, yval_loc, dval_loc)
    epoch_loss.append(loc_epoch_loss)
    total_val_loss.append(loc_total_val_loss)
    if test_prop != 0:
        pass

#
# for num_experiment, (hidden_dim, num_layers, batch_size, lr) in enumerate(list(
#         itertools.product(hidden_list, layers_list, batch_list,lr_list))):
#     print("NUM EXPERIMENT ####################################", num_experiment)
#     writer_dir = "LR{} B{} L{} H{} T{}".format(lr, batch_size, num_layers, hidden_dim,
#                                                int(time.time()) % 10000)
#     model = GRUNet(input_dim, output_dim, hidden_dim, num_layers, drop, writer_dir)
#
#     epoch_loss, epoch_times, batch_loss, total_val_loss =\
#         train(model, batch_size, lr, EPOCHS, counter_write, ytrain, dtrain, yval, dval)
#
#     print(hidden_dim, num_layers, batch_size, lr)
#     if num_experiment > 3:
#         break




