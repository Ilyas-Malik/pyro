import os
import time
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn as nn
import pickle

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm_notebook

# Define data root directory
#name = "custom prior less steps" # To change to match data
name = "normalisedfinall"
data_dir = "./run_outputs/regression_rollout/"
data_file = name + ".result_stream.pickle"

#### Creating the Dataset

with open(data_dir + 'ys' + data_file, 'rb') as f:
    ys = pickle.load(f)
with open(data_dir + 'ds' + data_file, 'rb') as f:
    ds = pickle.load(f)
ds = (ds / ds.norm(dim=-1, p=1, keepdim=True)).expand(ds.shape)
n_data, seq_len, n = ys.shape
output_dim = ds.shape[-1]
p = output_dim//n
dtemp = ds[:,:(seq_len-1),:]
ytemp = ys[:,:(seq_len-1),:]
in_yds = torch.cat((ytemp, dtemp), 2)
out_ds = ds[:,1:,:]
# ys and ds are inputs and outputs

class GRUNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, drop_prob=.0,
                 writer_dir = f"{int(time.time())}"):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.counter = 0
        self.s_epoch = 1
        self.writer = SummaryWriter("./run_outputs/regression_board/" + writer_dir)
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

    def forward(self, x, h):
        assert len(x.size()) == 3, '[GRU]: Input dimension must be of length 3 i.e. [MxSxN]' # M: Batch Size(if batch first), S: Seq Lenght, N: Number of features
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out))
        return out, h


def train(model, batch_size, learn_rate=.001, EPOCHS=5, counter_write = 50, ytrain = ys, dtrain = ds,
          yval = None, dval = None, loss_ind_train = None, loss_ind_val = None):
    validate = (yval != None)
    # Defining loss function and optimizer
    n_train = ytrain.shape[0]
    if validate:
        if loss_ind_val != None:
            dval = dval[:, loss_ind_val, :]
        n_val = yval.shape[0]
        assert n_train == dtrain.shape[0] and n_val == dval.shape[0],\
            'Sizes of inputs and outputs must match'
    if loss_ind_train != None:
        dtrain = dtrain[:,loss_ind_train,:]
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
            h = model.init_hidden(batch_size)
            indices = permutation[i:i + batch_size]
            y_batch = ytrain[indices]
            d_batch = dtrain[indices]
            model.counter += 1
            model.zero_grad()
            out, _ = model(y_batch.to(device).float(), h)
            if loss_ind_train != None:
                out = out[:,loss_ind_train,:]
            loss = criterion(out, d_batch.to(device).float())
            loss.backward()
            optimizer.step()
            if model.counter % counter_write == 0:
                batch_loss.append(loss)
                writer.add_scalar('Loss/Train/Batch', loss, model.counter)
#                print("Epoch {}...Step: {}... Batch Loss: {}".format(epoch, model.counter, loss))
        current_time = time.clock()
        train_h = model.init_hidden(n_train)
        train_out, _ = model(ytrain.to(device).float(), train_h)
        train_loss = criterion(train_out, dtrain.to(device).float())
        writer.add_scalar('Loss/Train/Total', train_loss, epoch)
        print("Epoch {}/{} Done, Total Loss: {:.5f}".format(epoch, EPOCHS, train_loss))
        epoch_loss.append(train_loss)
        if validate:
            with torch.no_grad():
                val_h = model.init_hidden(n_val)
                val_out, _ = model(yval.to(device).float(), val_h)
                if loss_ind_val != None:
                    val_out = val_out[:,loss_ind_val,:]
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


def evaluate(model, ytest, dtest):
    model.eval()
    criterion = nn.MSELoss()
    n = ytest.shape[0]
    h = model.init_hidden(n)
    out, _ = model(ytest.to(device).float(), h)
    loss = criterion(out, dtest.to(device).float()).item()
    print("Loss: {}%".format(loss))
    return loss


is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#b20 l27.7 b10 l26.7 b5 l26.2 with 2 layers: b10 l20.3
hidden_dim = 32
num_layers = 1

batch_size = 16
lr = .01
EPOCHS = 20
drop = 0
counter_write = 30
input_dim = n + n*p     # was just n

val_prop = .15
test_prop = .0

hidden_list = [16, 64, 256]
layers_list = [1,2]
batch_list = [8, 16]
lr_list = [.0001*5**i for i in range(4)]

i_test = int(test_prop*n_data)
i_val = int((val_prop+test_prop)*n_data)


permutation = torch.randperm(n_data)
test_ind = permutation[:i_test]
val_ind = permutation[i_test:i_val]
train_ind = permutation[i_val:]


in_test, out_test = in_yds[test_ind], out_ds[test_ind]
in_val, out_val = in_yds[val_ind], out_ds[val_ind]
in_train, out_train = in_yds[train_ind], out_ds[train_ind]

######## Monitoring the loss at each time step by training the whole GRU

# for i in range(seq_len):
#     torch.manual_seed(672)
#     writer_dir = "GRU{} LR{} B{} L{} H{} T{}".format(i+1, lr, batch_size, num_layers, hidden_dim,
#                                                int(time.time()) % 10000)
#     model = GRUNet(input_dim, output_dim, hidden_dim, num_layers, drop, writer_dir)
#     epoch_loss, epoch_times, batch_loss, total_val_loss = \
#         train(model, batch_size, lr, EPOCHS, counter_write, ytrain, dtrain, yval, dval,
#               loss_ind_train=None, loss_ind_val=i)



######## Training the GRU on the optimal hyperparameters

writer_dir = "LR{} B{} L{} H{} T{} design_outcome".format(lr, batch_size, num_layers, hidden_dim,
                                           int(time.time()) % 10000)
model = GRUNet(input_dim, output_dim, hidden_dim, num_layers, drop, writer_dir)
epoch_loss, epoch_times, batch_loss, total_val_loss = \
    train(model, batch_size, lr, EPOCHS, counter_write, in_train, out_train, in_val, out_val,
          loss_ind_train=None, loss_ind_val=None)


######## Training on a multidimensional hyperparameter grid

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





