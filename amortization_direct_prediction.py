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
name = "custom prior less steps" # To change to match data
data_chunks = 745 # Number of chunks of data, to change, put as arg at the end

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
            loss = criterion(out, d_batch.to(device).float())
            loss.backward()
            optimizer.step()
            if model.counter % counter_write == 0:
                batch_loss.append(loss)
                writer.add_scalar('Loss/Train/Batch', loss, model.counter)
                print("Epoch {}...Step: {}... Batch Loss: {}".format(epoch, model.counter, loss))
        current_time = time.clock()
        train_h = model.init_hidden(n_train)
        train_out, _ = model(ytrain.to(device).float(), train_h)
        train_loss = criterion(train_out, dtrain.to(device).float())
        writer.add_scalar('Loss/Train/Total', train_loss, epoch)
        print("Epoch {}/{} Done, Total Loss: {:.5f}".format(epoch, EPOCHS, train_loss))
        epoch_loss.append(train_loss)
        if validate:
            val_h = model.init_hidden(n_val)
            val_out, _ = model(yval.to(device).float(), val_h)
            val_loss = criterion(val_out, dval.to(device).float())
            writer.add_scalar('Loss/Val/Total', val_loss, epoch)
        print("Total Time Elapsed: {:.3f} seconds".format(current_time - start_time))
        epoch_times.append(current_time - start_time)
        writer.add_scalar('Time_per_epoch', current_time - start_time, epoch)
        model.s_epoch += 1
    print("Total Training Time: {:.3f} seconds".format(sum(epoch_times)))
    writer.close()
    return epoch_loss, epoch_times, batch_loss


def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.clock() - start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)
    print("sMAPE: {}%".format(sMAPE * 100))
    return outputs, targets, sMAPE


is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#b20 l27.7 b10 l26.7 b5 l26.2 with 2 layers: b10 l20.3

hidden_dim = 128
num_layers = 1

batch_size = 50
lr = .003
EPOCHS = 10
drop = 0
counter_write = 30

hidden_list = [2**i for i in range(4,10)]
layers_list = [1,2]
batch_list = [2**i for i in range(1,6)]
lr_list = [.0001*4**i for i in range(5)]


for hidden_dim, num_layers, batch_size, lr in list(
        itertools.product(hidden_list, layers_list, batch_list,lr_list)):
    writer_dir = "LR{} B{} L{} H{} T{}".format(lr, batch_size, num_layers, hidden_dim,
                                               int(time.time()) % 10000)
    model = GRUNet(input_dim, output_dim, hidden_dim, num_layers, drop, writer_dir)
    epoch_loss, epoch_times, batch_loss = train(model, batch_size, lr, EPOCHS, counter_write)
    print(hidden_dim, num_layers, batch_size, lr)








