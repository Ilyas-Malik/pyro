import os
import time
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd

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
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=.0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.writer = SummaryWriter(f"./run_outputs/regression_board/{int(time.time())}")
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

    def forward(self, x, h):
        assert len(x.size()) == 3, '[GRU]: Input dimension must be of length 3 i.e. [MxSxN]' # M: Batch Size(if batch first), S: Seq Lenght, N: Number of features
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out))
        return out, h


def train(model, batch_size, learn_rate=.001, EPOCHS=5):

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training")
    epoch_times = []
    epoch_loss = []
    batch_loss = []
    # Start training loop
    counter = 0
    for epoch in range(1, EPOCHS + 1):
        permutation = torch.randperm(n_data)
        start_time = time.clock()
        for i in range(0, n_data, batch_size):
            if i+batch_size > n_data:
                continue
            h = model.init_hidden(batch_size)
            indices = permutation[i:i + batch_size]
            y_batch = ys[indices]
            d_batch = ds[indices]
            counter += 1
            model.zero_grad()
            out, _ = model(y_batch.to(device).float(), h)
            loss = criterion(out, d_batch.to(device).float())
            loss.backward()
            optimizer.step()
            if counter % 50 == 0:
                batch_loss.append(loss)
                print("Epoch {}...Step: {}... Batch Loss: {}".format(epoch, counter, loss))
        current_time = time.clock()
        total_h = model.init_hidden(n_data)
        total_out, _ = model(ys.to(device).float(), total_h)
        total_loss = criterion(total_out, ds.to(device).float())
        print("Epoch {}/{} Done, Total Loss: {:.5f}".format(epoch, EPOCHS, total_loss))
        print("Total Time Elapsed: {:.3f} seconds".format(current_time - start_time))
        epoch_loss.append(total_loss)
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {:.3f} seconds".format(sum(epoch_times)))
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
batch_size = 10
lr = .001
EPOCHS = 20

model = GRUNet(input_dim, 256, output_dim, 2, 0)
print(next(model.parameters()))
epoch_loss, epoch_times, batch_loss = train(model, batch_size, lr, EPOCHS)
print(next(model.parameters()))




# batch_size = 10
# input_dim = 4
# seq_len = 8
# hidden_dim = 6
# n_layers = 1
# print('f')
# x = torch.tensor(np.random.normal(0, 1, seq_len * batch_size * input_dim))
# gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0)


