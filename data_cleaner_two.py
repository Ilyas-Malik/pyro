import os
import torch
import pickle
import argparse

#name = "“eigtest”"
name = "normalised3"
data_dir = "./run_outputs/regression_rollout/"
data_file = name + ".result_stream.pickle"
data_chunks = 3

for i in range(data_chunks):
    datum = data_dir + str(i) + data_file
    with open(datum, 'rb') as f:
        res = pickle.load(f)
    if i == 0:
        ys = torch.stack(res["y"])
        ds = torch.stack(res["d_star_design"])
        eigs = torch.stack(res["ape"])
        seq_len = len(res['step'])
        continue
    assert len(res['step']) >= seq_len
    y = torch.stack(res["y"])
    d = torch.stack(res["d_star_design"])
    eig = torch.stack(res["ape"])
    ys = torch.cat((ys, y), 1)
    ds = torch.cat((ds, d), 1)
    eigs = torch.cat((eigs, eig), 1)
ys.transpose_(0, 1)
ds.transpose_(0, 1)
n_data, seq_len, n, p = ds.shape
ys = ys.reshape(n_data, seq_len, -1)
ds = ds.reshape(n_data, seq_len, -1)
