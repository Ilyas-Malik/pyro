import os

import torch
import pickle
import argparse

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm_notebook

# Define data root directory

def main(name, data_chunks, remove):

    data_dir = "./run_outputs/regression_rollout/"
    data_file = name + ".result_stream.pickle"

    #### Creating the Dataset


    for i in range(data_chunks):
        datum = data_dir + str(i) + data_file
        with open(datum, 'rb') as f:
            res = pickle.load(f)
        if i == 0:
            ys = torch.stack(res["y"])
            ds = torch.stack(res["d_star_design"])
            seq_len = len(res['step'])
            continue
        assert len(res['step'])>=seq_len
        y = torch.stack(res["y"])
        d = torch.stack(res["d_star_design"])
        ys = torch.cat((ys,y), 1)
        ds = torch.cat((ds,d), 1)
    ys.transpose_(0,1)
    ds.transpose_(0,1)
    n_data, seq_len, n, p = ds.shape
    ys = ys.reshape(n_data, seq_len, -1)
    ds = ds.reshape(n_data, seq_len, -1)
    input_dim = n
    output_dim = n*p
    # ys and ds are inputs and outputs

    with open(data_dir + 'ys' + data_file, 'ab') as f:
        pickle.dump(ys, f)
    with open(data_dir + 'ds' + data_file, 'ab') as f:
        pickle.dump(ds, f)

    if remove:
        for i in range(data_chunks):
            datum = data_dir + str(i) + data_file
            os.remove(datum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression rollouts"
                                                 " iterated experiment design")
    parser.add_argument("--name", nargs="?", default="", type=str)
    parser.add_argument("--data-chunks", nargs="?", default=-1, type=int)
    parser.add_argument("--remove", nargs="?", default=False, type=bool)
    args = parser.parse_args()
    main(args.name, args.data_chunks, args.remove)





