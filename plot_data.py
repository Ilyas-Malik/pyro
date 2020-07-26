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
