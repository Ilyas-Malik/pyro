import os
import torch
import pickle

#name = "“eigtest”"
name = "lastest"
name = "n1p2"

output_dir = "./run_outputs/regression_rollout/"+name

with open(output_dir + "xi1", 'rb') as f:
    xi1 = pickle.load(f)
with open(output_dir + "xi2", 'rb') as f:
    xi2 = pickle.load(f)
with open(output_dir + "y1", 'rb') as f:
    y1 = pickle.load(f)
with open(output_dir + "y2", 'rb') as f:
    y2 = pickle.load(f)
with open(output_dir + "loss", 'rb') as f:
    est_loss_history = pickle.load(f)



