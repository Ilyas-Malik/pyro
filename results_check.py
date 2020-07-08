import torch
from torch.distributions import transform_to
import argparse
import subprocess
import datetime
import pickle
import time
import os
from functools import partial
from contextlib import ExitStack
import logging

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape, rexpand, rmv
from pyro.contrib.oed.eig import marginal_eig, elbo_learn, nmc_eig, pce_eig
import pyro.contrib.gp as gp
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss, differentiable_pce_eig, _differentiable_ace_eig_loss
from pyro.contrib.oed.eig import opt_eig_ape_loss
from pyro.util import is_bad

output_dir = "./run_outputs/ces/"
name = "regression-rollout-pce"
experiment_name = output_dir+name
results_file = experiment_name + '.result_stream.pickle'
print("jfjikd")
with open(results_file, 'ab') as f:
    res = pickle.load(f)
print(res)
