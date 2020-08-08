# Fix prior and model
#
# find optimal design for step 1 depends on prior of model only
#     return xi_1 # xi_1 randomly chosen from the 4 designs (1,0) etc...
#
# call true model with xi_1
#     return y_1
#
# OLD METHOD
#     computes posterior p(theta|y_1, xi_1)
#     replace prior with posterior and loop
#
# NEW METHOD
#     train net
#     call net(xi_1, y_1) return xi_2
#     compute objective function for xi_2 (for expemple PCE bound)


import torch
from torch.distributions import transform_to
import argparse
import subprocess
import datetime
import random
import pickle
import time
import os
from functools import partial
from contextlib import ExitStack
import logging
import os
import time
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F


import pyro
from pyro import poutine
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape, rexpand, rmv
from pyro.contrib.oed.eig import marginal_eig, elbo_learn, nmc_eig, pce_eig
import pyro.contrib.gp as gp
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss, differentiable_pce_eig, _differentiable_ace_eig_loss
from pyro.contrib.oed.eig import opt_eig_ape_loss
from pyro.util import is_bad

from ces_gradients import PosteriorGuide, LinearPosteriorGuide


# TODO read from torch float spec
epsilon = torch.tensor(2**-22)

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


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

#Creates model with predefined fixed theta (w and sigma), outputs y as a regression outcome
#same
def make_regression_model(w_loc, w_scale, sigma_scale, observation_label="y"):
    def regression_model(design):
        design = (design / design.norm(dim=-1, p=1, keepdim=True)).expand(design.shape)
        if is_bad(design):
            raise ArithmeticError("bad design, contains nan or inf")
        batch_shape = design.shape[:-2]
        with pyro.plate_stack("plate_stack", batch_shape):
            # `w` is shape p, the prior on each component is independent
            w = pyro.sample("w", dist.Laplace(w_loc, w_scale).to_event(1))
            # `sigma` is scalar
            sigma = 1e-6 + pyro.sample("sigma", dist.Exponential(sigma_scale)).unsqueeze(-1)
            mean = rmv(design, w)
            sd = sigma
            y = pyro.sample(observation_label, dist.Normal(mean, sd).to_event(1))
            return y, w, sigma

    return regression_model

# just x_2 will be output of the net

def make_learn_xi_model(model):
    def model_learn_xi(design_prototype):
        design = pyro.param("xi")
        design = design.expand(design_prototype.shape)
        return model(design)
    return model_learn_xi


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))
    return new_loss

# HELP what is loglevel, num_acquisition etc
# Creates rollout with initial fixed parameters, true values of theta fixed ?
# HELP what does function do, numsteps, num_parallel ?


def main(experiment_name, seed, num_samples, net, lr,
         num_contrast_samples, loglevel, n, p, scale, batch_size, train_steps):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(loglevel))
    logging.basicConfig(level=numeric_level)

    output_dir = "./run_outputs/regression_rollout/"
    if not experiment_name:
        experiment_name = output_dir+"{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir+experiment_name
    results_file = experiment_name + '.result_stream.pickle'
    try:
        os.remove(results_file)
    except OSError:
        logging.info("File {} does not exist yet".format(results_file))
    pyro.clear_param_store()
    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = int(torch.rand(tuple()) * 2**30)
        pyro.set_rng_seed(seed)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        # Change the prior distribution here
        # prior params
        w_loc = torch.zeros(p)
        w_scale = scale * torch.ones(p)
        sigma_scale = scale * torch.tensor(1.)

        true_w_cov = torch.tensor([[1., .2],
                                   [.2, 2.]])
        true_w_loc = torch.tensor([.5, -.5])
        true_w = torch.distributions.multivariate_normal.MultivariateNormal(true_w_loc, true_w_cov).sample()
        true_sigma_scale = .9
        true_sigma = torch.distributions.exponential.Exponential(true_sigma_scale).sample()
        true_model = pyro.condition(make_regression_model(w_loc, w_scale, sigma_scale),
                                    {"w": true_w, "sigma": true_sigma})

        contrastive_samples = num_samples
        targets = ["w", "sigma"]

        d_star_designs = torch.tensor([])
        ys = torch.tensor([])
        net.train()
        results = {'step': [], 'git-hash': get_git_revision_hash(), 'seed': seed, 'num_samples': num_samples,
                   'num_contrast_samples': num_contrast_samples, 'design_time': [], 'design': [],
                   'y': [], 'w': [], 'sigma':[], 'eig': []}
        model = make_regression_model(w_loc, w_scale, sigma_scale)
        for step in range(train_steps):
            # randomly sample xi_1 from optimal 4 options, sample y_1 from true model
            ind = random.choices(range(4), k=batch_size)
            xi_1_opt = [torch.tensor([0.,1.]), torch.tensor([1.,0.]), torch.tensor([0.,-1.]), torch.tensor([-1.,0.])]
            xi_1 = torch.stack([xi_1_opt[_] for _ in ind]).reshape((batch_size,n,p))
            res_1 = model(xi_1)
            y_1 = res_1[0]
            results['step'].append(step)

    # put net here xi_1 and y_1, design = net(xi1,y1)

    # no initialization of x_1, use net, evaluate the loss function,
    # scalar_loss = pce_eig(model = model, design=net(xi_1,y_1), else is the same),,,,, pce_eig(
            # #                        model=model_learn_xi, design=d, observation_labels=["y"], target_labels=targets,
            # #                        N=N, M=contrastive_samples, **kwargs)
            scalar_loss.backward()
            optimizer.step()
            elapsed = time.time() - t
            logging.info('elapsed design time {}'.format(elapsed))
            results['design_time'].append(elapsed)
            results['design'].append(design)
            d_star_designs = torch.cat([d_star_designs, d_star_design], dim=-2)

            ys = torch.cat([ys, y], dim=-1)
            logging.info('ys {} {}'.format(ys.squeeze(), ys.shape))
            results['y'].append(y)
            results['w'].append(w_loc)
            results['sigma'].append(sigma_scale)

        with open(results_file, 'ab') as f:
            pickle.dump(results, f)

n=1
p=2
design_dim = n*p
batch_size = 10
train_steps = 100
hidden_dim = 32
input_dim = n+design_dim
output_dim = design_dim
lr = .001
num_layers = 1
writer_dir = "LR{} B{} L{} H{} T{} end_to_end".format(lr, batch_size, num_layers, hidden_dim,
                                           int(time.time()) % 10000)
net = Net(input_dim, output_dim, hidden_dim, writer_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression rollouts"
                                                 " iterated experiment design")
    parser.add_argument("--num-steps", nargs="?", default=2, type=int) #num iterations
    parser.add_argument("--num-parallel", nargs="?", default=2, type=int) #batch size
    parser.add_argument("--name", nargs="?", default="", type=str)
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    parser.add_argument("--loglevel", default="info", type=str)
    parser.add_argument("--num-gradient-steps", default=1000, type=int) #gradient for convergence of svi to have good variational parameters and
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--num-contrast-samples", default=10, type=int)
    parser.add_argument("-n", default=1, type=int)
    parser.add_argument("-p", default=2, type=int)
    parser.add_argument("--scale", default=1., type=float)
    parser.add_argument("--num-data", default=10, type=int)
    parser.add_argument("--train-steps", default=30, type=int)
    parser.add_argument("--minibatch", default=10, type=int)
    args = parser.parse_args()
    if args.num_data != 1:
        message = str(args).replace(", ", "\n")
        output_msg = "./run_outputs/regression_rollout/" + args.name
        f = open(output_msg+".txt", "w+")
        f.write(message)
        f.close()
        for i in range(args.num_data):
            main(args.num_steps, args.num_parallel, str(i)+args.name, args.typs, args.seed,
                 args.num_gradient_steps, args.num_samples, args.num_contrast_samples,
                 args.loglevel, args.n, args.p, args.scale)
    else:
        main(args.num_steps, args.num_parallel, args.name, args.typs, args.seed,
             args.num_gradient_steps, args.num_samples, args.num_contrast_samples,
             args.loglevel, args.n, args.p, args.scale)
