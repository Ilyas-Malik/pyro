import argparse
import datetime
import math
import pickle
import subprocess
import time

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro import poutine
from pyro.contrib.oed.eig import _eig_from_ape, pce_eig, _ace_eig_loss, _posterior_loss
from pyro.contrib.util import rmv
from pyro.util import is_bad


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def make_regression_model(w_loc, w_scale, sigma_scale, design_net, observation_prefix="y"):
    def regression_model(design_prototype):
        batch_shape = design_prototype.shape[:-2]
        with pyro.plate_stack("plate_stack", batch_shape):
            ###################################################################################################
            # Get xi1
            ###################################################################################################
            xi1 = torch.tensor([1, 0]).to(design_prototype.device).expand(design_prototype.shape)

            ###################################################################################################
            # Sample theta
            ###################################################################################################
            # `w` is shape p, the prior on each component is independent
            w = pyro.sample("w", dist.Laplace(w_loc, w_scale).to_event(1))
            # `sigma` is scalar
            sigma = 1e-6 + pyro.sample("sigma", dist.Exponential(sigma_scale)).unsqueeze(-1)

            ###################################################################################################
            # Sample y1
            ###################################################################################################
            mean1 = rmv(xi1, w)
            sd = sigma
            y1 = pyro.sample(observation_prefix + '1', dist.Normal(mean1, sd).to_event(1))

            ###################################################################################################
            # Get xi2
            ###################################################################################################
            xi2 = design_net(y1, xi1)

            ###################################################################################################
            # Sample y2
            ###################################################################################################
            mean2 = rmv(xi2, w)
            sd = sigma
            y2 = pyro.sample(observation_prefix + '2', dist.Normal(mean2, sd).to_event(1))

            return y1, y2

    return regression_model


class TensorLinear(nn.Module):
    __constants__ = ['bias']

    def __init__(self, *shape, bias=True):
        super(TensorLinear, self).__init__()
        self.in_features = shape[-2]
        self.out_features = shape[-1]
        self.batch_dims = shape[:-2]
        self.weight = nn.Parameter(torch.Tensor(*self.batch_dims, self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(*self.batch_dims, self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return rmv(self.weight, input) + self.bias


class DesignNetwork:

    def __init__(self, y_dim, xi_dim, batching):
        super(DesignNetwork, self).__init__()
        n_hidden = 128
        self.linear1 = TensorLinear(*batching, y_dim + xi_dim, n_hidden)
        # self.linear2 = TensorLinear(*batching, n_hidden, n_hidden)
        self.output_layer = TensorLinear(*batching, n_hidden, xi_dim)
        self.relu = nn.ReLU()

    # Maps (y1, xi1) to xi2
    def forward(self, y1, xi1):
        inputs = torch.cat([y1, xi1], dim=-1)
        x = self.linear1(inputs)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))

    return new_loss


def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim, time_budget):
    params = None
    est_loss_history = []
    xi_history = []
    baseline = 0.
    t = time.time()
    wall_times = []
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = loss_fn(design, num_samples, evaluation=True, control_variate=baseline)
        baseline = -loss.detach()
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(agg_loss):
            raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
        agg_loss.backward(retain_graph=True)
        est_loss_history.append(loss.detach())
        wall_times.append(time.time() - t)
        optim(params)
        optim.step()
        print(pyro.param("xi")[0, 0, ...])
        print(step)
        print('eig', baseline.squeeze())
        if time_budget and time.time() - t > time_budget:
            break

    xi_history.append(pyro.param('xi').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)
    wall_times = torch.tensor(wall_times)

    return xi_history, est_loss_history, wall_times


def main(num_steps, num_samples, time_budget, experiment_name, estimators, seed, num_parallel, start_lr, end_lr,
         device, n, p, scale):
    output_dir = "./run_outputs/gradinfo/"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.pickle'
    estimators = estimators.split(",")

    for estimator in estimators:
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed)
        else:
            seed = int(torch.rand(tuple()) * 2 ** 30)
            pyro.set_rng_seed(seed)

        # xi_init = torch.randn((num_parallel, n, p), device=device)
        # Change the prior distribution here
        # prior params
        w_prior_loc = torch.zeros(p, device=device)
        w_prior_scale = scale * torch.ones(p, device=device)
        sigma_prior_scale = scale * torch.tensor(1., device=device)

        design_net = DesignNetwork(n, p, (num_parallel,)).to(device)

        model_learn_net = make_regression_model(
            w_prior_loc, w_prior_scale, sigma_prior_scale, design_net)

        contrastive_samples = num_samples

        # Fix correct loss
        targets = ["w", "sigma"]

        if estimator == 'pce':
            eig_loss = lambda d, N, **kwargs: pce_eig(
                model=model_learn_net, design=d, observation_labels=["y1", "y2"], target_labels=targets,
                N=N, M=contrastive_samples, **kwargs)
            loss = neg_loss(eig_loss)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr / start_lr) ** (1 / num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(num_parallel, n, p, device=device)  # this is annoying, code needs refactor

        xi_history, est_loss_history, wall_times = opt_eig_loss_w_history(
            design_prototype, loss, num_samples=num_samples, num_steps=num_steps, optim=scheduler,
            time_budget=time_budget)



        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history.cpu(),
                   'wall_times': wall_times.cpu()}

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=500000, type=int)
    parser.add_argument("--time-budget", default=1200, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.001, type=float)
    parser.add_argument("--end-lr", default=0.001, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("-n", default=20, type=int)
    parser.add_argument("-p", default=20, type=int)
    parser.add_argument("--scale", default=1., type=float)
    args = parser.parse_args()
    main(args.num_steps, args.num_samples, args.time_budget, args.name, args.estimator, args.seed, args.num_parallel,
         args.start_lr, args.end_lr, args.device, args.n, args.p, args.scale)
