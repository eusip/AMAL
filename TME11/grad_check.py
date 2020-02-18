from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.autograd import gradcheck

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

if options.mode == 'py':
    from python.distances_baseline import DistancesFunction
elif options.mode == 'cpp':
    from cpp.distances import DistancesFunction
else:
    from cuda.distances import DistancesFunction
    options.cuda = True

device = torch.device("cuda") if options.cuda else torch.device("cpu")

kwargs = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}

X = torch.randn(options.batch_size, options.features, **kwargs)
h = torch.randn(options.batch_size, options.state_size, **kwargs)
C = torch.randn(options.batch_size, options.state_size, **kwargs)
W = torch.randn(3 * options.state_size, options.features + options.state_size, **kwargs)
b = torch.randn(1, 3 * options.state_size, **kwargs)

variables = [X, W, b, h, C]


if gradcheck(DistancesFunction.apply, variables):
    print('Ok')
