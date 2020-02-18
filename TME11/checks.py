import argparse
import math
import time
import sys
import numpy as np
import torch
from torch.autograd import gradcheck

TIME_SCALES = {'s': 1, 'ms': 1000, 'µs': 1000000}

def get_inputs(dtype, device, options):
    kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
    input1 = torch.randn(options.x1_size, options.state_size, **kwargs)
    input2 = torch.randn(options.x2_size, options.state_size, **kwargs)
    return input1, input2, options.norm, options.eps

def benchmark(device, dtype, distances, options):
    variables = get_inputs(dtype, device, options)

    # Warm-up / compilation
    distances(*variables)

    forward_min = math.inf
    forward_time = 0
    backward_min = math.inf
    backward_time = 0

    for _ in range(options.runs):
        variables[0].grad = None
        variables[1].grad = None

        start = time.time()
        d = distances(*variables)
        elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed

        start = time.time()
        d.sum().backward()
        elapsed = time.time() - start
        backward_min = min(backward_min, elapsed)
        backward_time += elapsed

    scale = TIME_SCALES[options.scale]
    forward_min *= scale
    backward_min *= scale
    forward_average = forward_time / options.runs * scale
    backward_average = backward_time / options.runs * scale

    print('Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
        forward_min, forward_average, backward_min, backward_average,
        options.scale))

def check_gradient(device, dtype, distances, options):
    variables = get_inputs(dtype, device, options)
    if gradcheck(distances, variables):
        print('Ok')


def check_equal(first, second, verbose):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)

        rtol = 1e-4 if x.dtype == np.float32 else 1e-6
        print("====", x.dtype, isinstance(x.dtype, np.float32), np.float64, np.float32, type(x.dtype))
        np.testing.assert_allclose(x, y, rtol=1e-7, err_msg="Index: {}".format(i))

def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()

def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_forward(baseline_fn, tested_fn, variables, verbose):
    baseline_values = baseline_fn(*variables)
    cpp_values = tested_fn(*variables)

    print('Forward: Baseline (Python) vs. C++/CUDA ... ', end='')
    check_equal(baseline_values, cpp_values, verbose)
    print('Ok')


def check_backward(baseline_fn, tested_fn, variables, verbose):
    value = baseline_fn(*variables)
    value.sum().backward()
    grad_baseline = get_grads(variables[:1])

    zero_grad(variables[:1])

    cpp_value = tested_fn(*variables)
    cpp_value.sum().backward()
    grad_cpp = get_grads(variables[:1])

    print('Backward: Baseline (Python) vs. C++/CUDA... ', end='')
    check_equal(grad_baseline, grad_cpp, verbose)
    print('Ok')


def check(device, dtype, distances, options):
    variables = get_inputs(dtype, device, options)

    from distances_python import distances as baseline_fn

    if 'forward' in options.direction:
        check_forward(baseline_fn, distances, variables, options.verbose)

    if 'backward' in options.direction:
        check_backward(baseline_fn, distances, variables, options.verbose)


parser = argparse.ArgumentParser()
parser.add_argument('-1', '--x1-size', type=int, default=100)
parser.add_argument('-2', '--x2-size', type=int, default=101)
parser.add_argument('-s', '--state-size', type=int, default=256)
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
parser.add_argument('-p', '--norm', type=int, default=2, help='Norm degree')
parser.add_argument('-e', '--eps', type=float, default=1e-6, help='Epsilon to avoid zero')
parser.add_argument('--verbose', action='store_true')

subparsers = parser.add_subparsers()
parser_benchmark = subparsers.add_parser('benchmark')
parser_benchmark.add_argument('-r', '--runs', type=int, default=100)
parser_benchmark.add_argument('--scale', choices=['s', 'ms', 'µs'], default='µs')
parser_benchmark.add_argument('mode', choices=['py', 'cpp', 'cuda'])
parser_benchmark.set_defaults(func=benchmark)

parser_gradcheck = subparsers.add_parser('gradcheck')
parser_gradcheck.add_argument('mode', choices=['py', 'cpp', 'cuda'])
parser_gradcheck.set_defaults(func=check_gradient)

parser_check = subparsers.add_parser('check')
parser_check.add_argument('mode', choices=['cpp', 'cuda'])
parser_check.set_defaults(func=check)
parser_check.add_argument('direction', choices=['forward', 'backward'], nargs='+')

options = parser.parse_args()


device = torch.device("cuda") if options.cuda or options.mode == "cuda" else torch.device("cpu")
dtype = torch.float64 if options.double or options.func == check_gradient else torch.float32

if not options.func:
    sys.exit(0)

if options.mode == "py":
    from distances_python import distances
else:
    from distances import distances

options.func(device, dtype, distances, options)

print("---- FINISHED ----")
