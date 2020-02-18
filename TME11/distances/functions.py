import math
from torch import nn
from torch.autograd import Function
import torch

class DistancesFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2, p=2, eps=1e-6):
        if input1.is_cuda and input2.is_cuda:
            from .cuda import distances_cuda
            output = distances_cuda.forward(input1, input2, p, eps)
        else:
            from .cpp import distances_cpp
            output = distances_cpp.forward(input1, input2, p, eps)

        ctx.save_for_backward(input1, input2, output, torch.tensor(p), torch.tensor(eps))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, output, p, eps = ctx.saved_tensors
        p = float(p)
        eps = float(eps)

        if input1.is_cuda and input2.is_cuda:
            from .cuda import distances_cuda
            outputs = distances_cuda.backward(grad_output, output, input1, input2, p, eps)
        else:
            from .cpp import distances_cpp
            outputs = distances_cpp.backward(grad_output, output, input1, input2, p, eps)

        return outputs[0], outputs[1], None, None


distances = DistancesFunction.apply
