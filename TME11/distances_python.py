import math

from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

def distances(x, y, p: int = 2, eps: float = 1e-16):
    ##  TODO:  Implémenter en utilisant torch.nn.functional.pairwise_distance
    pdist=F.pairwise_distance(p, eps)
    
    return pdist(x, y)
