import math
import torch
from distances.functions import distances

class Distances(torch.nn.Module):
    def __init__(self, p:int=2):
        super().__init__()
        self.p  = p

    def forward(self, input1, input2):
        return distances(input1, input2, self.p)
