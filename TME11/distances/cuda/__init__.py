from pathlib import Path
from torch.utils.cpp_extension import load

path = Path(__file__).parent
distances_cuda = load(
    'distances_cuda', [ path / 'distances_cuda.cpp', path / 'distances_cuda_kernel.cu'], verbose=True)
