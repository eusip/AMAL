from pathlib import Path
from torch.utils.cpp_extension import load

distances_cpp = load(name="distances_cpp", sources=[ Path(__file__).parent / "distances.cpp"], verbose=True)
