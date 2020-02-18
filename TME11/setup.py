from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='distances',
    ext_modules=[
        CUDAExtension('distances_cuda', [
            'cuda/distances_cuda.cpp',
            'cuda/distances_cuda_kernel.cu',
            'cpp/distances.cpp'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
