from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rearrange_cuda',
    ext_modules=[
        CUDAExtension(
            name='rearrange_cuda',
            sources=['rearrange_tensors.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

