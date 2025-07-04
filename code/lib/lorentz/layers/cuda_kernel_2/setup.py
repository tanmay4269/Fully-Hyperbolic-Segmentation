from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_lorentz_conv2d',
    ext_modules=[
        CUDAExtension(
            name='fused_lorentz_conv2d',
            sources=['fused_lorentz_conv2d.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_86',  # Adjust based on GPU
                    '--use_fast_math',
                    '-lineinfo'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)