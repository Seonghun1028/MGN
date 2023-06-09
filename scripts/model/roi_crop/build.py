# from __future__ import print_function
# import os
# import torch
# from torch.utils.ffi import create_extension

# #this_file = os.path.dirname(__file__)

# sources = ['src/roi_crop.c']
# headers = ['src/roi_crop.h']
# defines = []
# with_cuda = False

# if torch.cuda.is_available():
#     print('Including CUDA code.')
#     sources += ['src/roi_crop_cuda.c']
#     headers += ['src/roi_crop_cuda.h']
#     defines += [('WITH_CUDA', None)]
#     with_cuda = True

# this_file = os.path.dirname(os.path.realpath(__file__))
# print(this_file)
# extra_objects = ['src/roi_crop_cuda_kernel.cu.o']
# extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

# ffi = create_extension(
#     '_ext.roi_crop',
#     headers=headers,
#     sources=sources,
#     define_macros=defines,
#     relative_to=__file__,
#     with_cuda=with_cuda,
#     extra_objects=extra_objects
# )

# if __name__ == '__main__':
#     ffi.build()

from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
import torch
sources = ['src/roi_crop.cpp']
headers = ['src/roi_crop.h']
defines = []
extra_objects = []

with_cuda = False



if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi_crop_cuda.cpp']
    headers += ['src/roi_crop_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

    this_file = os.path.dirname(os.path.realpath(__file__))
    print(this_file)
    extra_objects = ['src/roi_crop_cuda_kernel.cu.o']
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
    print(extra_objects)

nms_module = cpp_extension.CUDAExtension(
    'extroicrop',
    headers=headers,
    sources=sources,
    define_macros=defines,
    extra_objects=extra_objects,
    extra_compile_args={'cxx': ['-g', '-std=c++14'], 'nvcc': ['-O2']}
)

setup(
    name='extroicrop',
    ext_modules=[nms_module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)