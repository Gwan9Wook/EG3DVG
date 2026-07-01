# Copyright (c) 2024
# PointNet++ CUDA extension build script (for PyTorch 2.0+)
# ---------------------------------------------------------

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import glob
import os
import os.path as osp

print(f"[INFO] Building PointNet2 extensions with torch {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}, version: {torch.version.cuda}")

# Root directory of source
this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = "_ext_src"

# Source / header discovery
_ext_sources = glob.glob(f"{_ext_src_root}/src/*.cpp") + glob.glob(f"{_ext_src_root}/src/*.cu")
include_dir = osp.join(this_dir, _ext_src_root, "include")

# 🔹 Ensure CUDA architecture list (adjust to your GPU)
# e.g., 8.6: RTX 3090 / A10 / A6000, 8.9: RTX 4090, 9.0: H100
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6;8.9+PTX"
# os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0;9.0+PTX"

# 🔹 Compile flags
extra_cxx_flags = [
    "-O3",
    "-std=c++17",
    "-fPIC",
    "-Wno-sign-compare",
    "-Wno-unused-variable",
    "-Wno-deprecated-declarations",
]

extra_nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    f"-I{include_dir}",
]

# 🔹 Important: ensure ABI compatibility with current PyTorch build
abi_flag = torch._C._GLIBCXX_USE_CXX11_ABI
extra_cxx_flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={int(abi_flag)}")

# 🔹 Extension module definition
setup(
    name="pointnet2",
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=_ext_sources,
            include_dirs=[include_dir],
            extra_compile_args={"cxx": extra_cxx_flags, "nvcc": extra_nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
