from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

extra_compile_args = {
    'cxx': ['-g3', '-O0', '-fPIC'],  # Enable debug info and disable optimization
}

setup(
    name="my_custom_backend_called_virtd",
    ext_modules=[CppExtension(
        "virtd",
        sources=["virtd_backend.cpp"],
        extra_compile_args=extra_compile_args['cxx'],
    )],
    cmdclass={"build_ext": BuildExtension}
)
