from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'neutron_math',
        ['similarity.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3'], # Optimization
    ),
]

setup(
    name='neutron_math',
    ext_modules=ext_modules,
    install_requires=['pybind11'],
)