from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("haloutils.pyx")
    # Use th efollowing command to build the cython module
    # $ python setup.py build_ext --inplace
)