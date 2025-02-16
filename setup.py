from setuptools import setup
import numpy


setup(
    include_dirs=[numpy.get_include()],
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0,",
        "numba",
    ],
)
