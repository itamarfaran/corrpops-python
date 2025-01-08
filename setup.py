from setuptools import setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize("corrpops/corr_calc_cy.pyx"),
    include_dirs=[numpy.get_include()],
    compiler_directives={"language_level": "3"},
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.6.0,',
        'numba',
    ]
)
