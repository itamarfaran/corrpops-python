from setuptools import setup


setup(
    name="corrpops",
    version="0.0.0",
    description="",
    author="Itamar Faran",
    author_email="itamar.faran@gmail.com",
    url="https://github.com/itamarfaran/corrpops-python",
    packages=["corrpops"],
    python_requires=">=3.9",
    install_requires=[
        "numpy >= 1.23.0",
        "scipy >= 1.9.0",
        "numba >= 0.56.0",
    ],
    extras_require={
        "ray": ["ray >= 2.0.0"],
    },
    tests_require=[
        "pytest",
        "statsmodels",
    ],
)
