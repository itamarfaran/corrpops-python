from setuptools import setup


setup(
    name="corrpops",
    version="0.0.0",
    description="",
    author="Itamar Faran",
    author_email="itamar.faran@gmail.com",
    url="https://github.com/itamarfaran/corrpops-python",
    packages=["corrpops"],
    python_requires=">=3.9.5",
    install_requires=[
        "numpy >= 1.26.0",
        "scipy >= 1.11.0",
        "numba >= 0.59.0",
    ],
    extras_require={
        "ray": ["ray >= 2.10.0"],
    },
    tests_require=[
        "pytest >= 8.0.0",
        "statsmodels >= 0.12.2",
    ],
)
