from setuptools import setup, find_packages


setup(
    name="corrpops",
    version="0.0.0",
    description="",
    author="Itamar Faran",
    author_email="itamar.faran@gmail.com",
    url="https://github.com/itamarfaran/corrpops-python",
    packages=find_packages(include=["corrpops", "corrpops.*"]),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0,",
        "numba",
    ],
    tests_require=["pytest"],
)
