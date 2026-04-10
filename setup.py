import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    #0packages=[], 
    ext_modules=cythonize(
        ["train/oxono_train.pyx", "train/mcts_train.pyx"],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
    packages=[],
    include_dirs=[numpy.get_include()],
)
