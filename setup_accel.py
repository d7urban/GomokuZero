"""
Build:  python setup_accel.py build_ext --inplace
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "mcts_accel",
        ["mcts_accel.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "language_level": 3,
    }),
)
