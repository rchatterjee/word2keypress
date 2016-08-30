from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(
    name = "My hello app",
    ext_modules = cythonize('_keyboard.pyx'),  # accepts a glob pattern
    include_dirs = [np.get_include()]
)
