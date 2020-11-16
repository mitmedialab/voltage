from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

NAME = "fastCoordDescent"
VERSION = "0.0.1"
DESCR = "Coordinate Descent Fit"
LANGUAGE="c++"


examples_extension = Extension(
    name=NAME,
    sources=["fastCoordDescent.pyx"],
    libraries=["fastCoordDescent", "m", "pthread", "cudart", "cublas", "nppc", "nppif", "nppig", "nppial", "nppicc", "nppidei", "nppig", "nppim", "nppist", "nppisu", "nppitc", "npps"],
    language=LANGUAGE,
    library_dirs=["lib"],
    include_dirs=["lib", numpy.get_include()]
)
setup(
    name=NAME,
    ext_modules=cythonize([examples_extension]),
    version=VERSION,
    description=DESCR
)
