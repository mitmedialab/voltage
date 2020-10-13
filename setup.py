from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

NAME = "preprocess"
VERSION = "0.0.1"
DESCR = "Voltage Imaging Preprocessing"
LANGUAGE="c++"


examples_extension = Extension(
    name=NAME,
    sources=["preprocess.pyx"],
    libraries=["preprocess", "m", "pthread", "lzma", "jbig", "jpeg", "z", "cudart", "nppif", "nppig", "nppc", "nppial", "nppicc", "nppidei", "nppig", "nppim", "nppist", "nppisu", "nppitc", "npps"],
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
