from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

BASENAME = 'preproc'
NAME = 'lib' + BASENAME

extension = Extension(
    name=NAME,
    sources=[NAME + '.pyx'],
    libraries=[BASENAME, 'utils', 'm', 'pthread'],
    language='c++',
    library_dirs=['lib', '../utils'],
    include_dirs=['lib', '../utils', numpy.get_include()],
)

setup(
    name=NAME,
    ext_modules=cythonize([extension]),
    version='0.1',
    description='Preprocessing for voltage imaging data'
)
