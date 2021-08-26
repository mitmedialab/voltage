from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

BASENAME = 'demix'
NAME = 'lib' + BASENAME

extension = Extension(
    name=NAME,
    sources=[NAME + '.pyx'],
    libraries=[BASENAME, 'm', 'pthread'],
    language='c++',
    library_dirs=['lib'],
    include_dirs=['lib', numpy.get_include()],
)

setup(
    name=NAME,
    ext_modules=cythonize([extension]),
    version='0.1',
    description='Cell demixing for voltage imaging data'
)
