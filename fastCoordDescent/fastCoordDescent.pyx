import cython

import numpy as np
import time
from libcpp.string cimport string
from libcpp cimport map
from libcpp cimport bool
cimport numpy as np


cdef extern from "wrapper.hpp":
    void CoordDescent_fit(float *X, float *W, float *H, int num_samples, int num_features, int n_components, int max_iterations, double tolerance, int *n_iter)

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free


def fastCoordDescent(np.ndarray[np.float32_t, ndim=2] X, np.ndarray[np.float32_t, ndim=2] W, np.ndarray[np.float32_t, ndim=2] H, int max_iter, double tolerance):

    # Convert image from numpy array to a contiguous memory arrangement
    cdef int n_iter
    ns, nf, nc = X.shape[0], X.shape[1], W.shape[1]
    X = np.ascontiguousarray(X)
    W = np.ascontiguousarray(W)
    H = np.ascontiguousarray(H)

    CoordDescent_fit(&X[0,0], &W[0,0], &H[0,0], ns, nf, nc, max_iter, tolerance, &n_iter)

    return W, H, n_iter

