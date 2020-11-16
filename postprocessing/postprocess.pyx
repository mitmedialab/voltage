import cython
from cython.view cimport array as cvarray
import numpy as np
import time
from libcpp.string cimport string
from libcpp cimport map
from libcpp cimport bool
cimport numpy as np


cdef extern from "wrapper.hpp":
    void postprocess_frames(float *image, float *masks, int t, int h, int w, int n, float **sig)

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free



def postprocess(np.ndarray[np.float32_t, ndim=3] image, np.ndarray[np.float32_t, ndim=3] masks):

    cdef float *sig

    # Convert image from numpy array to a contiguous memory arrangement
    t, h, w, n = image.shape[0], image.shape[1], image.shape[2], masks.shape[0]
    image = np.ascontiguousarray(image)
    masks = np.ascontiguousarray(masks)

    postprocess_frames(&image[0,0,0], &masks[0,0,0], t, h, w, n, &sig)

    # Get the CPP buffer to numpy format
    cdef cvarray arr = <float [:n, :t]>sig
    arr.free_data = True
    signp = np.asarray(arr)

    return signp

