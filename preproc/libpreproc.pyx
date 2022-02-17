import cython
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as np


cdef extern from "preproc.h":
    int preprocess_video_cpu(int num_frames, int height, int width,
                             float *in_image,
                             float **out_temporal,
                             float **out_spatial,
                             int signal_method, int signal_period,
                             double signal_scale,
                             int num_threads) nogil

def preprocess_video_cython(np.ndarray[np.float32_t, ndim=3] in_image,
                            int signal_method, int signal_period,
                            double signal_scale,
                            int num_threads):
    """
    Binder function to call a C++ implementation of preprocess_video().
    Refer to preprocess_video() in preproc.py for parameter definitions.
    """
    cdef float *out_temporal
    cdef float *out_spatial
    cdef int num_out
    
    cdef int t = in_image.shape[0]
    cdef int h = in_image.shape[1]
    cdef int w = in_image.shape[2]
    in_image = np.ascontiguousarray(in_image)
    
    with nogil:
        num_out = preprocess_video_cpu(t, h, w,
                                       &in_image[0, 0, 0],
                                       &out_temporal,
                                       &out_spatial,
                                       signal_method, signal_period,
                                       signal_scale,
                                       num_threads)
    
    cdef cvarray arr_t = <float [:num_out, :h, :w]>out_temporal
    cdef cvarray arr_s = <float [:num_out, :h, :w]>out_spatial
    arr_t.free_data = True
    arr_s.free_data = True
    return np.asarray(arr_t), np.asarray(arr_s)
