import cython
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as np


cdef extern from "preproc.h":
    int preprocess_cpu(int num_frames, int height, int width,
                       float *in_image,
                       float **out_image,
                       float **out_temporal,
                       int motion_search_level, int motion_search_size,
                       int motion_patch_size, int motion_patch_offset,
                       int shading_period,
                       int signal_method, int signal_period,
                       double signal_scale,
                       int num_threads) nogil

def preprocess_cython(np.ndarray[np.float32_t, ndim=3] in_image,
                      int motion_search_level, int motion_search_size,
                      int motion_patch_size, int motion_patch_offset,
                      int shading_period,
                      int signal_method, int signal_period,
                      double signal_scale,
                      int num_threads):
    """
    Binder function to call a C++ implementation of preprocess().
    Refer to preprocess() in preproc.py for parameter definitions.
    """
    cdef float *out_image
    cdef float *out_temporal
    cdef int num_out
    
    cdef int t = in_image.shape[0]
    cdef int h = in_image.shape[1]
    cdef int w = in_image.shape[2]
    in_image = np.ascontiguousarray(in_image)
    
    with nogil:
        num_out = preprocess_cpu(t, h, w,
                                 &in_image[0, 0, 0],
                                 &out_image,
                                 &out_temporal,
                                 motion_search_level, motion_search_size,
                                 motion_patch_size, motion_patch_offset,
                                 shading_period,
                                 signal_method, signal_period, signal_scale,
                                 num_threads)
    
    cdef cvarray arr_i = <float [:t, :h, :w]>out_image
    cdef cvarray arr_t = <float [:num_out, :h, :w]>out_temporal
    arr_i.free_data = True
    arr_t.free_data = True
    return np.asarray(arr_i), np.asarray(arr_t)
