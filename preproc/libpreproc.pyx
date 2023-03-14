import cython
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as np


cdef extern from "preproc.h":
    void preprocess_video_cpu(int in_num_frames, int in_height, int in_width,
                              float *in_image,
                              int *out_num_frames, int *out_height, int *out_width,
                              float **out_temporal, float **out_spatial,
                              int signal_method, int signal_period,
                              double signal_scale, double downsampling_factor,
                              int num_threads) nogil

def preprocess_video_cython(np.ndarray[np.float32_t, ndim=3] in_image,
                            int signal_method, int signal_period,
                            double signal_scale, double downsampling_factor,
                            int num_threads):
    """
    Binder function to call a C++ implementation of preprocess_video().
    Refer to preprocess_video() in preproc.py for parameter definitions.
    """
    cdef float *out_temporal
    cdef float *out_spatial
    cdef int out_t, out_h, out_w

    cdef int in_t = in_image.shape[0]
    cdef int in_h = in_image.shape[1]
    cdef int in_w = in_image.shape[2]
    in_image = np.ascontiguousarray(in_image)
    
    with nogil:
        preprocess_video_cpu(in_t, in_h, in_w, &in_image[0, 0, 0],
                             &out_t, &out_h, &out_w,
                             &out_temporal, &out_spatial,
                             signal_method, signal_period, signal_scale,
                             downsampling_factor, num_threads)

    cdef cvarray arr_t = <float [:out_t, :out_h, :out_w]>out_temporal
    cdef cvarray arr_s = <float [:out_t, :out_h, :out_w]>out_spatial
    arr_t.free_data = True
    arr_s.free_data = True
    return np.asarray(arr_t), np.asarray(arr_s)
