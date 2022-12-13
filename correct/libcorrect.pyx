import cython
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as np


cdef extern from "correct.h":
    void correct_video_cpp(int num_frames, int height, int width,
                           float *in_image,
                           float **out_image,
                           float **out_x, float **out_y,
                           int normalize,
                           int motion_search_level, int motion_search_size,
                           int motion_patch_size, int motion_patch_offset,
                           int shading_period,
                           int use_gpu, int num_threads)

def correct_video_cython(np.ndarray[np.float32_t, ndim=3] in_image,
                         int normalize,
                         int motion_search_level, int motion_search_size,
                         int motion_patch_size, int motion_patch_offset,
                         int shading_period,
                         int use_gpu, int num_threads):
    """
    Binder function to call a C++ implementation of correct_video().
    Refer to correct_video() in correct.py for parameter definitions.
    """
    cdef float *out_image
    cdef float *out_x
    cdef float *out_y
    
    cdef int t = in_image.shape[0]
    cdef int h = in_image.shape[1]
    cdef int w = in_image.shape[2]
    in_image = np.ascontiguousarray(in_image)

    correct_video_cpp(t, h, w,
                      &in_image[0, 0, 0],
                      &out_image,
                      &out_x, &out_y,
                      normalize,
                      motion_search_level, motion_search_size,
                      motion_patch_size, motion_patch_offset,
                      shading_period,
                      use_gpu, num_threads)

    cdef cvarray arr_c = <float [:t, :h, :w]>out_image
    cdef cvarray arr_x = <float [:t]>out_x
    cdef cvarray arr_y = <float [:t]>out_y
    arr_c.free_data = True
    arr_x.free_data = True
    arr_y.free_data = True
    return np.asarray(arr_c), np.asarray(arr_x), np.asarray(arr_y)
