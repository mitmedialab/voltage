import cython
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as np


cdef extern from "demix.h":
    int demix_cells_cpu(int num_frames, int height, int width,
                        double *probability_maps,
                        int num_cells, double *z_init,
                        double **z_out, double **c_out, double *err,
                        int max_iter, double update_step, double iter_thresh,
                        int num_threads) nogil

def demix_cells_cython(np.ndarray[np.float64_t, ndim=3] probability_maps,
                       int num_cells, np.ndarray[np.float64_t, ndim=3] z_init,
                       int max_iter, double update_step, double iter_thresh,
                       int num_threads=1):
    """
    Binder function to call a C++ implementation of demix_cells().
    Refer to demix_cells() in demix.py for parameter definitions.
    """
    cdef double *z_out
    cdef double *c_out
    cdef double err
    cdef int num_iter

    cdef int num_frames = probability_maps.shape[0]
    cdef int height = probability_maps.shape[1]
    cdef int width = probability_maps.shape[2]

    probability_maps = np.ascontiguousarray(probability_maps)
    z_init = np.ascontiguousarray(z_init)
    
    with nogil:
        num_iter = demix_cells_cpu(num_frames, height, width,
                                   &probability_maps[0, 0, 0],
                                   num_cells, &z_init[0, 0, 0],
                                   &z_out, &c_out, &err,
                                   max_iter, update_step, iter_thresh,
                                   num_threads)

    cdef cvarray arr_z = <double [:num_cells, :height, :width]>z_out
    cdef cvarray arr_c = <double [:num_cells, :num_frames]>c_out
    arr_z.free_data = True
    arr_c.free_data = True
    return np.asarray(arr_z), np.asarray(arr_c), err, num_iter
