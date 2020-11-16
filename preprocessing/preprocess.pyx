import cython
from cython.view cimport array as cvarray
import numpy as np
from libcpp.string cimport string
from libcpp cimport map
from libcpp cimport bool
cimport numpy as np

cdef struct s_param_list:
    int magnification
    int out_dim
    int is_motion_correction
    int mp_level
    int mp_search_size
    int mp_patch_size
    int mp_patch_offset
    
    float mp_x_range
    float mp_y_range

    float mp_a_stdev
    float mp_m_stdev
    float mp_thresh_xy
    int mp_length
    float mp_thresh_c

cdef struct s_out_list:
    float *mc_out
    float *proc_out
    int t_out;



ctypedef s_param_list param_list
ctypedef s_out_list out_preprocess

cdef extern from "wrapper.hpp":
    void* populate_parameters(void* allparams)
    void vol_preprocess(int t, int w, int h, float *img, void *allparams, void *out) except +

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free

def preprocess(np.ndarray[np.float32_t, ndim=3] image, dict mp):

    cdef float *mc
    cdef out_preprocess out
    cdef void* params
    cdef param_list *lis = <param_list *> malloc(sizeof(param_list))
    cdef np.ndarray ndarray

    # Conver image from numpy array to a contiguous memory arrangement
    t, h, w = image.shape[0], image.shape[1], image.shape[2]
    image = np.ascontiguousarray(image)

    # Copy all the parameters from python
    lis.magnification = mp['magnification']
    lis.out_dim = mp['output_dimension']
    lis.is_motion_correction = mp['is_motion_correction'];
    lis.mp_level = 2;
    lis.mp_search_size = mp['motion_correction']['search_size'];
    lis.mp_patch_size = mp['motion_correction']['patch_size'];
    lis.mp_patch_offset = mp['motion_correction']['patch_offset'];
    lis.mp_length = mp['motion_correction']['length'];
    lis.mp_x_range = 0.7;
    lis.mp_y_range = 1.0;
    lis.mp_a_stdev = 1.0;
    lis.mp_m_stdev = 3.0;
    lis.mp_thresh_xy = 1.0;
    lis.mp_thresh_c = 0.4;

    # Populate the parameters to CPP
    params = populate_parameters(lis)

    # Preprocessing
    vol_preprocess(t, h, w, &image[0,0,0], params, &out)

    # Get the CPP buffer to numpy format
    cdef cvarray arr = <float [:t, :h, :w]>out.mc_out
    arr.free_data = True
    mc_out = np.asarray(arr)

    # cdef float[:,:,:,:] arr1 = <float [:out.t_out, :lis.out_dim, :lis.out_dim, :3]>out.proc_out
    cdef cvarray arr1 = <float [:out.t_out, :lis.out_dim, :lis.out_dim, :3]>out.proc_out
    arr1.free_data = True
    proc_out = np.asarray(arr1)

    free(lis)

    return mc_out, proc_out