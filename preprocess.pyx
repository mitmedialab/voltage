import cython

import numpy as np
from libcpp.string cimport string
from libcpp cimport map
cimport numpy as np

cdef struct s_param_list:
    int is_motion_correction
    int level
    int search_size
    int patch_size
    int patch_offset
    
    float x_range
    float y_range

    float a_stdev
    float m_stdev
    float thresh_xy
    int length
    float thresh_c

ctypedef s_param_list param_list

cdef extern from "wrapper.hpp":
    void* populate_parameters(void* allparams)
    float *vol_preprocess(int t, int w, int h, float *img, void *allparams)

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free
from libcpp cimport bool


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.

cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size
    cdef bool skip_dealloc

    cdef set_data(self, int size, void* data_ptr, bool skip_dealloc=False):
        """ Set the data of the array
        This cannot be done in the constructor as it must recieve C-level
        arguments.
        Parameters:
        -----------
        size: int
            Length of the array.
        data_ptr: void*
            Pointer to the data            
        """
        self.data_ptr = data_ptr
        self.size = size
        self.skip_dealloc = skip_dealloc

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_FLOAT32, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        if(self.skip_dealloc == False):
            free(<void*>self.data_ptr)



def preprocess(np.ndarray[np.float32_t, ndim=3] image, dict mp):

    cdef float *mc
    cdef void* params
    cdef param_list *lis = <param_list *> malloc(sizeof(param_list))
    cdef np.ndarray ndarray

    t, h, w = image.shape[0], image.shape[1], image.shape[2]
    image = np.ascontiguousarray(image)

    lis.is_motion_correction = mp['is_motion_correction'];
    lis.level = mp['level'];
    lis.search_size = mp['search_size'];
    lis.patch_size = mp['patch_size'];
    lis.patch_offset = mp['patch_offset'];
    lis.x_range = mp['x_range'];
    lis.y_range = mp['y_range'];
    lis.a_stdev = mp['a_stdev'];
    lis.m_stdev = mp['m_stdev'];
    lis.thresh_xy = mp['thresh_xy'];
    lis.length = mp['length'];
    lis.thresh_c = mp['thresh_c'];

    params = populate_parameters(lis)
    mc = vol_preprocess(t, h, w, &image[0,0,0], params)

    array_wrapper = ArrayWrapper()
    array_wrapper.set_data(t * h * w, <void*> mc, skip_dealloc=True)
    ndarray = np.array(array_wrapper, copy=False)
    # Assign our object to the 'base' of the ndarray object
    ndarray.base = <PyObject*> array_wrapper
    # Increment the reference count, as the above assignement was done in
    # C, and Python does not know that there is this additional reference
    Py_INCREF(array_wrapper)

    ndarray = np.reshape(ndarray, (t, h, w))
    # print("in pyx")
    free(lis)
    return image