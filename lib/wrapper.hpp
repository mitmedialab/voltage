#include <map>
#include <string>

#ifndef __WRAPPER_HPP__
#define __WRAPPER_HPP__

#include "motion.hpp"
#include "pre_unet.hpp"

#define BYTESPERPIXEL 1

typedef struct param_list {
    int magnification;
    int out_dim;
	int is_motion_correction;
    int mp_level;
    int mp_search_size;
    int mp_patch_size;
    int mp_patch_offset;
    
    float mp_x_range;
    float mp_y_range;

    float mp_a_stdev;
    float mp_m_stdev;
    float mp_thresh_xy;
    int mp_length;
    float mp_thresh_c;
    
} param_list;




typedef struct preprocess_params {

	bool is_motion_correction;
    int magnification;
    int out_dim;
	motion_param_t mp;

} preprocess_params;

typedef struct out_preprocess {
    float *mc_out;
    float *proc_out;
    int t_out;
} out_preprocess;


inline float* loc3D(float *img, int t, int h, int w, int k, int i, int j)
{
    return img + ((k * w * h) + ((i * w) + j)) * BYTESPERPIXEL; 
}

inline float* loc3D_3C(float *img, int t, int h, int w, int k, int i, int j)
{
    return img + ((k * w * h) + ((i * w) + j)) * 3; 
}

inline float* loc2D(float *img, int h, int w, int i, int j)
{
    return img + ((i * w) + j) * BYTESPERPIXEL; 
}

inline double* loc2D(double *img, int h, int w, int i, int j)
{
    return img + ((i * w) + j) * BYTESPERPIXEL; 
}

void* populate_parameters(void* allparams);
void vol_preprocess(int t, int w, int h, float *img, void* allparams, void *out);

#endif