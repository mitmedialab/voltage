#include <map>
#include <string>

#ifndef __WRAPPER_HPP__
#define __WRAPPER_HPP__

#include "motion.hpp"

#define BYTESPERPIXEL 1

typedef struct param_list {
	int is_motion_correction;
    int level;
    int search_size;
    int patch_size;
    int patch_offset;
    
    float x_range;
    float y_range;

    float a_stdev;
    float m_stdev;
    float thresh_xy;
    int length;
    float thresh_c;
} param_list;

typedef struct preprocess_params {

	bool is_motion_correction;
	motion_param_t mp;

} preprocess_params;


float* loc3D(float *img, int t, int w, int h, int k, int i, int j);
float* loc2D(float *img, int w, int h, int i, int j);
double* loc2D(double *img, int h, int w, int i, int j);
void* populate_parameters(void* allparams);
float* vol_preprocess(int t, int w, int h, float *img, void* allparams);

#endif