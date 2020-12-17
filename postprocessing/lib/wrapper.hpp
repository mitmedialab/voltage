#include <map>
#include <string>

#ifndef __WRAPPER_HPP__
#define __WRAPPER_HPP__

#include "postprocess.hpp"

inline float* loc3D(float *img, int t, int h, int w, int k, int i, int j)
{
    return img + ((k * w * h) + ((i * w) + j)); 
}

inline float* loc3D_3C(float *img, int t, int h, int w, int k, int i, int j)
{
    return img + ((k * w * h) + ((i * w) + j)) * 3; 
}

inline float* loc2D(float *img, int h, int w, int i, int j)
{
    return img + ((i * w) + j); 
}

inline double* loc2D(double *img, int h, int w, int i, int j)
{
    return img + ((i * w) + j); 
}

void postprocess_frames(float *image, float *masks, int t, int h, int w, int n, float **sig);
void exp_spread(float *image, int t, int h, int w, float **out);
#endif
