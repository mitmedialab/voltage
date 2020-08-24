#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include "wrapper.hpp"


void* populate_parameters(void* allparams)
{
    param_list *pm = (param_list *) allparams;
    preprocess_params *params = (preprocess_params *) malloc(sizeof(preprocess_params));

    if(pm->is_motion_correction == 1) {
        params->is_motion_correction = true;
    } else {
        params->is_motion_correction = false;
    }

    params->mp.level = pm->level;
    params->mp.search_size = pm->search_size;
    params->mp.patch_size = pm->patch_size;
    params->mp.patch_offset = pm->patch_offset;
    params->mp.x_range = pm->x_range;
    params->mp.y_range = pm->y_range;
    params->mp.a_stdev = pm->a_stdev;
    params->mp.m_stdev = pm->m_stdev;
    params->mp.thresh_xy = pm->thresh_xy;
    params->mp.length = pm->length;
    params->mp.thresh_c = pm->thresh_c;

    return (void *) params;
}

void normalize_intensity(int t, int w, int h, float* img)
{
    float *lumi = (float *) malloc (t * sizeof(float));
    float max = 0;
    int i, j, k;
    float scale;
    float val;
    float *im;

    for(k = 0; k < t; ++k) {
        lumi[k] = 0;
        im = loc3D(img, t, h, w, k, 0, 0);
        for(i = 0; i < h * w; ++i) {
            val = im[i];
            lumi[k] += val;
            if(max < val) {
                max = val;
            }
        }
    }

    #pragma omp parallel for 
    for(k = 0; k < t; ++k) {
        scale = lumi[0] / lumi[k] / max;
        im = loc3D(img, t, h, w, k, 0, 0);
        // printf("k: %d, scale: %f\n", k, scale);

        for(i = 0; i < h * w; ++i) {
            im[i] *= scale;
        }
    }
}

float* vol_preprocess(int t, int h, int w, float *img, void* pm)
{
    preprocess_params *params = (preprocess_params *) pm;
    printf("T: %d, W: %d, H: %d, SS: %d\n", t, w, h, params->mp.search_size);
    
    std::vector<motion_t> motion_list;
    motion_range_t range;

    // remove the black line at bottom
    for(int k = 0; k < t; k++) {
        for(int j = 0; j < w; j++) {
            *loc3D(img, t, h, w, k, h-1, j) = *loc3D(img, t, h, w, k, h-2, j);
        }
    }
    normalize_intensity(t, w, h, img);
    

    if(params->is_motion_correction == true) {
        motion_list = correct_motion(params->mp, t, w, h, img, range);
    }

    // if(params->is_shading_correction == true) {
    //  correct_shading(params->sp, t, w, h, img, motion_list);
    // }

    // blood_out *bld = NULL;
    // if(params->is_blood_suppression == true) {
    //  bld = suppress_blood(params->bp, t, w, h, img);
    // }

    // int num_out = 0;
    // if(params->is_signal_extraction == true) {
    //  sig = extract_signal(params->sigp, t, w, h, img, motion_list, range, &num_out);
    // }



    // float *B = (float *) malloc (t * w * h * sizeof(float));
    // memcpy(B, img, t * w * h * sizeof(float));

    return img;
}
