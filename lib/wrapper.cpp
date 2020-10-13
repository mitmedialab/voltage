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

	params->magnification = pm->magnification;
	params->out_dim = pm->out_dim;

	if(pm->is_motion_correction == 1) {
		params->is_motion_correction = true;
	} else {
		params->is_motion_correction = false;
	}

	params->mp.level = pm->mp_level;
    params->mp.search_size = pm->mp_search_size;
    params->mp.patch_size = pm->mp_patch_size;
    params->mp.patch_offset = pm->mp_patch_offset;
    params->mp.x_range = pm->mp_x_range;
    params->mp.y_range = pm->mp_y_range;
	params->mp.a_stdev = pm->mp_a_stdev;
    params->mp.m_stdev = pm->mp_m_stdev;
    params->mp.thresh_xy = pm->mp_thresh_xy;
    params->mp.length = pm->mp_length;
    params->mp.thresh_c = pm->mp_thresh_c;

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

    // #pragma omp parallel for 
	for(k = 0; k < t; ++k) {
		scale = (lumi[0] / lumi[k]) / max;
		im = loc3D(img, t, h, w, k, 0, 0);
		for(i = 0; i < h * w; ++i) {
			im[i] *= scale;
		}
	}
}

void vol_preprocess(int t, int h, int w, float *img, void* pm, void *out_v)
{
	preprocess_params *params = (preprocess_params *) pm;
	out_preprocess *out = (out_preprocess *) out_v;

	// remove the black line at bottom
	for(int k = 0; k < t; k++) {
		for(int j = 0; j < w; j++) {
			*loc3D(img, t, h, w, k, h-1, j) = *loc3D(img, t, h, w, k, h-2, j);
		}
	}
	normalize_intensity(t, w, h, img);
	
	motion_buffer_t *mbuf = new motion_buffer_t(img, t, h, w);
	if(params->is_motion_correction == true) {
		out->mc_out = correct_motion(params->mp, mbuf);
	}

	out->proc_out = preprocess_unet(mbuf, params->magnification, &out->t_out, params->out_dim);

}
