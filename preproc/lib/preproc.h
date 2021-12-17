#ifndef __PREPROC_H__
#define __PREPROC_H__

/*

C implementation of preprocessing with OpenMP parallelization on CPU.

*/
int preprocess_cpu(int num_frames, int height, int width,
                   float *in_image,
                   float **out_image,
                   float **out_temporal,
                   int motion_search_level, int motion_search_size,
                   int motion_patch_size, int motion_patch_offset,
                   int shading_period,
                   int signal_method, int signal_period, double signal_scale);

#endif

