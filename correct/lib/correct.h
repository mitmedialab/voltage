#ifndef __CORRECT_H__
#define __CORRECT_H__

/*

C++ implementation of motion/shading correction
with OpenMP parallelization on CPU.

*/
void correct_video_cpu(int num_frames, int height, int width,
                       float *in_image,
                       float **out_image,
                       float **out_x, float **out_y,
                       int motion_search_level, int motion_search_size,
                       int motion_patch_size, int motion_patch_offset,
                       int shading_period,
                       int num_threads);

#endif

