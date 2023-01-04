#ifndef __CORRECT_H__
#define __CORRECT_H__

/*

C++ implementation of motion/shading correction
with CPU multithreaded parallelization using OpenMP
and GPU parallelization using CUDA

*/
void correct_video_cpp(int num_frames, int height, int width,
                       float *in_image,
                       float **out_image,
                       float **out_x, float **out_y,
                       int normalize,
                       int motion_search_level, int motion_search_size,
                       int motion_patch_size, int motion_patch_offset,
                       float motion_x_range, float motion_y_range,
                       int shading_period,
                       int use_gpu, int num_frames_per_batch, int num_threads);

#endif

