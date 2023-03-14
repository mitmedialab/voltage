#ifndef __PREPROC_H__
#define __PREPROC_H__

/*

C++ implementation of preprocessing with OpenMP parallelization on CPU.

*/
void preprocess_video_cpu(int in_num_frames, int in_height, int in_width,
                          float *in_image,
                          int *out_num_frames, int *out_height, int *out_width,
                          float **out_temporal, float **out_spatial,
                          int signal_method, int signal_period, double signal_scale,
                          double downsampling_factor, int num_threads);

#endif

