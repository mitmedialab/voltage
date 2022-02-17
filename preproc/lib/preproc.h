#ifndef __PREPROC_H__
#define __PREPROC_H__

/*

C++ implementation of preprocessing with OpenMP parallelization on CPU.

*/
int preprocess_video_cpu(int num_frames, int height, int width,
                         float *in_image,
                         float **out_temporal,
                         float **out_spatial,
                         int signal_method, int signal_period, double signal_scale,
                         int num_threads);

#endif

