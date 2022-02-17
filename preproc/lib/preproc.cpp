#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "malloc_util.h"
#include "image_util.h"
#include "TimerUtil.h"
#include "signal.h"


int preprocess_video_cpu(int num_frames, int height, int width,
                         float *in_image,
                         float **out_temporal,
                         float **out_spatial,
                         int signal_method, int signal_period, double signal_scale,
                         int num_threads)
{
    if(num_threads > 0)
    {
        omp_set_num_threads(num_threads);
    }

    const int t = num_frames;
    const int h = height;
    const int w = width;

    signal_param_t signal_param;
    signal_param.method = signal_method;
    signal_param.period = signal_period;
    signal_param.frames_per_sec = 1000.0;
    signal_param.freq_max = 0.0;
    signal_param.normalize = false;
    signal_param.downsample = true;
    signal_param.patch_size = 8;
    signal_param.patch_offset = 1;
    signal_param.smooth_scale = (float)signal_scale;

    float ***img = malloc_float3d(t, w, h);
    copy1d_to_3d(t, w, h, in_image, img);

    int num_out = 0;
    float ***temporal = NULL;
    float ***spatial = NULL;

    TimerUtil *tu = new TimerUtil("signal extraction");
    num_out = extract_signal(signal_param, t, w, h, img, &temporal, &spatial);
    delete tu;

    free_float3d(img);
    if(num_out > 0)
    {
        *out_temporal = malloc_float1d(num_out * h * w);
        copy3d_to_1d(num_out, w, h, temporal, *out_temporal);
        free_float3d(temporal);

        *out_spatial = malloc_float1d(num_out * h * w);
        copy3d_to_1d(num_out, w, h, spatial, *out_spatial);
        free_float3d(spatial);
    }
    return num_out;
}

