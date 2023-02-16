#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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

    TimerUtil *tu = new TimerUtil("signal extraction");
    int num_out = extract_signal(signal_param, num_frames, width, height, in_image,
                                 out_temporal, out_spatial);
    delete tu;

    return num_out;
}

