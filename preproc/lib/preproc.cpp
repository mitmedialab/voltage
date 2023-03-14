#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "TimerUtil.h"
#include "malloc_util.h"
#include "signal.h"


static float *downsample_image(int num_frames, int height, int width,
                               float *in_image, double downsampling_factor,
                               int *new_height, int *new_width)
{
    const int h = (int)floor(height / downsampling_factor + 0.5);
    const int w = (int)floor(width  / downsampling_factor + 0.5);
    const double scale_x = width  / (double)w;
    const double scale_y = height / (double)h;
    float *out_image = malloc_float1d(num_frames * h * w);

    #pragma omp parallel for
    for(int k = 0; k < num_frames; k++)
    {
        float *in_p = &in_image[k * height * width];
        float *out_p = &out_image[k * h * w];

        for(int j = 0; j < h; j++)
        for(int i = 0; i < w; i++)
        {
            const double fsx = i * scale_x;
            const double fsy = j * scale_y;
            const double fex = fsx + scale_x;
            const double fey = fsy + scale_y;
            const int isx = (int)floor(fsx);
            const int isy = (int)floor(fsy);
            const int iex = (int)floor(fex);
            const int iey = (int)floor(fey);

            double sum = 0;
            double area = 0;
            for(int y = isy; y <= iey; y++)
            {
                double frac_y = 1.0;
                if(y == isy) frac_y -= fsy - isy;
                else if(y == iey) frac_y = fey - iey;

                for(int x = isx; x <= iex; x++)
                {
                    double frac_x = 1.0;
                    if(x == isx) frac_x -= fsx - isx;
                    else if(x == iex) frac_x = fex - iex;

                    sum += in_p[y * width + x] * frac_x * frac_y;
                    area += frac_x * frac_y;
                }
            }
            out_p[j * w + i] = (float)(sum / area);
        }
    }
    *new_height = h;
    *new_width = w;
    return out_image;
}

void preprocess_video_cpu(int in_num_frames, int in_height, int in_width,
                          float *in_image,
                          int *out_num_frames, int *out_height, int *out_width,
                          float **out_temporal, float **out_spatial,
                          int signal_method, int signal_period, double signal_scale,
                          double downsampling_factor, int num_threads)
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

    TimerUtil *tu;
    if(downsampling_factor == 1.0)
    {
        *out_height = in_height;
        *out_width = in_width;

        tu = new TimerUtil("signal extraction");
        *out_num_frames = extract_signal(signal_param,
                                         in_num_frames, in_width, in_height, in_image,
                                         out_temporal, out_spatial);
        delete tu;
    }
    else
    {
        tu = new TimerUtil("downsampling");
        float *down = downsample_image(in_num_frames, in_height, in_width, in_image,
                                       downsampling_factor, out_height, out_width);
        delete tu;

        tu = new TimerUtil("signal extraction");
        *out_num_frames = extract_signal(signal_param,
                                         in_num_frames, *out_width, *out_height, down,
                                         out_temporal, out_spatial);
        delete tu;

        free_float1d(down);
    }
}

