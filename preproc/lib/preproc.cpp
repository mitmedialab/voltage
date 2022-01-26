#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <vector>

#include "malloc_util.h"
#include "TimerUtil.h"
#include "motion.h"
#include "shading.h"
#include "signal.h"


static void normalize_intensity(int num_frames, int width, int height, float ***img)
{
    float *lumi = malloc_float1d(num_frames);
    float max = 0;
    
    for(int k = 0; k < num_frames; k++)
    {
        lumi[k] = 0;
        for(int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
        {
            lumi[k] += img[k][i][j];
            if(max < img[k][i][j]) max = img[k][i][j];
        }
    }

    #pragma omp parallel for
    for(int k = 0; k < num_frames; k++)
    {
        float scale = lumi[0] / lumi[k] / max;
        for(int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
        {
            img[k][i][j] *= scale;
        }
    }
    
    free_float1d(lumi);
}

static void copy1d_to_3d(int num_frames, int width, int height, float *in, float ***out)
{
    size_t n = 0;
    for(int k = 0; k < num_frames; k++)
    for(int j = 0; j < height; j++)
    for(int i = 0; i < width; i++)
    {
        out[k][i][j] = in[n++];
    }
}

static void copy3d_to_1d(int num_frames, int width, int height, float ***in, float *out)
{
    size_t n = 0;
    for(int k = 0; k < num_frames; k++)
    for(int j = 0; j < height; j++)
    for(int i = 0; i < width; i++)
    {
        out[n++] = in[k][i][j];
    }
}


int preprocess_cpu(int num_frames, int height, int width,
                   float *in_image,
                   float **out_image,
                   float **out_temporal,
                   float **out_spatial,
                   float **out_x, float **out_y,
                   int motion_search_level, int motion_search_size,
                   int motion_patch_size, int motion_patch_offset,
                   int shading_period,
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
    
    motion_param_t motion_param;
    motion_param.level = motion_search_level;
    motion_param.search_size = motion_search_size;
    motion_param.patch_size = motion_patch_size;
    motion_param.patch_offset = motion_patch_offset;
    motion_param.x_range = 1.0; // used to be 0.7 for horizontal images
    motion_param.y_range = 1.0;
    motion_param.thresh_xy = 10.0;
    motion_param.length = 1000;
    motion_param.thresh_c = 1.0;

    shading_param_t shading_param;
    shading_param.period = shading_period;

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

    bool skip_motion_correction = false;
    bool skip_shading_correction = false;
    bool skip_signal_extraction = false;

    float ***img = malloc_float3d(t, w, h);
    copy1d_to_3d(t, w, h, in_image, img);
    
    TimerUtil *tu;

    std::vector<motion_t> motion_list;
    motion_range_t range;
    if(skip_motion_correction)
    {
        printf("motion correction skipped\n");
        //motion_list = read_motion_file(motion_file, t, range);
    }
    else
    {
        // hack: eliminate black line at the bottom
	    for(int i = 0; i < t; i++)
	    {
	        for(int j = 0; j < w; j++) img[i][j][h-1] = img[i][j][h-2];
	    }

        normalize_intensity(t, w, h, img);

        tu = new TimerUtil("motion correction");
	    motion_list = correct_motion(motion_param, t, w, h, img, range);
	    delete tu;

        *out_x = malloc_float1d(t);
        *out_y = malloc_float1d(t);
        float *px = *out_x;
        float *py = *out_y;
        for(auto m : motion_list)
        {
            *px++ = m.x;
            *py++ = m.y;
        }
    }
    if(motion_list.empty()) return -1;
    printf("(x, y) in [%.1f, %.1f] x [%.1f, %.1f]\n",
           range.min_x, range.max_x, range.min_y, range.max_y);

    if(skip_shading_correction)
    {
        printf("shading correction skipped\n");
    }
    else
    {
        tu = new TimerUtil("shading correction");
        correct_shading(shading_param, t, w, h, img, motion_list);
        delete tu;
    }
    
    int num_out = 0;
    float ***temporal = NULL;
    float ***spatial = NULL;
    if(skip_signal_extraction)
    {
        printf("signal extraction skipped\n");
    }
    else
    {
        tu = new TimerUtil("signal extraction");
        num_out = extract_signal(signal_param, t, w, h, img, motion_list, range, &temporal, &spatial);
        delete tu;
    }
    
    *out_image = malloc_float1d(t * h * w);
    copy3d_to_1d(t, w, h, img, *out_image);
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

