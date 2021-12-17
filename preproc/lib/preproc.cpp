#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "malloc_util.h"
#include "TimerUtil.h"
#include "motion.h"
#include "shading.h"
#include "signal.h"


static void normalize_intensity(int num_pages, int width, int height, float ***img)
{
    float *lumi = malloc_float1d(num_pages);
    float max = 0;
    
    for(int k = 0; k < num_pages; k++)
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
    for(int k = 0; k < num_pages; k++)
    {
        float scale = lumi[0] / lumi[k] / max;
        for(int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
        {
            img[k][i][j] *= scale;
        }
    }
}


static void print_help(char *command, float frames_per_sec,
                       motion_param_t  &motion_param,
                       shading_param_t &shading_param,
                       signal_param_t  &signal_param)
{
    printf("%s [options] in.tiff out_path\n", command);
    printf("\n");
    printf("  -dm <path>: disable motion correction (motion file required)\n");
    printf("  -dh       : disable shading correction\n");
    printf("  -db       : disable blood suppression\n");
    printf("  -ds       : disable signal extraction\n");
    printf("\n");
    printf("  -fr <float>: frame rate (%.1f Hz)\n", frames_per_sec);
    printf("\n");
    printf("  motion correction parameters\n");
    printf("  -ms <int>  : search size (%d pixels)\n", motion_param.search_size);
    printf("  -mp <int>  : patch size (%d pixels)\n", motion_param.patch_size);
    printf("  -mo <int>  : patch offset (%d pixels)\n", motion_param.patch_offset);
    printf("\n");
    printf("  shading correction parameters\n");
    printf("  -hw <int>  : window length (%d frames)\n", shading_param.period);
    printf("\n");
    printf("  signal extraction parameters\n");
    printf("  -sm <int>  : method PCA=0, max-median=1 (%d)\n", signal_param.method);
    printf("  -sw <int>  : window length (%d frames)\n", signal_param.period);
    printf("  -sc <float>: cutoff frequency (%.1f Hz)\n", signal_param.freq_max);
    printf("  -sp <int>  : PCA patch size (%d pixels)\n", signal_param.patch_size);
    printf("  -so <int>  : PCA patch offset (%d pixels)\n", signal_param.patch_offset);
    printf("  -ss <float>: max-median spatial smoothing (%.1f pixels)\n", signal_param.smooth_scale);
    exit(0);
}

int preprocess_cpu(int num_frames, int height, int width,
                   float *in_image,
                   float **out_image,
                   float **out_temporal)
                   //float **out_spatial)
{
    const int t = num_frames;
    const int h = height;
    const int w = width;
    
    motion_param_t motion_param;
    motion_param.level = 2;
    motion_param.search_size = 3;
    motion_param.patch_size = 10;
    motion_param.patch_offset = 7;
    motion_param.x_range = 0.7;
    motion_param.y_range = 1.0;
    motion_param.a_stdev = 1.0;
    motion_param.m_stdev = 3.0;
    motion_param.thresh_xy = 10.0;
    motion_param.length = 2000;
    motion_param.thresh_c = 1.0;

    shading_param_t shading_param;
    shading_param.period = 1000;

    signal_param_t signal_param;
    signal_param.method = 1;
    signal_param.period = 50;
    signal_param.freq_max = 0.0;
    signal_param.normalize = false;
    signal_param.downsample = true;
    signal_param.patch_size = 8;
    signal_param.patch_offset = 1;
    signal_param.smooth_scale = 3.0;

    bool skip_motion_correction = false;
    bool skip_shading_correction = false;
    bool skip_signal_extraction = false;
    float frames_per_sec = 1000.0;

    signal_param.frames_per_sec = frames_per_sec;

    float ***img = malloc_float3d(t, h, w);
    const int data_size = t * h * w * sizeof(float);
    memcpy(img[0][0], in_image, data_size);
    
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
    }
    if(motion_list.empty()) exit(1);
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
    float ***out = NULL;
    if(skip_signal_extraction)
    {
        printf("signal extraction skipped\n");
    }
    else
    {
        tu = new TimerUtil("signal extraction");
        out = extract_signal(signal_param, t, w, h, img, motion_list, range, &num_out);
        delete tu;
    }
    
    *out_image = malloc_float1d(t * h * w);
    memcpy(*out_image, img[0][0], data_size);
    free_float3d(img);
    if(out != NULL)
    {
        *out_temporal = malloc_float1d(num_out * h * w);
        memcpy(*out_temporal, out[0][0], num_out * h * w * sizeof(float));
        free_float3d(out);
    }
    return num_out;
}

