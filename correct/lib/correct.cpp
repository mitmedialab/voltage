#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <vector>

#include "malloc_util.h"
#include "image_util.h"
#include "TimerUtil.h"
#include "motion.h"
#include "shading.h"


static void normalize_intensity(int num_frames, int width, int height, float ***img)
{
    // equalize average frame intensities
    #pragma omp parallel for
    for(int k = 0; k < num_frames; k++)
    {
        float sum = 0;
        for(int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
        {
            sum += img[k][i][j];
        }
        const float scale = 1.0 / sum;
        for(int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
        {
            img[k][i][j] *= scale;
        }
    }
    
    // rescale intensities so that [min, max] = [0, 1]
    float min = img[0][0][0];
    float max = img[0][0][0];
    #pragma omp parallel for reduction(min: min) reduction(max: max)
    for(int k = 0; k < num_frames; k++)
    for(int i = 0; i < width; i++)
    for(int j = 0; j < height; j++)
    {
        if(min > img[k][i][j]) min = img[k][i][j];
        if(max < img[k][i][j]) max = img[k][i][j];
    }

    const float scale = 1.0 / (max - min);
    #pragma omp parallel for
    for(int k = 0; k < num_frames; k++)
    for(int i = 0; i < width; i++)
    for(int j = 0; j < height; j++)
    {
        img[k][i][j] = (img[k][i][j] - min) * scale;
    }
}


void correct_video_cpu(int num_frames, int height, int width,
                       float *in_image,
                       float **out_image,
                       float **out_x, float **out_y,
                       int normalize,
                       int motion_search_level, int motion_search_size,
                       int motion_patch_size, int motion_patch_offset,
                       int shading_period,
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
    motion_param.x_range = 1.0;
    motion_param.y_range = 1.0;
    //motion_param.thresh_xy = 10.0;
    motion_param.length = 1000;
    motion_param.thresh_c = 1.0;

    shading_param_t shading_param;
    shading_param.period = shading_period;

    float ***img = malloc_float3d(t, w, h);
    copy1d_to_3d(t, w, h, in_image, img);
    
    TimerUtil *tu;

    if(normalize)
    {
        tu = new TimerUtil("intensity normalization");
        normalize_intensity(t, w, h, img);
        delete tu;
    }

    std::vector<motion_t> motion_list;
    motion_range_t range;

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

    printf("(x, y) in [%.1f, %.1f] x [%.1f, %.1f]\n",
           range.min_x, range.max_x, range.min_y, range.max_y);

    tu = new TimerUtil("shading correction");
    correct_shading(shading_param, t, w, h, img, motion_list);
    delete tu;
    
    *out_image = malloc_float1d(t * h * w);
    copy3d_to_1d(t, w, h, img, *out_image);
    free_float3d(img);
}

