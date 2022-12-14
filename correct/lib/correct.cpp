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


static void normalize_intensity(size_t num_frames, size_t width, size_t height, float *img)
{
    const size_t num_pixels = width * height;

    // equalize average frame intensities
    #pragma omp parallel for
    for(size_t t = 0; t < num_frames; t++)
    {
        float sum = 0;
        for(size_t i = 0; i < num_pixels; i++)
        {
            sum += img[t * num_pixels + i];
        }
        const float scale = 1.0 / sum;
        for(size_t i = 0; i < num_pixels; i++)
        {
            img[t * num_pixels + i] *= scale;
        }
    }

    // rescale intensities so that [min, max] = [0, 1]
    float min = img[0];
    float max = img[0];
    #pragma omp parallel for reduction(min: min) reduction(max: max)
    for(size_t t = 0; t < num_frames; t++)
    for(size_t i = 0; i < num_pixels; i++)
    {
        float val = img[t * num_pixels + i];
        if(min > val) min = val;
        if(max < val) max = val;
    }

    const float scale = 1.0 / (max - min);
    #pragma omp parallel for
    for(size_t t = 0; t < num_frames; t++)
    for(size_t i = 0; i < num_pixels; i++)
    {
        img[t * num_pixels + i] = (img[t * num_pixels + i] - min) * scale;
    }
}


void correct_video_cpp(int num_frames, int height, int width,
                       float *in_image,
                       float **out_image,
                       float **out_x, float **out_y,
                       int normalize,
                       int motion_search_level, int motion_search_size,
                       int motion_patch_size, int motion_patch_offset,
                       int shading_period,
                       int use_gpu, int num_threads)
{
    if(num_threads > 0)
    {
        omp_set_num_threads(num_threads);
    }
    
    const size_t t = num_frames;
    const size_t h = height;
    const size_t w = width;
    
    motion_param_t motion_param;
    motion_param.level = motion_search_level;
    motion_param.search_size = motion_search_size;
    motion_param.patch_size = motion_patch_size;
    motion_param.patch_offset = motion_patch_offset;
    motion_param.x_range = 0.7;//1.0;
    motion_param.y_range = 1.0;
    //motion_param.thresh_xy = 10.0;
    motion_param.length = 1000;
    motion_param.thresh_c = 1.0;

    shading_param_t shading_param;
    shading_param.period = shading_period;

    
    TimerUtil *tu;

    if(normalize)
    {
        tu = new TimerUtil("intensity normalization");
        normalize_intensity(t, w, h, in_image);
        delete tu;
    }

    std::vector<motion_t> motion_list;
    *out_image = malloc_float1d(t * h * w);

    tu = new TimerUtil("motion correction");
    if(use_gpu)
    {
        motion_list = correct_motion_gpu(motion_param, t, w, h,
                                         in_image, *out_image);
    }
    else
    {
        float ***tmp = malloc_float3d(t, w, h);
        copy1d_to_3d(t, w, h, in_image, tmp);
        motion_list = correct_motion(motion_param, t, w, h, tmp);
        copy3d_to_1d(t, w, h, tmp, *out_image);
        free_float3d(tmp);
    }
    delete tu;

    float min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    *out_x = malloc_float1d(t);
    *out_y = malloc_float1d(t);
    float *px = *out_x;
    float *py = *out_y;
    for(auto m : motion_list)
    {
        *px++ = m.x;
        *py++ = m.y;
        if(min_x > m.x) min_x = m.x;
        if(max_x < m.x) max_x = m.x;
        if(min_y > m.y) min_y = m.y;
        if(max_y < m.y) max_y = m.y;
    }
    printf("(x, y) in [%.1f, %.1f] x [%.1f, %.1f]\n", min_x, max_x, min_y, max_y);

    tu = new TimerUtil("shading correction");
    correct_shading(shading_param, t, w, h, *out_image, motion_list);
    delete tu;
}

