#ifndef __MOTION_H__
#define __MOTION_H__

#include <vector>


typedef struct
{
    float x;     // motion in X
    float y;     // motion in Y
    double corr; // correlation (normalized cross-correlation) score
    bool valid;  // validity of this motion estimation
} motion_t;

typedef struct
{
    int level;        // multiresolution level (0, 1, 2...)
    int search_size;  // [-search_size, +search_size] pixels will be searched in X and Y at each level
    int patch_size;   // [-patch_size, +patch_size] pixels will be used as a patch in X and Y
    int patch_offset; // offset (both in X and Y) between adjacent patches
    
    float x_range;    // width  * x_range around center will be searched (1 means entire width)
    float y_range;    // heigth * y_range around center will be searched (1 means entire height)

    //float thresh_xy;  // motion spatial error threshold [pixels]
    int length;       // length of running median for correlation values
    float thresh_c;   // motion correlation error threshold

} motion_param_t;


std::vector<motion_t> correct_motion(motion_param_t &param,
                                     int num_pages, int width, int height, float ***img);

std::vector<motion_t> correct_motion_gpu(motion_param_t &param,
                                        int num_pages, int width, int height,
                                        int num_frames_per_batch,
                                        float *in_image, float *out_image);

#endif

