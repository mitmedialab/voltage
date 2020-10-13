#ifndef __MOTION_HPP__
#define __MOTION_HPP__

#include <vector>

typedef struct
{
    float x;     // motion in X
    float y;     // motion in Y
    double corr; // correlation (normalized cross-correlation) score
    bool valid;  // validity of this motion estimation
} motion_t;

class motion_buffer_t {
public:
    float *in;
    float *out;
    float *gout;
    int t;
    int w;
    int h;
    motion_buffer_t(float *img, int num_pages, int height, int width) {
        in = img;
        t = num_pages;
        h = height;
        w = width;
    }
    ~motion_buffer_t() {
        // cudaFree(gout);
    }
};

typedef struct
{
    int level;        // multiresolution level (0, 1, 2...)
    int search_size;  // [-search_size, +search_size] pixels will be searched in X and Y at each level
    int patch_size;   // [-patch_size, +patch_size] pixels will be used as a patch in X and Y
    int patch_offset; // offset (both in X and Y) between adjacent patches
    
    float x_range;    // width  * x_range around center will be searched (1 means entire width)
    float y_range;    // heigth * y_range around center will be searched (1 means entire height)

    float a_stdev;    // standard deviation of acceleration [pixels/frame^2]
    float m_stdev;    // standard deviation of motion estimation error [pixels]
    float thresh_xy;  // motion spatial error threshold [pixels]
    int length;       // length of running median for correlation values
    float thresh_c;   // motion correlation error threshold

} motion_param_t;

typedef struct
{
    float min_x;
    float max_x;
    float min_y;
    float max_y;
} motion_range_t;


float * correct_motion(motion_param_t &param, motion_buffer_t *mbuf);

#endif