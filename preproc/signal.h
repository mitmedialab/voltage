#ifndef __SIGNAL_H__
#define __SIGNAL_H__


#include "motion.h"


typedef struct
{
    bool normalize;   // normalize data
    bool downsample;  // downsample image by half (effectively doubling patch_size/offset)
    int period;       // signal detection will be performed per this time period [frames]
    int patch_size;   // square patch of patch_size x patch_size pixels will be used
    int patch_offset; // offset (both in X and Y) between adjacent patches
} signal_param_t;


float ***extract_signal(signal_param_t &param,
                        int num_pages, int width, int height, float ***img,
                        std::vector<motion_t> motion, motion_range_t range, int *num_out);


#endif

