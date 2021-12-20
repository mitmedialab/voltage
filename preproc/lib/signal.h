#ifndef __SIGNAL_H__
#define __SIGNAL_H__


#include "motion.h"


typedef struct
{
    int method;       // 0: PCA, 1: max-median
    int period;       // signal detection will be performed per this time period [frames]
    float frames_per_sec; // video frame rate
    float freq_max;   // cutoff frequency [Hz] for temporal Gaussian low-pass filter (2 sigma)
    // PCA parameters
    bool normalize;   // normalize data
    bool downsample;  // downsample image by half (effectively doubling patch_size/offset)
    int patch_size;   // square patch of patch_size x patch_size pixels will be used
    int patch_offset; // offset (both in X and Y) between adjacent patches
    // Max-median parameters
    float smooth_scale; // spatial Gaussian low-pass filter scale (standard deviation, 1 sigma)
} signal_param_t;


int extract_signal(signal_param_t &param,
                   int num_pages, int width, int height, float ***img,
                   std::vector<motion_t> motion, motion_range_t range,
                   float ****temporal, float ****spatial);


#endif

