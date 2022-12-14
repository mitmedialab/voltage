#ifndef __SHADING_H__
#define __SHADING_H__

#include "motion.h"


typedef struct
{
    int period; // time period [frames] for modeling shading
} shading_param_t;


void correct_shading(shading_param_t &param,
                     int num_pages, int width, int height, float *img,
                     std::vector<motion_t> motion);

#endif

