#ifndef __PRE_UNET_HPP__
#define __PRE_UNET_HPP__

#include <vector>

float * preprocess_unet(motion_buffer_t *mbuf, int magnification, int *t_out, int out_dim);

#endif