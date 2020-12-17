
#ifndef __POSTPROCESS_HPP__
#define __POSTPROCESS_HPP__

float* get_signals(float *image, float *masks, int t, int h, int w, int n);
float* _exp_spread(float *img, int t, int h, int w);
#endif
