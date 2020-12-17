#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

#include "wrapper.hpp"

void postprocess_frames(float *image, float *masks, int t, int h, int w, int n, float **sig)
{
	*sig = get_signals(image, masks, t, h, w, n);
}

void exp_spread(float *image, int t, int h, int w, float **out)
{
	*out = _exp_spread(image, t, h, w);
}
