#ifndef __IMAGE_UTIL_H__
#define __IMAGE_UTIL_H__

void copy1d_to_3d(int num_frames, int width, int height, float *in, float ***out);
void copy3d_to_1d(int num_frames, int width, int height, float ***in, float *out);

#endif

