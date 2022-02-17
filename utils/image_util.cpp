#include <stddef.h>

void copy1d_to_3d(int num_frames, int width, int height, float *in, float ***out)
{
    size_t n = 0;
    for(int k = 0; k < num_frames; k++)
    for(int j = 0; j < height; j++)
    for(int i = 0; i < width; i++)
    {
        out[k][i][j] = in[n++];
    }
}

void copy3d_to_1d(int num_frames, int width, int height, float ***in, float *out)
{
    size_t n = 0;
    for(int k = 0; k < num_frames; k++)
    for(int j = 0; j < height; j++)
    for(int i = 0; i < width; i++)
    {
        out[n++] = in[k][i][j];
    }
}

