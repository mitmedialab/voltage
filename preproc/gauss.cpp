#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "malloc_util.h"


static float s2q(float stdev)
{
    if(stdev < 0.5)
    {
        fprintf(stderr, "too small scale: %f\n", stdev); exit(1);
    }
    else if(stdev < 2.5)
    {
        return 3.97156 - 4.14554 * sqrt(1.0 - 0.26891 * stdev);
    }
    else return 0.98711 * stdev - 0.9633;
}

void recursive_gauss_set_filter(float stdev, float *c)
{
    float q = s2q(stdev);
    float q2 = q * q;
    float q3 = q2 * q;
    
    float b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
    float b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
    float b2 = -1.4281 * q2 - 1.26661 * q3;
    float b3 = 0.422205 * q3;

    c[1] = b1 / b0;
    c[2] = b2 / b0;
    c[3] = b3 / b0;
    c[0] = 1.0 - c[1] - c[2] - c[3];
}

void recursive_gauss_apply_filter(float *c, int n, float *dat)
{
    for(int i = 3; i < n; i++)
    {
        dat[i] = c[0] * dat[i] + c[1] * dat[i-1] + c[2] * dat[i-2] + c[3] * dat[i-3];
    }
    for(int i = n - 4; i >= 0; i--)
    {
        dat[i] = c[0] * dat[i] + c[1] * dat[i+1] + c[2] * dat[i+2] + c[3] * dat[i+3];
    }
}

static void recursive_gauss_x(float *c, int w, int h, int n, float *t, float **in, float **out)
{
    int i, j;
    float on[128];
    
    for(j = 0; j < h; j++)
    {
        t[0] = t[1] = t[2] = in[0][j]; // clamp to edge
        for(i = 0; i < w; i++) t[i+3] = c[0] * in[i][j] + c[1] * t[i+2] + c[2] * t[i+1] + c[3] * t[i];
        for(i = w; i < w+n; i++) t[i+3] = c[0] * in[w-1][j] + c[1] * t[i+2] + c[2] * t[i+1] + c[3] * t[i]; // clamp to edge

        on[n] = on[n+1] = on[n+2] = t[w+n+2];

        for(i = n-1; i >= 0; i--) on[i] = c[0] * t[w+3+i] + c[1] * on[i+1] + c[2] * on[i+2] + c[3] * on[i+3];
    
        out[w-1][j] = c[0] * t[w+2] + c[1] * on[0]       + c[2] * on[1]       + c[3] * on[2];
        out[w-2][j] = c[0] * t[w+1] + c[1] * out[w-1][j] + c[2] * on[0]       + c[3] * on[1];
        out[w-3][j] = c[0] * t[w]   + c[1] * out[w-2][j] + c[2] * out[w-1][j] + c[3] * on[0];
        for(i = w-4; i >= 0; i--) out[i][j] = c[0] * t[i+3] + c[1] * out[i+1][j] + c[2] * out[i+2][j] + c[3] * out[i+3][j];
    }
}

static void recursive_gauss_y(float *c, int w, int h, int n, float *t, float **in, float **out)
{
    int i, j;
    float on[128];
    
    for(i = 0; i < w; i++)
    {
        t[0] = t[1] = t[2] = in[i][0]; // clamp to edge
        for(j = 0; j < h; j++) t[j+3] = c[0] * in[i][j] + c[1] * t[j+2] + c[2] * t[j+1] + c[3] * t[j];
        for(j = h; j < h+n; j++) t[j+3] = c[0] * in[i][h-1] + c[1] * t[j+2] + c[2] * t[j+1] + c[3] * t[j]; // clamp to edge

        on[n] = on[n+1] = on[n+2] = t[h+n+2];

        for(j = n-1; j >= 0; j--) on[j] = c[0] * t[h+3+j] + c[1] * on[j+1] + c[2] * on[j+2] + c[3] * on[j+3];
    
        out[i][h-1] = c[0] * t[h+2] + c[1] * on[0]       + c[2] * on[1]       + c[3] * on[2];
        out[i][h-2] = c[0] * t[h+1] + c[1] * out[i][h-1] + c[2] * on[0]       + c[3] * on[1];
        out[i][h-3] = c[0] * t[h]   + c[1] * out[i][h-2] + c[2] * out[i][h-1] + c[3] * on[0];
        for(j = h-4; j >= 0; j--) out[i][j] = c[0] * t[j+3] + c[1] * out[i][j+1] + c[2] * out[i][j+2] + c[3] * out[i][j+3];
    }
}

void recursive_gauss_apply_filter2d(float *c, float stdev, int w, int h, float **dat)
{
    const int n = (int)(stdev * 2.0);
    float *work = malloc_float1d((w > h ? w : h) + n + 3);
    float **tmp = malloc_float2d(w, h);

    recursive_gauss_x(c, w, h, n, work, dat, tmp);
    recursive_gauss_y(c, w, h, n, work, tmp, dat);

    free_float1d(work);
    free_float2d(tmp);
}

