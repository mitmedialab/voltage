#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "blood.h"
#include "malloc_util.h"


extern "C" {
void cdft(int n, int isgn, double *a, int *ip, double *w);
}


static int power2_no_less(int in)
{
    int out = 1;
    while(out < in) out *= 2;
    return out;
}

blood_out *suppress_blood(blood_param_t &param, int num_pages, int width, int height, float *img)
{    
    const int period = param.period;
    const int fft_len = power2_no_less(period);
    const int num_freq = fft_len / 2; // ignore the highest frequency
    
	// float ***out = malloc_float3d(3 + num_freq, width, height);
    
    blood_out *bld = (blood_out *) malloc(sizeof(blood_out));
    bld->mask = (float *) malloc (height * width * sizeof(float));
    bld->avg_lo = (float *) malloc (height * width * sizeof(float));
    bld->avg_hi = (float *) malloc (height * width * sizeof(float));
    bld->spctr = (float *) malloc (num_freq * height * width * sizeof(float));


    // float *out = (float *) malloc ( (3 + num_freq) * height * width * sizeof(float));
    memset(bld->mask, 0, height * width * sizeof(float));
    memset(bld->avg_lo, 0, height * width * sizeof(float));
    memset(bld->avg_hi, 0, height * width * sizeof(float));
    memset(bld->spctr, 0, num_freq * height * width * sizeof(float));


    // window function (Hann window)
    float *win = (float *) malloc(period * sizeof(float));
    for(int m = 0; m < period; m++) {
        float s = sin(M_PI * m / (period - 1));
        win[m] = s * s;
    }
    
    #pragma omp parallel for
    for(int i = 0; i < height; i++) {
        int *ip = (int *) malloc((2 + (int)sqrt(fft_len + 0.5)) * sizeof(int));
        ip[0] = 0;
        double *x = (double *) malloc((2 * fft_len) * sizeof(double)); 
        double *work = (double *) malloc((fft_len / 2) * sizeof(double));

        for(int j = 0; j < width; j++)
        {
            for(int k = 0; k + period <= num_pages; k += period)
            {
                memset(x, 0, 2 * fft_len * sizeof(double));
                for(int m = 0; m < period; m++)
                {
                    x[m * 2] = *loc3D(img, num_pages, height, width, (k + m), i, j) * win[m];
                }
                cdft(2 * fft_len, 1, x, ip, work);
                for(int f = 0; f < num_freq; f++)
                {
                    double mag = sqrt(x[f * 2] * x[f * 2] + x[f * 2 + 1] * x[f * 2 + 1]);
                    if(*loc3D(bld->spctr, num_freq, height, width, f, i, j) < mag) 
                        *loc3D(bld->spctr, num_freq, height, width, f, i, j) = mag;
                }
            }
        }
        free(ip);
        free(x);
        free(work);
    }
    free(win);
    
    
    const float fps = param.frames_per_sec;
    const int f_min = (int)(param.freq_min * fft_len / fps);
    const int f_max = (int)(param.freq_max * fft_len / fps);
    const float thresh = param.thresh;
    
    #pragma omp parallel for
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++)
        {
            // average low frequency magnitude (blood flow + others)
            for(int f = f_min; f < f_max; f++)
            {
                *loc2D(bld->avg_lo, height, width, i, j) += *loc3D(bld->spctr, num_freq, height, width, f, i, j);
            }
            *loc2D(bld->avg_lo, height, width, i, j) /= f_max - f_min;
            
            // average high frequency magnitude (other things than blood)
            for(int f = f_max; f < num_freq; f++)
            {
                *loc2D(bld->avg_hi, height, width, i, j) += *loc3D(bld->spctr, num_freq, height, width, f, i, j);
            }
            *loc2D(bld->avg_hi, height, width, i, j) /= num_freq - f_max;
            
            float s = (*loc2D(bld->avg_hi, height, width, i, j) > *loc2D(bld->avg_lo, height, width, i, j) * thresh) ? 1.0 : 0.1;
            *loc2D(mask, height, width, i, j) = s;
            for(int k = 0; k < num_pages; k++)
            {
                *loc3D(img, num_pages, height, width, k, i, j) *= s;
            }
        }
    }
    
    return bld;
}

