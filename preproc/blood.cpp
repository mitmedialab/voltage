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

float ***suppress_blood(blood_param_t &param, int num_pages, int width, int height, float ***img)
{    
    const int period = param.period;
    const int fft_len = power2_no_less(period);
    const int num_freq = fft_len / 2; // ignore the highest frequency
    
	float ***out = malloc_float3d(3 + num_freq, width, height);
    memset(out[0][0], 0, (3 + num_freq) * width * height * sizeof(float));
    float **mask = out[0]; // mask for suppressing blood regions
    float **avg_lo = out[1]; // average low frequency magnitude (blood flow)
    float **avg_hi = out[2]; // average high frequency magnitude
    float ***spctr = &out[3]; // temporal frequency spectrum

    // window function (Hann window)
    float *win = malloc_float1d(period);
    for(int m = 0; m < period; m++)
    {
        float s = sin(M_PI * m / (period - 1));
        win[m] = s * s;
    }
    
    #pragma omp parallel for
    for(int i = 0; i < width; i++)
    {
        int *ip = malloc_int1d(2 + (int)sqrt(fft_len + 0.5));
        ip[0] = 0;
        double *x = malloc_double1d(2 * fft_len); 
        double *work = malloc_double1d(fft_len / 2);

        for(int j = 0; j < height; j++)
        {
            for(int k = 0; k + period <= num_pages; k += period)
            {
                memset(x, 0, 2 * fft_len * sizeof(double));
                for(int m = 0; m < period; m++)
                {
                    x[m * 2] = img[k + m][i][j] * win[m];
                }
                cdft(2 * fft_len, 1, x, ip, work);
                for(int f = 0; f < num_freq; f++)
                {
                    double mag = sqrt(x[f * 2] * x[f * 2] + x[f * 2 + 1] * x[f * 2 + 1]);
                    if(spctr[f][i][j] < mag) spctr[f][i][j] = mag;
                }
            }
        }
        free_int1d(ip);
        free_double1d(x);
        free_double1d(work);
    }
    free_float1d(win);
    
    
    const float fps = param.frames_per_sec;
    const int f_min = (int)(param.freq_min * fft_len / fps);
    const int f_max = (int)(param.freq_max * fft_len / fps);
    const float thresh = param.thresh;
    
    #pragma omp parallel for
    for(int i = 0; i < width; i++)
    for(int j = 0; j < height; j++)
    {
        // average low frequency magnitude (blood flow + others)
        for(int f = f_min; f < f_max; f++)
        {
            avg_lo[i][j] += spctr[f][i][j];
        }
        avg_lo[i][j] /= f_max - f_min;
        
        // average high frequency magnitude (other things than blood)
        for(int f = f_max; f < num_freq; f++)
        {
            avg_hi[i][j] += spctr[f][i][j];
        }
        avg_hi[i][j] /= num_freq - f_max;
        
        float s = (avg_hi[i][j] > avg_lo[i][j] * thresh) ? 1.0 : 0.1;
        mask[i][j] = s;
        for(int k = 0; k < num_pages; k++)
        {
            img[k][i][j] *= s;
        }
    }
    
    return out;
}

