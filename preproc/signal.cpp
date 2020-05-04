#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "malloc_util.h"
#include "signal.h"


static float **compute_principal_component(std::vector<int> frames, int x, int y, int size, float ***img,
                                           bool normalize, bool downsample)
{
    const int r = (int)frames.size();
    const int c = size * size;
    double **data = malloc_double2d(r, c);
    for(int k = 0; k < r; k++)
    for(int i = 0; i < size; i++)
    for(int j = 0; j < size; j++)
    {
        int frame = frames[k];
        if(downsample)
        {
            double sum = 0;
            for(int u = 0; u < 2; u++) // 2x2 binning to reduce dimension
            for(int v = 0; v < 2; v++)
            {
                sum += img[frame][(x + i) * 2 + u][(y + j) * 2 + v];
            }
            data[k][size * j + i] = 0.25 * sum;
        }
        else
        {
            data[k][size * j + i] = img[frame][x + i][y + j];
        }        
    }
    
    // subtract mean for each column
    for(int j = 0; j < c; j++)
    {
        double sum = 0;
        for(int i = 0; i < r; i++)
        {
            sum += data[i][j];
        }
        double avg = sum / r;
        for(int i = 0; i < r; i++)
        {
            data[i][j] -= avg;
        }
    }

    if(normalize)
    {
        // normalize stdev for each column
        for(int j = 0; j < c; j++)
        {
            double sum = 0;
            for(int i = 0; i < r; i++)
            {
                sum += data[i][j] * data[i][j];
            }
            double stdev = sqrt(sum / r);
            for(int i = 0; i < r; i++)
            {
                data[i][j] /= stdev;
            }
        }
    }
    
    double *v = malloc_double1d(c);
    double *s = malloc_double1d(c);
    for(int i = 0; i < c; i++)
    {
        //v[i] = 1.0 / sqrt(c);
        v[i] = i == 0 ? 1.0 : 0;
    }
    
    double eigen = 0;
    for(int iter = 0; iter < 100; iter++)
    {
        memset(s, 0, c * sizeof(double));
        for(int i = 0; i < r; i++)
        {
            double dot = 0;
            for(int j = 0; j < c; j++)
            {
                dot += data[i][j] * v[j];
            }
            for(int j = 0; j < c; j++)
            {
                s[j] += dot * data[i][j];
            }
        }

        eigen = 0;
        for(int i = 0; i < c; i++)
        {
            eigen += v[i] * s[i];
        }

        double err = 0;
        double mag = 0;
        for(int i = 0; i < c; i++)
        {
            double dif = eigen * v[i] - s[i];
            err += dif * dif;
            mag += s[i] * s[i];
        }
        err = sqrt(err / c);
        mag = sqrt(mag);
        for(int i = 0; i < c; i++)
        {
            v[i] = s[i] / mag;
        }
        
        if(err < 1e-6)
        {
            //printf("iter %d: err %e, eigen %e\n", iter, err, eigen);
            break;
        }
    }

/*
    // normalizing eigenval by trace (sum of all eigenvals) didn't help
    double trace = 0;
    for(int i = 0; i < c; i++)
    {
        for(int j = 0; j < r; j++)
        {
            trace += data[j][i] * data[j][i];
        }
    }
*/    
    float **out = malloc_float2d(size, size);
    for(int i = 0; i < size; i++)
    for(int j = 0; j < size; j++)
    {
        out[i][j] = (float)(eigen * v[size * j + i]);
    }
    free_double2d(data);
    free_double1d(v);
    free_double1d(s);
    return out;
}

static void upsample2x2(int size, float **in, float **out, int x, int y)
{
    const float c0 = 9.0 / 16.0;
    const float c1 = 3.0 / 16.0;
    const float c2 = 1.0 / 16.0;
    
    for(int u = 0; u < size; u++)
    for(int v = 0; v < size; v++)
    {
#if 0
        int um = u > 0 ? u - 1 : 0;
        int up = u < size - 1 ? u + 1 : size - 1;
        int vm = v > 0 ? v - 1 : 0;
        int vp = v < size - 1 ? v + 1 : size - 1;
        
        float mm = fabsf(in[um][vm]);
        float mc = fabsf(in[um][v]);
        float mp = fabsf(in[um][vp]);
        float cm = fabsf(in[u][vm]);
        float cc = fabsf(in[u][v]);
        float cp = fabsf(in[u][vp]);
        float pm = fabsf(in[up][vm]);
        float pc = fabsf(in[up][v]);
        float pp = fabsf(in[up][vp]);
#else
        int um = u - 1;
        int up = u + 1;
        int vm = v - 1;
        int vp = v + 1;
        bool umo = um < 0;
        bool upo = up >= size;
        bool vmo = vm < 0;
        bool vpo = vp >= size;
        float mm = (umo || vmo) ? 0 : fabsf(in[um][vm]);
        float mc = (umo       ) ? 0 : fabsf(in[um][v]);
        float mp = (umo || vpo) ? 0 : fabsf(in[um][vp]);
        float cm = (       vmo) ? 0 : fabsf(in[u][vm]);
        float cc = fabsf(in[u][v]);
        float cp = (       vpo) ? 0 : fabsf(in[u][vp]);
        float pm = (upo || vmo) ? 0 : fabsf(in[up][vm]);
        float pc = (upo       ) ? 0 : fabsf(in[up][v]);
        float pp = (upo || vpo) ? 0 : fabsf(in[up][vp]);
#endif
        int xm = (x + u) * 2;
        int ym = (y + v) * 2;
        out[xm    ][ym    ] += c0 * cc + c1 * (mc + cm) + c2 * mm;
        out[xm + 1][ym    ] += c0 * cc + c1 * (cm + pc) + c2 * pm;
        out[xm    ][ym + 1] += c0 * cc + c1 * (mc + cp) + c2 * mp;
        out[xm + 1][ym + 1] += c0 * cc + c1 * (cp + pc) + c2 * pp;
    }
}

float ***extract_signal(signal_param_t &param,
                        int num_pages, int width, int height, float ***img,
                        std::vector<motion_t> motion, motion_range_t range, int *num_out)

{
    const bool normalize = param.normalize;
    const bool downsample = param.downsample;
    const int period = param.period;
    const int patch_size = param.patch_size;
    const int patch_offset = param.patch_offset;

    const int down = downsample ? 2 : 1;
    const int w_start = (int)ceil(range.min_x < 0 ? -range.min_x / down : 0);
    const int w_end = width / down - (int)ceil(range.max_x > 0 ? range.max_x / down : 0);
    const int h_start = (int)ceil(range.min_y < 0 ? -range.min_y / down : 0);
    const int h_end = height / down - (int)ceil(range.max_y > 0 ? range.max_y / down : 0);

    float **cnt = malloc_float2d(width, height);
    memset(cnt[0], 0, width * height * sizeof(float));
    float **one = malloc_float2d(patch_size, patch_size);
    for(int k = 0; k < patch_size * patch_size; k++) one[0][k] = 1.0;
    
    for(int i = w_start; i <= w_end - patch_size; i += patch_offset)
	for(int j = h_start; j <= h_end - patch_size; j += patch_offset)
	{
        if(downsample)
        {
            upsample2x2(patch_size, one, cnt, i, j);
        }
        else
        {
            for(int u = 0; u < patch_size; u++)
            for(int v = 0; v < patch_size; v++)
            {
                cnt[i + u][j + v] += 1.0;
            }
        }
    }
    free_float2d(one);

    const int n = (num_pages + period - 1) / period;
    float ***out = malloc_float3d(n, width, height);
    
    #pragma omp parallel for
	for(int k = 0; k < n; k++)
	{
        memset(out[k][0], 0, width * height * sizeof(float));

        std::vector<int> frames;
        int start_frame = k * period;
        int end_frame = start_frame + period;
        if(end_frame > num_pages) end_frame = num_pages;
        for(int f = start_frame; f < end_frame; f++)
        {
            if(motion[f].valid) frames.push_back(f);
        }
        if((int)frames.size() < period / 2)
	    {
	        fprintf(stderr, "too few (%d) valid frames at frame %d, skipping\n",
	                (int)frames.size(), start_frame);
	        continue; // out[k] is all zero
	    }
   
        // below cannot be parallelized as-is because destination out[][][]
        // overlaps due to overlapping patches
        // need to separate it into multiple sets of non-overlapping patches
        //#pragma omp parallel for
	    for(int i = w_start; i <= w_end - patch_size; i += patch_offset)
	    for(int j = h_start; j <= h_end - patch_size; j += patch_offset)
	    {
	        float **pc = compute_principal_component(frames, i, j, patch_size, img,
	                                                 normalize, downsample);
	        if(downsample)
	        {
	            upsample2x2(patch_size, pc, out[k], i, j);
	        }
	        else
	        {
	            for(int u = 0; u < patch_size; u++)
	            for(int v = 0; v < patch_size; v++)
	            {
	                out[k][i + u][j + v] += fabsf(pc[u][v]);
	            }
	        }
	            
	        free_float2d(pc);
	    }
	    
	    for(int i = 0; i < width; i++)
	    for(int j = 0; j < height; j++)
	    {
	        if(cnt[i][j] > 0) out[k][i][j] /= cnt[i][j];
	    }
	}

    free_float2d(cnt);
    
    *num_out = n;
    return out;
}


