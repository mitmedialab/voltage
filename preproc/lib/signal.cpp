#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include "malloc_util.h"
#include "signal.h"


void recursive_gauss_set_filter(float stdev, float *c);
void recursive_gauss_apply_filter(float *c, int n, float *dat);
void recursive_gauss_apply_filter2d(float *c, float stdev, int w, int h, float **dat);


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
    // subtracting the average eigenval (lower bound for the largest eigenval)
    // didn't help either, as the average is smaller by an order of magnitude
    const double eigen_avg = trace / c;
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

static float ***temporal_filter(int start, int end,
                                int t, int w, int h, float *img, float stdev)
{
    const int len = end - start;
    float ***out = malloc_float3d(len, w, h);
    
    if(stdev < 0.5) // copy without filtering
    {
        for(int f = start; f < end; f++)
        {
            int k = f - start;
            size_t p = (size_t)f * w * h;
            for(int j = 0; j < h; j++)
            for(int i = 0; i < w; i++)
            {
                out[k][i][j] = img[p++];
            }
        }
    }
    else
    {
        float c[4];
        recursive_gauss_set_filter(stdev, c);

        const int margin = (int)(stdev * 2.0 + 3.5);
        const int ext_len = len + margin * 2;
        float *dat = malloc_float1d(ext_len);
        for(int i = 0; i < w; i++)
        for(int j = 0; j < h; j++)
        {
            for(int k = 0; k < ext_len; k++)
            {
                size_t f = start - margin + k;
                if(f < 0) f = 0;
                else if(f >= t) f = t - 1;
                dat[k] = img[(f * h + j) * w + i];
            }
            recursive_gauss_apply_filter(c, ext_len, dat);
            for(int k = 0; k < len; k++)
            {
                out[k][i][j] = dat[k + margin];
            }
        }
        free_float1d(dat);
    }
            
    return out;
}

int extract_signal(signal_param_t &param,
                   int num_pages, int width, int height, float *img,
                   float **temporal, float **spatial)

{
    const int method = param.method;
    const int period = param.period;
    const bool normalize = param.normalize;
    const bool downsample = param.downsample && method == 0;
    const int patch_size = param.patch_size;
    const int patch_offset = param.patch_offset;
    const float temp_stdev = param.freq_max <= 0 ? 0 : param.frames_per_sec / (M_PI * param.freq_max);
    const float space_stdev = param.smooth_scale;
    const int down = downsample ? 2 : 1;

    float **cnt = malloc_float2d(width, height);
    memset(cnt[0], 0, width * height * sizeof(float));
    float **one = malloc_float2d(patch_size, patch_size);
    for(int k = 0; k < patch_size * patch_size; k++) one[0][k] = 1.0;

    for(int i = 0; i <= width / down - patch_size; i += patch_offset)
    for(int j = 0; j <= height / down - patch_size; j += patch_offset)
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

    const int num_out = num_pages / period;
    const size_t num_pixels = width * height;
    float *out = malloc_float1d(num_out * num_pixels);
    float *avg = malloc_float1d(num_out * num_pixels);
    memset(out, 0, num_out * num_pixels * sizeof(float));
    memset(avg, 0, num_out * num_pixels * sizeof(float));
    
    #pragma omp parallel for
    for(int k = 0; k < num_out; k++)
    {
        std::vector<int> frames;
        int start_frame = k * period;
        int end_frame = start_frame + period;
        if(end_frame > num_pages) end_frame = num_pages;
        for(int f = start_frame; f < end_frame; f++)
        {
            //if(motion_list.valid[f]) frames.push_back(f - start_frame);
            frames.push_back(f - start_frame); // assume all the frames are valid
        }
        if((int)frames.size() < period / 2)
	    {
	        fprintf(stderr, "too few (%d) valid frames at frame %d, skipping\n",
	                (int)frames.size(), start_frame);
	        continue; // out[k] is all zero
	    }

        float ***buf = temporal_filter(start_frame, end_frame,
                                       num_pages, width, height, img, temp_stdev);

        // spatial signal extraction (average)
        size_t p;
        for(auto f : frames)
        {
            p = k * num_pixels;
            for(int j = 0; j < height; j++)
            for(int i = 0; i < width; i++)
            {
                avg[p++] += buf[f][i][j];
            }
        }
        p = k * num_pixels;
        for(int i = 0; i < width * height; i++)
        {
            avg[p++] /= frames.size();
        }

        // temporal signal extraction
        if(method == 0) // PCA
        {
            // below cannot be parallelized as-is because destination out[][][]
            // overlaps due to overlapping patches
            // need to separate it into multiple sets of non-overlapping patches
            //#pragma omp parallel for
            for(int i = 0; i <= width / down - patch_size; i += patch_offset)
            for(int j = 0; j <= height / down - patch_size; j += patch_offset)
            {
                float **pc = compute_principal_component(frames, i, j, patch_size, buf,
                                                         normalize, downsample);
                if(downsample)
                {
                    float **tmp = malloc_float2d(width, height);
                    memset(tmp[0], 0, width * height * sizeof(float));
                    upsample2x2(patch_size, pc, tmp, i, j);
                    p = k * num_pixels;
                    for(int v = 0; v < height; v++)
                    for(int u = 0; u < width; u++)
                    {
                        out[p++] += tmp[u][v];
                    }
                    free_float2d(tmp);
                }
                else
                {
                    for(int u = 0; u < patch_size; u++)
                    for(int v = 0; v < patch_size; v++)
                    {
                        int x = i + u;
                        int y = j + v;
                        p = k * num_pixels + y * width + x;
                        out[p] += fabsf(pc[u][v]);
                    }
                }
                
                free_float2d(pc);
            }

            p = k * num_pixels;
            for(int j = 0; j < height; j++)
            for(int i = 0; i < width; i++)
            {
                if(cnt[i][j] > 0) out[p++] /= cnt[i][j];
            }
        }
        else if(method == 1) // max-median
        {
            if(space_stdev >= 0.5)
            {
                float c[4];
                recursive_gauss_set_filter(space_stdev, c);
                for(auto f : frames)
                {
                    recursive_gauss_apply_filter2d(c, space_stdev, width, height, buf[f]);
                }
            }
            const size_t m = frames.size() / 2;
            p = k * num_pixels;
            for(int j = 0; j < height; j++)
            for(int i = 0; i < width; i++)
            {
                std::vector<float> v;
                float max = 0;
                for(auto f : frames)
                {
                    float val = buf[f][i][j];
                    v.push_back(val);
                    if(max < val) max = val;
                }
                std::nth_element(v.begin(), v.begin() + m, v.end());
                out[p++] = max - v[m];
            }
        }
        else // median-min
        {
            if(space_stdev >= 0.5)
            {
                float c[4];
                recursive_gauss_set_filter(space_stdev, c);
                for(auto f : frames)
                {
                    recursive_gauss_apply_filter2d(c, space_stdev, width, height, buf[f]);
                }
            }
            const size_t m = frames.size() / 2;
            p = k * num_pixels;
            for(int j = 0; j < height; j++)
            for(int i = 0; i < width; i++)
            {
                std::vector<float> v;
                float min = 1.0;
                for(auto f : frames)
                {
                    float val = buf[f][i][j];
                    v.push_back(val);
                    if(min > val) min = val;
                }
                std::nth_element(v.begin(), v.begin() + m, v.end());
                out[p++] = v[m] - min;
            }
        }

        free_float3d(buf);
	}

    free_float2d(cnt);
    
    *temporal = out;
    *spatial = avg;
    return num_out;
}

