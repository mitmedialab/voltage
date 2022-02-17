#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "malloc_util.h"
#include "motion.h"


static float **downsample_image(int w, int h, float **img)
{
    float **out = malloc_float2d(w, h);
    
    #pragma omp parallel for
    for(int i = 0; i < w; i++)
    for(int j = 0; j < h; j++)
    {
        int i0 = i * 2;
        int i1 = i0 + 1;
        int j0 = j * 2;
        int j1 = j0 + 1;
        out[i][j] = (img[i0][j0] + img[i1][j0] + img[i0][j1] + img[i1][j1]) * 0.25f;
    }
    
    return out;
}

static double normalized_cross_correlation(int w, int h, int cx, int cy, int patch_size,
                                           float **ref, float **tgt, int x, int y)
{
    double ref_sum = 0, tgt_sum = 0;
    double ref_sum2 = 0, tgt_sum2 = 0;
    double cross = 0;
    int n = 0;
    int start_i = cx - patch_size; if(start_i < 0) start_i = 0;
    int end_i   = cx + patch_size; if(end_i  >= w) end_i   = w - 1;
    int start_j = cy - patch_size; if(start_j < 0) start_j = 0;
    int end_j   = cy + patch_size; if(end_j  >= h) end_j   = h - 1;
    for(int i = start_i; i <= end_i; i++)
    for(int j = start_j; j <= end_j; j++)
    {
        int u = i + x;
        int v = j + y;
        if(u < 0) u = 0; else if(u >= w) u = w - 1; // probably better not clamp
        if(v < 0) v = 0; else if(v >= h) v = h - 1;
        
        ref_sum += ref[i][j];
        ref_sum2 += ref[i][j] * ref[i][j];
        tgt_sum += tgt[u][v];
        tgt_sum2 += tgt[u][v] * tgt[u][v];
        cross += ref[i][j] * tgt[u][v];
        n++;
    }
    ref_sum /= n;
    ref_sum2 /= n;
    tgt_sum /= n;
    tgt_sum2 /= n;
    cross /= n;
    double covar = cross - ref_sum * tgt_sum;
    double ref_var = ref_sum2 - ref_sum * ref_sum; if(ref_var < 0) ref_var = 0;
    double tgt_var = tgt_sum2 - tgt_sum * tgt_sum; if(tgt_var < 0) tgt_var = 0;
    //return covar / sqrt(ref_var * tgt_var + 1e-6); // raw NCC
    //return ref_sum * covar / sqrt(ref_var * tgt_var + 1e-6); // scaled by average intensity of reference patch
    return ref_sum2 * covar / sqrt(ref_var * tgt_var + 1e-6); // scaled by average squared intensity of reference patch
}

static double get_peak(int search_size, double **v, bool subpixel, float *x, float *y)
{
    const int size2 = search_size * 2;
    double max_corr = 0;
    int max_i = 0, max_j = 0;
    for(int i = 0; i <= size2; i++)
    for(int j = 0; j <= size2; j++)
    {
        if(max_corr < v[i][j])
        {
            max_corr = v[i][j];
            max_i = i;
            max_j = j;
        }
    }
    
    // check if max_i and max_j is within [1, search_size*2-1]
    // otherwise it's likely that the true minimum is outside of the search space
    // (but proceed anyway even if so)
    if(max_i == 0 || max_i == size2 || max_j == 0 || max_j == size2)
    {
        fprintf(stderr, "peak is at the edge of search space: (%d, %d)\n", max_i, max_j);
    }

    if(subpixel)
    {
        int im = max_i - 1; if(im < 0) im = 0;
        int ic = max_i;
        int ip = max_i + 1; if(ip >= search_size * 2) ip = search_size * 2;
        int jm = max_j - 1; if(jm < 0) jm = 0;
        int jc = max_j;
        int jp = max_j + 1; if(jp >= search_size * 2) jp = search_size * 2;
        
        double a = (v[im][jm] - 2 * v[ic][jm] + v[ip][jm] + v[im][jc] - 2 * v[ic][jc] + v[ip][jc] + v[im][jp] - 2 * v[ic][jp] + v[ip][jp]) / 6;
        double b = (v[im][jm] - v[ip][jm] - v[im][jp] + v[ip][jp]) / 4;
        double c = (v[im][jm] + v[ic][jm] + v[ip][jm] - 2 * v[im][jc] - 2 * v[ic][jc] - 2 * v[ip][jc] + v[im][jp] + v[ic][jp] + v[ip][jp]) / 6;
        double d = (-v[im][jm] + v[ip][jm] - v[im][jc] + v[ip][jc] - v[im][jp] + v[ip][jp]) / 6;
        double e = (-v[im][jm] - v[ic][jm] - v[ip][jm] + v[im][jp] + v[ic][jp] + v[ip][jp]) / 6;
        double f = (-v[im][jm] + 2 * v[ic][jm] - v[ip][jm] + 2 * v[im][jc] + 5 * v[ic][jc] + 2 * v[ip][jc] - v[im][jp] + 2 * v[ic][jp] - v[ip][jp]) / 9;
        double denom = 4 * a * c - b * b;
        double px = (b * e - 2 * c * d) / denom;
        double py = (b * d - 2 * a * e) / denom;
        if(-1 < px && px < 1 && -1 < py && py < 1)
        {
            *x = (float)(max_i - search_size + px);
            *y = (float)(max_j - search_size + py);
            return a * px * px + b * px * py + c * py * py + d * px + e * py + f;
        }
        else
        {
            fprintf(stderr, "subpixel estimate exceeded one pixel: (%.1f, %.1f)\n", px, py);
            // revert to pixel-unit estimate
            *x = max_i - search_size;
            *y = max_j - search_size;
            return max_corr;
        }
    }
    else
    {
        *x = max_i - search_size;
        *y = max_j - search_size;
        return max_corr;
    }
}

static double estimate_motion_recursive(int level, bool top,
                                        int search_size, int patch_size, int patch_offset,
                                        float x_range, float y_range,
                                        int w, int h, float **ref, float **tgt, float *x, float *y)
{
    int bx = 0, by = 0;
    if(level > 0)
    {
        const int patch_size_half = (patch_size + 1) / 2;
        const int patch_offset_half = (patch_offset + 1) / 2;
        const int w_half = w / 2;
        const int h_half = h / 2;
        float **ref_half = downsample_image(w_half, h_half, ref);
        float **tgt_half = downsample_image(w_half, h_half, tgt);
        float x_half, y_half;
        estimate_motion_recursive(level - 1, false,
                                  search_size, patch_size_half, patch_offset_half,
                                  x_range, y_range,
                                  w_half, h_half, ref_half, tgt_half, &x_half, &y_half);
        free_float2d(ref_half);
        free_float2d(tgt_half);
        bx = (int)(2 * x_half);
        by = (int)(2 * y_half);
    }
    
    int x_space = (int)(w * x_range * 0.5) - patch_size - search_size;
    int y_space = (int)(h * y_range * 0.5) - patch_size - search_size;
    int patch_num_l = (x_space + bx) / patch_offset;
    int patch_num_r = (x_space - bx) / patch_offset;
    int patch_num_b = (y_space + by) / patch_offset;
    int patch_num_t = (y_space - by) / patch_offset;
    if(patch_num_l < 0) patch_num_l = 0;
    if(patch_num_r < 0) patch_num_r = 0;
    if(patch_num_b < 0) patch_num_b = 0;
    if(patch_num_t < 0) patch_num_t = 0;
    
    const int size = search_size * 2 + 1;
    double **corr = malloc_double2d(size, size);
    memset(corr[0], 0, size * size * sizeof(double));

    #pragma omp parallel for
    for(int k = 0; k < size * size; k++)
    {
        int u = k % size;
        int v = k / size;
        int s = u - search_size + bx;
        int t = v - search_size + by;
        for(int i = -patch_num_l; i <= patch_num_r; i++)
        for(int j = -patch_num_b; j <= patch_num_t; j++)
        {
            int cx = w / 2 + i * patch_offset;
            int cy = h / 2 + j * patch_offset;
            corr[u][v] += normalized_cross_correlation(w, h, cx, cy, patch_size, ref, tgt, s, t);
        }
    }

    float px, py;
    double ret = get_peak(search_size, corr, top, &px, &py);
    *x = bx + px;
    *y = by + py;

    free_double2d(corr);
    return ret;
}

static double estimate_motion(motion_param_t &param,
                              int w, int h, float **ref, float **tgt, float *x, float *y)
{
    return estimate_motion_recursive(param.level, true,
                                     param.search_size, param.patch_size, param.patch_offset,
                                     param.x_range, param.y_range,
                                     w, h, ref, tgt, x, y);
}

static float median(std::vector<float> &v)
{
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

static float sample_bilinear(float **dat, int w, int h, float x, float y)
{
    int xm1, xp1, ym1, yp1;
    float frac_x, frac_y;

    xm1 = (int)x;
    ym1 = (int)y;
    xp1 = xm1 + 1;
    yp1 = ym1 + 1;
    frac_x = x - xm1;
    frac_y = y - ym1;
    
    if(xm1 < 0) xm1 = 0;
    else if(xm1 >= w) xm1 = w - 1;
    if(ym1 < 0) ym1 = 0;
    else if(ym1 >= h) ym1 = h - 1;
    if(xp1 < 0) xp1 = 0;
    else if(xp1 >= w) xp1 = w - 1;
    if(yp1 < 0) yp1 = 0;
    else if(yp1 >= h) yp1 = h - 1;

    return (dat[xm1][ym1] * (1 - frac_x) + dat[xp1][ym1] * frac_x) * (1 - frac_y)
         + (dat[xm1][yp1] * (1 - frac_x) + dat[xp1][yp1] * frac_x) * frac_y;
}

static void apply_motion(int w, int h, float **in, float x, float y, float **out)
{
    #pragma omp parallel for
    for(int i = 0; i < w; i++)
    for(int j = 0; j < h; j++)
    {
        out[i][j] = sample_bilinear(in, w, h, i+x, j+y);
    }
}

std::vector<motion_t> correct_motion(motion_param_t &param,
                                     int num_pages, int width, int height, float ***img,
                                     motion_range_t &range)
{    
    //const float thresh_xy = param.thresh_xy;
    const float thresh_c = param.thresh_c;
    const int length = param.length;
    int c_count = 0;
    int c_index = 0;
    float *c_ring = malloc_float1d(length); // ring buffer
    float med = 0;
    
    std::vector<motion_t> motion_list;
    float **out = malloc_float2d(width, height);
	float min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    
    // first frame
    {
        motion_t m;
        m.x = 0;
        m.y = 0;
        m.corr = 0;
        m.valid = true;
        motion_list.push_back(m);
    }
        
	for(int i = 1; i < num_pages; i++)
	{
	    // motion estimation
	    float x, y;
	    double c = estimate_motion(param, width, height, img[0], img[i], &x, &y);
                
        // check anomaly of correlation coefficient
        // ToDo: we could do the same for x and y values
        {
            c_ring[c_index] = c;
            c_index = (c_index + 1) % length;
            if(c_count < length) c_count++;
            
            // running median is much better than running average
            std::vector<float> c_vec(c_ring, c_ring + c_count);
            med = median(c_vec);
        }
        float dif_c = fabsf(med - c);
        bool valid_c = (c_count < 10) || (dif_c < thresh_c * med);

        // shift image (regardless of confidence in motion estimation
        // as otherwise it would be harder to tell what went wrong)
   	    apply_motion(width, height, img[i], x, y, out);
   	    memcpy(img[i][0], out[0], width * height * sizeof(float));

        // save motion vector
        motion_t m;
        m.x = x;
        m.y = y;
        m.corr = c;
        if(valid_c)
        {
            m.valid = true;
            if(min_x > x) min_x = x;
            if(max_x < x) max_x = x;
            if(min_y > y) min_y = y;
            if(max_y < y) max_y = y;
        }
        else
        {
            m.valid = false;
        }
        motion_list.push_back(m);
	}
    
    free_float1d(c_ring);
    free_float2d(out);
    
    range.min_x = min_x;
    range.max_x = max_x;
    range.min_y = min_y;
    range.max_y = max_y;
    return motion_list;
}

