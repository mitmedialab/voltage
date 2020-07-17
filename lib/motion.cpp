#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
// #include <cuda_runtime_api.h>
// #include <cuda.h>
#include "wrapper.hpp"
#include "motion.hpp"
#include "KalmanFilter.h"

#define checkCudaErrors(func) {                                        \
    cudaError_t error = func;                                          \
    if (error != 0) {                                                  \
        printf("Cuda failure Error: %s", cudaGetErrorString(error));  \
        fflush(stdout);                                                 \
    }                                                                  \
}


// __global__
static void downsample_image(int wh, int hh, int w, int h, float *img, float *out)
{
    
    #pragma omp parallel for
    for(int i = 0; i < hh; i++) {
        for(int j = 0; j < wh; j++) {

            int i0 = i * 2;
            int i1 = i0 + 1;
            int j0 = j * 2;
            int j1 = j0 + 1;
            
            *loc2D(out, hh, wh, i, j) = ( *loc2D(img, h, w, i0, j0) + \
                                        *loc2D(img, h, w, i1, j0) + \
                                        *loc2D(img, h, w, i0, j1) + \
                                        *loc2D(img, h, w, i1, j1) ) * 0.25f;
        }
    }
}

static double normalized_cross_correlation(int w, int h, int cx, int cy, int patch_size, float *ref, float *tgt, int x, int y)
{
    double ref_sum = 0;
    double tgt_sum = 0;
    double ref_sum2 = 0;
    double tgt_sum2 = 0;
    double cross = 0;
    int n = 0;
    bool show = false;
    
    int start_i = cy - patch_size; 
    if(start_i < 0) 
        start_i = 0;
    int end_i = cy + patch_size; 
    if(end_i  >= h) 
        end_i = h - 1;

    int start_j = cx - patch_size; 
    if(start_j < 0) 
        start_j = 0;
    int end_j   = cx + patch_size; 
    if(end_j  >= w) 
        end_j = w - 1;

    for(int i = start_i; i <= end_i; i++) {
        for(int j = start_j; j <= end_j; j++) {
            int u = i + y;
            int v = j + x;

            if(u < 0) 
                u = 0; 
            else if(u >= h) 
                u = h - 1; // probably better not clamp

            if(v < 0)
                v = 0;
            else if(v >= w)
                v = w - 1;

            ref_sum += *loc2D(ref, h, w, i, j);
            ref_sum2 += *loc2D(ref, h, w, i, j) * *loc2D(ref, h, w, i, j);

            tgt_sum += *loc2D(tgt, h, w, u, v);
            tgt_sum2 += *loc2D(tgt, h, w, u, v) * *loc2D(tgt, h, w, u, v);

            cross += *loc2D(ref, h, w, i, j) * *loc2D(tgt, h, w, u, v);
            n++;
        }
    }

    ref_sum /= n;
    ref_sum2 /= n;
    tgt_sum /= n;
    tgt_sum2 /= n;
    cross /= n;

    double covar = cross - ref_sum * tgt_sum;
    
    double ref_var = ref_sum2 - ref_sum * ref_sum;
    if(ref_var < 0)
        ref_var = 0;

    double tgt_var = tgt_sum2 - tgt_sum * tgt_sum;
    if(tgt_var < 0)
        tgt_var = 0;

    // scaled by average intensity of reference patch
    return ref_sum * covar / sqrt(ref_var * tgt_var + 1e-6);    

}

static double get_peak(int search_size, double *v, bool subpixel, float *x, float *y)
{
    const int size2 = search_size * 2;
    const int size = search_size * 2 + 1;
    double max_corr = 0;
    int max_i = 0;
    int max_j = 0;

    for(int i = 0; i <= size2; i++) {
        for(int j = 0; j <= size2; j++) {

            if(max_corr < *loc2D(v, size, size, i, j)) {
                max_corr = *loc2D(v, size, size, i, j);
                max_i = i;
                max_j = j;
            }
        }
    }

    if(max_i == 0 || max_i == size2 || max_j == 0 || max_j == size2) {
        // fprintf(stderr, "peak is at the edge of search space: (%d, %d)\n", max_i, max_j);
    }


    if(subpixel) {
        int im = max_i - 1; 
        if(im < 0) 
            im = 0;
        int ic = max_i;
        int ip = max_i + 1; 
        if(ip >= search_size * 2) 
            ip = search_size * 2;
        
        int jm = max_j - 1; 
        if(jm < 0) 
            jm = 0;
        int jc = max_j;
        int jp = max_j + 1; 
        if(jp >= search_size * 2) 
            jp = search_size * 2;

        double a = (         *loc2D(v, size, size, im, jm)     \
                        +    *loc2D(v, size, size, ic, jm)     \
                        +    *loc2D(v, size, size, ip, jm)     \
                        -2 * *loc2D(v, size, size, im, jc)     \
                        -2 * *loc2D(v, size, size, ic, jc)     \
                        -2 * *loc2D(v, size, size, ip, jc)     \
                        +    *loc2D(v, size, size, im, jp)     \
                        +    *loc2D(v, size, size, ic, jp)     \
                        +    *loc2D(v, size, size, ip, jp)   ) / 6;

        double b = (         *loc2D(v, size, size, im, jm)     \
                        -    *loc2D(v, size, size, ip, jm)     \
                        -    *loc2D(v, size, size, im, jp)     \
                        +    *loc2D(v, size, size, ip, jp)   ) / 4;

        double c = (         *loc2D(v, size, size, im, jm)     \
                        -2 * *loc2D(v, size, size, ic, jm)     \
                        +    *loc2D(v, size, size, ip, jm)     \
                        +    *loc2D(v, size, size, im, jc)     \
                        -2 * *loc2D(v, size, size, ic, jc)     \
                        +    *loc2D(v, size, size, ip, jc)     \
                        +    *loc2D(v, size, size, im, jp)     \
                        -2 * *loc2D(v, size, size, ic, jp)     \
                        +    *loc2D(v, size, size, ip, jp)   ) / 6;

        double d = (   -     *loc2D(v, size, size, im, jm)     \
                       -     *loc2D(v, size, size, ic, jm)     \
                       -     *loc2D(v, size, size, ip, jm)     \
                       +     *loc2D(v, size, size, im, jp)     \
                       +     *loc2D(v, size, size, ic, jp)     \
                       +     *loc2D(v, size, size, ip, jp)   ) / 6;

        double e = (   -     *loc2D(v, size, size, im, jm)     \
                       +     *loc2D(v, size, size, ip, jm)     \
                       -     *loc2D(v, size, size, im, jc)     \
                       +     *loc2D(v, size, size, ip, jc)     \
                       -     *loc2D(v, size, size, im, jp)     \
                       +     *loc2D(v, size, size, ip, jp)   ) / 6;

        double f = (   -     *loc2D(v, size, size, im, jm)     \
                       + 2 * *loc2D(v, size, size, ic, jm)     \
                       -     *loc2D(v, size, size, ip, jm)     \
                       + 2 * *loc2D(v, size, size, im, jc)     \
                       + 5 * *loc2D(v, size, size, ic, jc)     \
                       + 2 * *loc2D(v, size, size, ip, jc)     \
                       -     *loc2D(v, size, size, im, jp)     \
                       + 2 * *loc2D(v, size, size, ic, jp)     \
                       -     *loc2D(v, size, size, ip, jp)   ) / 9;

        double denom = 4 * a * c - b * b;
        double px = (b * e - 2 * c * d) / denom;
        double py = (b * d - 2 * a * e) / denom;

        if(-1 < px && px < 1 && -1 < py && py < 1) {
            *y = (float)(max_i - search_size + py);
            *x = (float)(max_j - search_size + px);
            return a * px * px + b * px * py + c * py * py + d * px + e * py + f;
        } else {
            *y = max_i - search_size;
            *x = max_j - search_size;
            return max_corr;
        }
    } else {
        *y = max_i - search_size;
        *x = max_j - search_size;
        return max_corr;
    }
}



void correlation_image(double *corr, float *ref, float *tgt, int w, int h, int search_size, int bx, int by,  int patch_num_l, int patch_num_r, int patch_num_b, int patch_num_t, int patch_offset, int patch_size)
{
    const int size = search_size * 2 + 1;

    #pragma omp parallel for
    for(int k = 0; k < size * size; k++) {
        
        int u = k / size;
        int v = k % size;

        int s = u - search_size + by;
        int t = v - search_size + bx;
        
        for(int i = -patch_num_b; i <= patch_num_t; ++i) {
            for(int j = -patch_num_l; j <= patch_num_r; ++j) {
                int cy = h / 2 + i * patch_offset;
                int cx = w / 2 + j * patch_offset;
                double val = normalized_cross_correlation(w, h, cx, cy, patch_size, ref, tgt, t, s);
                corr[k] += val;
            }
        }
    }
}

static double estimate_motion_recursive(int level, bool top,
                                        int search_size, int patch_size, int patch_offset,
                                        float x_range, float y_range,
                                        int w, int h, float *ref, float *tgt, float *x, float *y)
{
    int by = 0;
    int bx = 0;

    if(level > 0) {
        const int patch_size_half = (patch_size + 1) / 2;
        const int patch_offset_half = (patch_offset + 1) / 2;
        const int w_half = w / 2;
        const int h_half = h / 2;
        float *ref_half = (float *) malloc(w_half * h_half * sizeof(float));

        // checkCudaErrors(cudaMallocManaged((void **)&ref_half, w_half * h_half * sizeof(float)));
        float *tgt_half = (float *) malloc(w_half * h_half * sizeof(float));

        // checkCudaErrors(cudaMallocManaged((void **)&tgt_half, w_half * h_half * sizeof(float)));
        downsample_image(w_half, h_half, w, h, tgt, tgt_half);
        downsample_image(w_half, h_half, w, h, ref, ref_half);
        
        float x_half, y_half;
        estimate_motion_recursive(level - 1, false,
                                  search_size, patch_size_half, patch_offset_half,
                                  x_range, y_range,
                                  w_half, h_half, ref_half, tgt_half, &x_half, &y_half);

        free(ref_half);
        free(tgt_half);
        by = (int)(2 * y_half);
        bx = (int)(2 * x_half);
    }
    int y_space = (int)(h * y_range * 0.5) - patch_size - search_size;
    int x_space = (int)(w * x_range * 0.5) - patch_size - search_size;
    int patch_num_b = (y_space + by) / patch_offset;
    int patch_num_t = (y_space - by) / patch_offset;
    int patch_num_l = (x_space + bx) / patch_offset;
    int patch_num_r = (x_space - bx) / patch_offset;

    if(patch_num_b < 0)
        patch_num_b = 0;
    
    if(patch_num_t < 0)
        patch_num_t = 0;
    
    if(patch_num_l < 0)
        patch_num_l = 0;
    
    if(patch_num_r < 0)
        patch_num_r = 0;
    
    const int size = search_size * 2 + 1;
    double *corr = (double *) malloc(size * size * sizeof(double));
    // checkCudaErrors(cudaMallocManaged((void **)corr, size * size * sizeof(float)));
    // cudaMemset(corr, 0, size * size * sizeof(float));
    memset(corr, 0, size * size * sizeof(double));

    correlation_image(corr, ref, tgt, w, h, search_size, bx, by,  patch_num_l, patch_num_r, patch_num_b, patch_num_t, patch_offset, patch_size);
    float py;
    float px;
    double ret = get_peak(search_size, corr, top, &px, &py);
    *y = by + py;
    *x = bx + px;

    free(corr);
    return ret;
}

static double estimate_motion(motion_param_t &param,
                              int w, int h, float *ref, float *tgt, float *x, float *y)
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

        
static float sample_bilinear(float *dat, int w, int h, float x, float y)
{
    int ym1;
    int yp1;
    int xm1;
    int xp1;
    float frac_y;
    float frac_x;

    ym1 = (int)y;
    xm1 = (int)x;
    yp1 = ym1 + 1;
    xp1 = xm1 + 1;
    frac_y = y - ym1;
    frac_x = x - xm1;
    
    if(ym1 < 0) 
        ym1 = 0;
    else if(ym1 >= h) 
        ym1 = h - 1;
    
    if(xm1 < 0) 
        xm1 = 0;
    else if(xm1 >= w) 
        xm1 = w - 1;
    
    if(yp1 < 0) 
        yp1 = 0;
    else if(yp1 >= h) 
        yp1 = h - 1;
    
    if(xp1 < 0) 
        xp1 = 0;
    else if(xp1 >= w) 
        xp1 = w - 1;

    return    (*loc2D(dat, h, w, ym1, xm1) * (1 - frac_x) \
               + *loc2D(dat, h, w, ym1, xp1) * frac_x)  * (1 - frac_y)
            + (*loc2D(dat, h, w, yp1, xm1) * (1 - frac_x) \
               + *loc2D(dat, h, w, yp1, xp1) * frac_x) * frac_y;

}

static void apply_motion(int w, int h, float *in, float x, float y, float *out)
{
    #pragma omp parallel for
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            *loc2D(out, h, w, i, j) = sample_bilinear(in, w, h, j+x, i+y);
        }
    }
}

std::vector<motion_t> correct_motion(motion_param_t &param,
                                     int num_pages, int width, int height, float *img,
                                     motion_range_t &range)
{
    const float a_var = param.a_stdev * param.a_stdev;
    const float m_var = param.m_stdev * param.m_stdev;
    const float time_step = 1.0;
    KalmanFilter *kf_x = new KalmanFilter(a_var, m_var, time_step);
    KalmanFilter *kf_y = new KalmanFilter(a_var, m_var, time_step);
    kf_x->initialize(0, 0, 0, 0);
    kf_y->initialize(0, 0, 0, 0);
    const float thresh_xy = param.thresh_xy;
    const float thresh_c = param.thresh_c;
    const int length = param.length;
    int c_count = 0;
    int c_index = 0;
    float *c_ring = (float *) malloc(length * sizeof(float));
    float med = 0;
    std::vector<motion_t> motion_list;

    float *in; //= (float *) malloc(width * height * sizeof(float));
    float *ref;// = (float *) malloc(width * height * sizeof(float));
    float *out = (float *) malloc(width * height * sizeof(float));
    float min_x = 0, max_x = 0, min_y = 0, max_y = 0;

    // First Frame
    {
        motion_t m;
        m.x = 0;
        m.y = 0;
        m.corr = 0;
        m.valid = true;
        motion_list.push_back(m);
    }
    ref = loc3D(img, num_pages, height, width, 0, 0, 0);

    for(int i = 1; i < num_pages; i++) {
        in = loc3D(img, num_pages, height, width, i, 0, 0);
        float x, y;
        double c = estimate_motion(param, width, height, ref, in, &x, &y);
        if(i < 100) {
            printf("%d C: %f %f %f\n", i, c, x, y);
        }

        float ux = kf_x->update(x);
        float uy = kf_y->update(y);
        float dif_x = fabsf(ux - x);
        float dif_y = fabsf(uy - y);
        bool valid_x = dif_x < thresh_xy;
        bool valid_y = dif_y < thresh_xy;
        if(valid_x && valid_y)
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

        apply_motion(width, height, in, x, y, out);
        memcpy(loc3D(img, num_pages, height, width, i, 0, 0), out, width * height * sizeof(float));

        // save motion vector
        motion_t m;
        m.x = x;
        m.y = y;
        m.corr = c;
        if(valid_x && valid_y && valid_c)
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
    free(out);

    range.min_x = min_x;
    range.max_x = max_x;
    range.min_y = min_y;
    range.max_y = max_y;

    return motion_list;        
}

