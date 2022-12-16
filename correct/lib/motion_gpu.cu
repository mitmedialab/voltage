#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "motion.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdexcept>

#include <thread>

#define SHMEM_SIZE  1024
#define TGT_DIM  10
#define checkCudaErrors(func) {                                        \
    cudaError_t error = func;                                          \
    if (error != 0) {                                                   \
        throw std::runtime_error(std::string("Cuda failure Error") );     \
        printf("%s-%s(%d): Cuda failure Error: %s\n", __FILE__, __func__, __LINE__, cudaGetErrorString(error));  \
        fflush(stdout);                                                 \
    }                                                                  \
}


#define BYTESPERPIXEL 1

inline float* loc3D(float *img, int t, int h, int w, int k, int i, int j)
{
    return img + ((k * w * h) + ((i * w) + j)) * BYTESPERPIXEL; 
}


typedef struct
{
    int gpu_device_id;
    int t_start;
    int t_end;
    int delta_t;
    float *in;
    float *out;
    float *ref;
    float *ref_l1;
    float *ref_l0;
    float *tgt_l1;
    float *tgt_l0;
    double *corr_buf;
    double *ref_sum;
    double *ref_var;
    double *tgt_sum;
    double *tgt_sum2;
    double *corr;
    int *narray;
} gpu_buffer_t;

typedef struct
{
    int T;
    int W;
    int H;
    int size;
    int corr_size;
    int w_l1;
    int h_l1;
    int w_l0;
    int h_l0;
    int loop23_ct_estimate;
    int loop23_ct_rnd;
    float *himg;
    float *hout;
    gpu_buffer_t *gbuf;
    motion_param_t *mp;
    std::vector<motion_t> motion_list;
} gpuMotionCorrect_t;



__forceinline__ __device__
float* gloc3D(float *img, int t, int h, int w, int k, int i, int j)
{
    return img + ((k * w * h) + ((i * w) + j)) * BYTESPERPIXEL; 
}

__forceinline__ __device__
float* gloc2D(float *img, int h, int w, int i, int j)
{
    return img + ((i * w) + j) * BYTESPERPIXEL; 
}

__forceinline__ __device__
double* gloc2D(double *img, int h, int w, int i, int j)
{
    return img + ((i * w) + j) * BYTESPERPIXEL; 
}

__global__
static void downsample_image(int wh, int hh, int w, int h, float *img, float *out)
{
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    int j;

    float *im;
    float v1, v2, v3, v4;
    
    for(int ctr = index; ctr < wh * hh; ctr+=stride) {

        i = ctr / wh;
        j = ctr % wh;
        im = gloc2D(img, h, w, i*2, j*2);
        v1 = im[0];
        v2 = im[w];
        v3 = im[1];
        v4 = im[w+1];
        
        out[ctr] = (v1 + v2 + v3 + v4) * 0.25f;
    }
}

__global__
void get_gpu_peak(int search_size, double *v, bool subpixel)
{
    int size = search_size * 2 + 1;
    double max_corr = 0;
    int max_i = 0;
    int max_j = 0;
    int i, j = 0;
    double *im;
    for(i=0; i < size; ++i) {
        im = v + i*size;
        for(j=0;j<size;++j) {
            if(max_corr < im[j]) {
                max_corr = im[j];
                max_i = i;
                max_j = j;
            }
        }
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

        double vmm = *gloc2D(v, size, size, im, jm);
        double vcm = *gloc2D(v, size, size, ic, jm);
        double vpm = *gloc2D(v, size, size, ip, jm);
        double vmc = *gloc2D(v, size, size, im, jc);
        double vcc = *gloc2D(v, size, size, ic, jc);
        double vpc = *gloc2D(v, size, size, ip, jc);
        double vmp = *gloc2D(v, size, size, im, jp);
        double vcp = *gloc2D(v, size, size, ic, jp);
        double vpp = *gloc2D(v, size, size, ip, jp);


        double a = (         vmm     \
                        +    vcm     \
                        +    vpm     \
                        -2 * vmc     \
                        -2 * vcc     \
                        -2 * vpc     \
                        +    vmp     \
                        +    vcp     \
                        +    vpp   ) / 6;

        double b = (         vmm     \
                        -    vpm     \
                        -    vmp     \
                        +    vpp   ) / 4;

        double c = (         vmm     \
                        -2 * vcm     \
                        +    vpm     \
                        +    vmc     \
                        -2 * vcc     \
                        +    vpc     \
                        +    vmp     \
                        -2 * vcp     \
                        +    vpp   ) / 6;

        double d = (   -     vmm     \
                       -     vcm     \
                       -     vpm     \
                       +     vmp     \
                       +     vcp     \
                       +     vpp   ) / 6;

        double e = (   -     vmm     \
                       +     vpm     \
                       -     vmc     \
                       +     vpc     \
                       -     vmp     \
                       +     vpp   ) / 6;

        double f = (   -     vmm     \
                       + 2 * vcm     \
                       -     vpm     \
                       + 2 * vmc     \
                       + 5 * vcc     \
                       + 2 * vpc     \
                       -     vmp     \
                       + 2 * vcp     \
                       -     vpp   ) / 9;

        double denom = 4 * a * c - b * b;
        double px = (b * e - 2 * c * d) / denom;
        double py = (b * d - 2 * a * e) / denom;

        if(-1 < px && px < 1 && -1 < py && py < 1) {
            v[0] = a * px * px + b * px * py + c * py * py + d * px + e * py + f;
            v[1] = (float)(max_i - search_size + py);
            v[2] = (float)(max_j - search_size + px);

            return;
        } else {
            v[0] = max_corr;
            v[1] = (float)(max_i - search_size);
            v[2] = (float)(max_j - search_size);
            return;
        }
    } else {
        v[0] = max_corr;
        v[1] = (float)(max_i - search_size);
        v[2] = (float)(max_j - search_size);
        return;
    }
}        

__device__
double norm_corr(float *ref, float *tgt, int w, int h, int search_size, int cx, int cy, int x, int y, int patch_size, double ref_sum, double ref_var, int n, double tgt_sum, double tgt_sum2, int idx, int iy, int ix)
{

    __shared__ double cs[TGT_DIM][TGT_DIM];
    __shared__ double covar[TGT_DIM][TGT_DIM];
    int start_i;
    int end_i;
    int start_j;
    int end_j;
    double tgt_var;
    int i, j, u, v;
    float *r1, *t1;
    double ret;

    start_i = cy - patch_size; if(start_i < 0) start_i = 0;
    end_i = cy + patch_size; if(end_i  >= h) end_i = h-1;
    start_j = cx - patch_size; if(start_j < 0) start_j = 0;
    end_j   = cx + patch_size; if(end_j  >= w) end_j = w-1;

    cs[iy][ix] = 0;
    for(i = start_i; i <= end_i; i++) {
        u = i + y;

        if(u < 0) u = 0;
        if(u >= h) u = h-1;   
        
        r1 = gloc2D(ref, h, w, i, 0);
        t1 = gloc2D(tgt, h, w, u, 0);
        
        for(j = start_j; j <= end_j; j++) {

            v = j + x;
            if(v < 0) v = 0;
            if(v >= w) v = w-1;
            cs[iy][ix] = __fadd_rd(cs[iy][ix], __fmul_rd(r1[j], t1[v]));

        }
    }

    tgt_sum = __fdiv_rd(tgt_sum, n);
    tgt_sum2 = __fdiv_rd(tgt_sum2, n);
    cs[iy][ix] = __fdiv_rd(cs[iy][ix], n);
    covar[iy][ix] = __fsub_rd(cs[iy][ix], __fmul_rd(ref_sum, tgt_sum));
    tgt_var = __fsub_rd(tgt_sum2, tgt_sum * tgt_sum);
    if(tgt_var < 0) tgt_var = 0;
    ret = __fdiv_rd(ref_sum * covar[iy][ix], __powf(__fadd_rd(__fmul_rd(ref_var, tgt_var), 1e-6), 0.5));
    return ret;
}

__global__
void get_tgt_horiz(float *tgt, int w, int h, int search_size, int bx, int by, int patch_num_l, int patch_num_b, int patch_offset, int patch_size, int max_lim_l0, double *tgt_sum, double *tgt_sum2)
{
    double t;
    double ts;

    int y = 0 - search_size + by;
    int x = threadIdx.x - search_size + bx;

    int i = blockIdx.y - patch_num_b;
    int j = blockIdx.x - patch_num_l;
    int idx = blockIdx.y * max_lim_l0 + blockIdx.x;
    int di = (idx * blockDim.x * blockDim.x) + threadIdx.x;
    int cy = (h>>1) + i * patch_offset;
    int cx = (w>>1) + j * patch_offset;
    int start_i;
    int end_i;
    int start_j;
    int end_j;
    int u, v;
    float *t1;

    start_i = cy - patch_size; if(start_i < 0) start_i = 0;
    end_i = cy + patch_size; if(end_i  >= h) end_i = h-1;
    start_j = cx - patch_size; if(start_j < 0) start_j = 0;
    end_j   = cx + patch_size; if(end_j  >= w) end_j = w-1;

    t = 0;
    ts = 0;

    for(i = start_i; i<=end_i; ++i) {
        u = i + y;
        if(u < 0) u = 0;
        if(u >= h) u = h-1;
        t1 = gloc2D(tgt, h, w, u, 0);
        for(j = start_j; j<=end_j; ++j) {
            v = j + x;
            if(v < 0) v = 0;
            if(v >= w) v = w-1;
            
            t = __fadd_rd(t, t1[v]);
            ts = __fadd_rd(ts, t1[v] * t1[v]);

        }
    }

    tgt_sum[di] = t;
    tgt_sum2[di] = ts;
}


__global__
void get_tgt_verti(float *tgt, int w, int h, int search_size, int bx, int by, int patch_num_l, int patch_num_b, int patch_offset, int patch_size, int max_lim_l0, double *tgt_sum, double *tgt_sum2)
{
    __shared__ double top[TGT_DIM][TGT_DIM];
    __shared__ double bottom[TGT_DIM][TGT_DIM];

    __shared__ double sq_top[TGT_DIM][TGT_DIM];
    __shared__ double sq_bottom[TGT_DIM][TGT_DIM];

    int y = threadIdx.y - search_size + by;
    int x = threadIdx.x - search_size + bx;

    int i = blockIdx.y - patch_num_b;
    int j = blockIdx.x - patch_num_l;

    int idx = blockIdx.y * max_lim_l0 + blockIdx.x;
    int di = (idx * blockDim.x * blockDim.y + threadIdx.x);
    
    int cy = (h>>1) + i * patch_offset;
    int cx = (w>>1) + j * patch_offset;
    int start_i;
    int end_i;
    int start_j;
    int end_j;
    int v;
    int ti, bi;
    float *t1, *t2;

    start_i = cy - patch_size; if(start_i < 0) start_i = 0;
    end_i = cy + patch_size; if(end_i  >= h) end_i = h-1;
    start_j = cx - patch_size; if(start_j < 0) start_j = 0;
    end_j   = cx + patch_size; if(end_j  >= w) end_j = w-1;

    ti = start_i + y;
    if(ti < 0) ti = 0;
    if(ti >= h) ti = h-1;
    bi = end_i + y;
    if(bi < 0) bi = 0;
    if(bi >= h) bi = h-1;

    t1 = gloc2D(tgt, h, w, ti, 0);
    t2 = gloc2D(tgt, h, w, bi, 0);
    top[threadIdx.y][threadIdx.x] = 0;
    bottom[threadIdx.y][threadIdx.x] = 0;
    sq_top[threadIdx.y][threadIdx.x] = 0;
    sq_bottom[threadIdx.y][threadIdx.x] = 0;
    for(j = start_j; j <= end_j; ++j) {
        v = j + x;
        top[threadIdx.y][threadIdx.x] = __fadd_rd(top[threadIdx.y][threadIdx.x], t1[v]);
        bottom[threadIdx.y][threadIdx.x] = __fadd_rd(bottom[threadIdx.y][threadIdx.x], t2[v]);

        sq_top[threadIdx.y][threadIdx.x] = __fadd_rd(sq_top[threadIdx.y][threadIdx.x], t1[v] * t1[v]);
        sq_bottom[threadIdx.y][threadIdx.x] = __fadd_rd(sq_bottom[threadIdx.y][threadIdx.x], t2[v] * t2[v]);
    }
    __syncthreads();

    if(threadIdx.y == 0) {
        idx = 0;
        for(i = 1; i < blockDim.y; ++i) {
            idx += blockDim.x;
            tgt_sum[di + idx] = __fadd_rd(__fsub_rd(tgt_sum[di + idx - blockDim.x], top[i - 1][threadIdx.x]), bottom[i][threadIdx.x]);
            tgt_sum2[di + idx] = __fadd_rd(__fsub_rd(tgt_sum2[di + idx - blockDim.x], sq_top[i - 1][threadIdx.x]), sq_bottom[i][threadIdx.x]);
        }
    }
}


__global__
void get_refs(float *ref, int w, int h, int search_size, int patch_num_l, int patch_num_b, int patch_offset, int patch_size, int max_lim_l0, double *ref_sum, double *ref_var, int *n)
{
    __shared__ double r[100];
    __shared__ double rs[100];
    __shared__ int nv[100];


    int start_i;
    int end_i;
    int start_j;
    int end_j;
    int i, j;
    float *r1;
    int cy;
    int cx;
    // double rs = 0;
    double rv = 0;
    // int nv = 0;
    int idx;

    i = blockIdx.x - patch_num_b;
    j = threadIdx.x - patch_num_l;
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    cy = (h>>1) + i * patch_offset;
    cx = (w>>1) + j * patch_offset;

    r[threadIdx.x] = 0;
    rs[threadIdx.x] = 0;
    nv[threadIdx.x] = 0;

    start_i = cy - patch_size; if(start_i < 0) start_i = 0;
    end_i = cy + patch_size; if(end_i  >= h) end_i = h-1;
    start_j = cx - patch_size; if(start_j < 0) start_j = 0;
    end_j   = cx + patch_size; if(end_j  >= w) end_j = w-1;

    for(i = start_i; i <= end_i; i++) {
        r1 = gloc2D(ref, h, w, i, 0);
        for(j = start_j; j <= end_j; j++) {
            r[threadIdx.x] = __fadd_rd(r[threadIdx.x], r1[j]);
            rs[threadIdx.x] = __fadd_rd(rs[threadIdx.x], r1[j] * r1[j]);
            nv[threadIdx.x] = nv[threadIdx.x] + 1;
        }
    }
    r[threadIdx.x] = __fdiv_rd(r[threadIdx.x], nv[threadIdx.x]);
    rs[threadIdx.x] = __fdiv_rd(rs[threadIdx.x], nv[threadIdx.x]);
    rv = __fsub_rd(rs[threadIdx.x], r[threadIdx.x] * r[threadIdx.x]);
    if(rv < 0) rv = 0;

    ref_sum[idx] = r[threadIdx.x];
    ref_var[idx] = rv;    
    n[idx] = nv[threadIdx.x];
}





__global__
void correlation_image(float *ref, float *tgt, int w, int h, int search_size, int bx, int by, int patch_num_l, int patch_num_b, int patch_offset, int patch_size, double *corr_buf, int stride, int max_lim_l0, double *ref_sum, double *ref_var, int *n, double *tgt_sum, double *tgt_sum2)
{
    int size = search_size * 2 + 1;
    int s;
    int t;
    int cy;
    int cx;
    int idx;
    s = threadIdx.y - search_size + by;
    t = threadIdx.x - search_size + bx;

    __shared__ double ts[TGT_DIM][TGT_DIM];
    __shared__ double ts2[TGT_DIM][TGT_DIM];
    __shared__ double r;
    __shared__ double rv;
    __shared__ int nv;
    
    int i = blockIdx.y - patch_num_b;
    int j = blockIdx.x - patch_num_l;
    idx = blockIdx.y * max_lim_l0 + blockIdx.x;     

    int di = (idx * blockDim.x * blockDim.y + threadIdx.y*blockDim.x + threadIdx.x);

    r = ref_sum[idx];
    rv = ref_var[idx];
    nv = n[idx];
    ts[threadIdx.y][threadIdx.x] = tgt_sum[di];
    ts2[threadIdx.y][threadIdx.x] = tgt_sum2[di];

    __syncthreads();

    cy = (h>>1) + i * patch_offset;
    cx = (w>>1) + j * patch_offset;

    corr_buf[(threadIdx.y * size + threadIdx.x) * stride + idx] = norm_corr(ref, tgt, w, h, search_size, cx, cy, t, s, patch_size, \
        r, rv, nv, ts[threadIdx.y][threadIdx.x], ts2[threadIdx.y][threadIdx.x], idx, threadIdx.y, threadIdx.x);
}



__global__
void sum_reduction(double *in, double *out, int h, int w, int max_lim)
{
    __shared__ double partial_sum[SHMEM_SIZE];
    int shift = w*blockIdx.x;
    int tid = threadIdx.x;
    int mul = !(int)(threadIdx.x / max_lim);
    partial_sum[threadIdx.x] = mul * in[shift + tid];
    __syncthreads();

    for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = partial_sum[0];
    }

}

static double estimate_motion_recursive(int level, bool top,
                                        int search_size, int patch_size, int patch_offset,
                                        float x_range, float y_range,
                                        int w, int h, float *ref, float *tgt, float *x, float *y, double *corr, float *ref_l1, float *ref_l0, float *tgt_l1, float *tgt_l0, double *corr_buf, double *ref_sum, double *ref_var, int *narray, double *tgt_sum, double *tgt_sum2)
{
    int by = 0;
    int bx = 0;

    if(level > 0) {
        const int patch_size_half = (patch_size + 1) / 2;
        const int patch_offset_half = (patch_offset + 1) / 2;
        const int w_half = w / 2;
        const int h_half = h / 2;
        float *ref_half;
        float *tgt_half;

        if(level == 2) {
            ref_half = ref_l1;
            tgt_half = tgt_l1;
        } else {
            ref_half = ref_l0;
            tgt_half = tgt_l0;
        }


        int blockSize = 256;
        int numBlocks = (w_half * h_half + blockSize - 1) / blockSize;
        
        downsample_image<<<numBlocks, blockSize>>>(w_half, h_half, w, h, tgt, tgt_half);
        
        float x_half, y_half;
        estimate_motion_recursive(level - 1, false,
                                  search_size, patch_size_half, patch_offset_half,
                                  x_range, y_range,
                                  w_half, h_half, ref_half, tgt_half, &x_half, &y_half, corr, ref_l1, ref_l0, tgt_l1, tgt_l0, corr_buf, ref_sum, ref_var, narray, tgt_sum, tgt_sum2);

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
    
    int numBlocks = size * size;
    int max_lim_l1 = (patch_num_t + patch_num_b + 1);
    int max_lim_l0 = (patch_num_r + patch_num_l + 1);
    int max_lim = max_lim_l0 * max_lim_l1;
    
    dim3 A(max_lim_l0, max_lim_l1);
    dim3 B(size, size);


    int stride = pow(2, ceil(log(max_lim)/log(2)));

    // Get reference values for different grids
    get_refs<<<max_lim_l1, max_lim_l0>>>(ref, w, h, search_size, patch_num_l, patch_num_b, patch_offset, patch_size, max_lim_l0, ref_sum, ref_var, narray);

    // For each grid, we get the first row of tgt values
    get_tgt_horiz<<<A, size>>>(tgt, w, h, search_size, bx, by, patch_num_l, patch_num_b, patch_offset, patch_size, max_lim_l0, tgt_sum, tgt_sum2);

    // For each grid, we compute all tgt values using overlap information vertically down.
    get_tgt_verti<<<A, B>>>(tgt, w, h, search_size, bx, by, patch_num_l, patch_num_b, patch_offset, patch_size, max_lim_l0, tgt_sum, tgt_sum2);

    // Compute the covariance and NCC
    correlation_image<<<A, B>>>(ref, tgt, w, h, search_size, bx, by, patch_num_l, patch_num_b, patch_offset, patch_size, corr_buf, stride, max_lim_l0, ref_sum, ref_var, narray, tgt_sum, tgt_sum2);

    // Sum the grids to get a correlation value of search area (7x7 for FOV2)
    sum_reduction<<<numBlocks, stride>>>(corr_buf, corr, numBlocks, stride, max_lim);

    // Get the peak points
    get_gpu_peak<<<1,1>>>(search_size, corr, top);

    cudaDeviceSynchronize();

    *y = by + corr[1];
    *x = bx + corr[2];

    return corr[0];
}

static double estimate_motion(gpuMotionCorrect_t *gmc, int gpu_id, float *in, float *x, float *y)
{
    gpu_buffer_t *gp = &(gmc->gbuf[gpu_id]);
    return estimate_motion_recursive(gmc->mp->level, true,
                                     gmc->mp->search_size, gmc->mp->patch_size, gmc->mp->patch_offset,
                                     gmc->mp->x_range, gmc->mp->y_range,
                                     gmc->W, gmc->H, gp->ref, in, x, y, gp->corr, gp->ref_l1, gp->ref_l0, gp->tgt_l1, gp->tgt_l0, gp->corr_buf, gp->ref_sum, gp->ref_var, gp->narray, gp->tgt_sum, gp->tgt_sum2);
}

__device__
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
    frac_y = __fsub_rd(y, ym1);
    frac_x = __fsub_rd(x, xm1);
    
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

    return    __fadd_rd(__fmul_rd(__fadd_rd(__fmul_rd(*gloc2D(dat, h, w, ym1, xm1), __fsub_rd(1, frac_x)), __fmul_rd(*gloc2D(dat, h, w, ym1, xp1), frac_x)), __fsub_rd(1, frac_y)), 
        __fmul_rd(__fadd_rd(__fmul_rd(*gloc2D(dat, h, w, yp1, xm1), __fsub_rd(1, frac_x)), __fmul_rd(*gloc2D(dat, h, w, yp1, xp1), frac_x)), frac_y));

}

__global__
void apply_motion(int w, int h, float *in, float x, float y, float *out)
{
    *gloc2D(out, h, w, blockIdx.x, threadIdx.x) = sample_bilinear(in, w, h, threadIdx.x+x, blockIdx.x+y);
}



void allocate_device_buffers(gpuMotionCorrect_t *gmc, int gpu_id)
{   
    cudaSetDevice(gpu_id);

    gpu_buffer_t *gp = &(gmc->gbuf[gpu_id]);

    checkCudaErrors(cudaMalloc((void **)&gp->in, (gp->delta_t + 1) * gmc->W * gmc->H * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&gp->ref, gmc->W * gmc->H * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&gp->ref_l1, gmc->h_l1 * gmc->w_l1 * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&gp->ref_l0, gmc->h_l0 * gmc->w_l0 * sizeof(float)));

    //checkCudaErrors(cudaMalloc((void **)&gp->out, gp->delta_t * gmc->W * gmc->H * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&gp->tgt_l1, gmc->h_l1 * gmc->w_l1 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&gp->tgt_l0, gmc->h_l0 * gmc->w_l0 * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&gp->corr, gmc->corr_size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&gp->corr_buf, gmc->size * gmc->size * gmc->loop23_ct_rnd * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&gp->ref_sum, gmc->loop23_ct_rnd * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&gp->ref_var, gmc->loop23_ct_rnd * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&gp->tgt_sum, gmc->size * gmc->size * gmc->loop23_ct_rnd * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&gp->tgt_sum2, gmc->size * gmc->size * gmc->loop23_ct_rnd * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&gp->narray, gmc->loop23_ct_rnd * sizeof(int)));
}

void process_gpu_threads(gpuMotionCorrect_t *gmc, int gpu_id)
{
    cudaSetDevice(gpu_id);

    gpu_buffer_t *gp = &(gmc->gbuf[gpu_id]);

    const int blockSize = 256;
    int numBlocks;

    //checkCudaErrors(cudaMemcpy(gp->in, loc3D(gmc->himg, gmc->T, gmc->H, gmc->W, gp->t_start, 0, 0), gp->delta_t * gmc->W * gmc->H * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gp->in + gmc->W * gmc->H, loc3D(gmc->himg, gmc->T, gmc->H, gmc->W, gp->t_start, 0, 0), gp->delta_t * gmc->W * gmc->H * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(gp->ref, gmc->himg, gmc->W * gmc->H * sizeof(float), cudaMemcpyHostToDevice));

    numBlocks = (gmc->w_l1 * gmc->h_l1 + blockSize - 1) / blockSize;
    downsample_image<<<numBlocks, blockSize>>>(gmc->w_l1, gmc->h_l1, gmc->W, gmc->H, gp->ref, gp->ref_l1);

    numBlocks = (gmc->w_l0 * gmc->h_l0 + blockSize - 1) / blockSize;
    downsample_image<<<numBlocks, blockSize>>>(gmc->w_l0, gmc->h_l0, gmc->w_l1, gmc->h_l1, gp->ref_l1, gp->ref_l0);

    cudaMemset(gp->corr, 0, gmc->corr_size * sizeof(double));

    for(int i = 0; i < gp->delta_t; i++) {
        float x = 0, y = 0;
        double c = estimate_motion(gmc, gpu_id, loc3D(gp->in, gmc->T, gmc->H, gmc->W, i+1, 0, 0), &x, &y);
#ifdef DEBUG
        if(i < 100)
            printf("[%d] C: %f, x %f , y %f\n", i, c, x, y);
#endif
        //apply_motion<<<gmc->H, gmc->W>>>(gmc->W, gmc->H, loc3D(gp->in, gmc->T, gmc->H, gmc->W, i, 0, 0), x, y, loc3D(gp->out, gmc->T, gmc->H, gmc->W, i, 0, 0));
        apply_motion<<<gmc->H, gmc->W>>>(gmc->W, gmc->H, loc3D(gp->in, gmc->T, gmc->H, gmc->W, i+1, 0, 0), x, y, loc3D(gp->in, gmc->T, gmc->H, gmc->W, i, 0, 0));
        motion_t m;
        m.x = x;
        m.y = y;
        m.corr = c;
        m.valid = true;
        gmc->motion_list.at(gp->t_start + i) = m;
    }

    //cudaMemcpy(loc3D(gmc->hout, gmc->T, gmc->H, gmc->W, gp->t_start, 0, 0), gp->out, gp->delta_t * gmc->H * gmc->W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(loc3D(gmc->hout, gmc->T, gmc->H, gmc->W, gp->t_start, 0, 0), gp->in, gp->delta_t * gmc->H * gmc->W * sizeof(float), cudaMemcpyDeviceToHost);
}


void free_device_buffers(gpuMotionCorrect_t *gmc, int gpu_id)
{
    cudaSetDevice(gpu_id);

    gpu_buffer_t *gp = &(gmc->gbuf[gpu_id]);
    cudaFree(gp->in);
    cudaFree(gp->ref);
    cudaFree(gp->ref_l1);
    cudaFree(gp->ref_l0);
    //cudaFree(gp->out);
    cudaFree(gp->tgt_l1);
    cudaFree(gp->tgt_l0);
    cudaFree(gp->corr);
    cudaFree(gp->corr_buf);
    cudaFree(gp->ref_sum);
    cudaFree(gp->ref_var);
    cudaFree(gp->tgt_sum);
    cudaFree(gp->tgt_sum2);
    cudaFree(gp->narray);
}


int* get_time_frame_splits(int T, int n)
{
    int Q = T/n;
    int R = T%n;
    int r = n-R;
    int *bins = (int *) malloc((r + R) * sizeof(int));

    for(int i = 0; i<n; ++i) {
        if(i < R)
            bins[i] = Q+1;
        else
            bins[i] = Q;
    }

    return bins;
}


std::vector<motion_t> correct_motion_gpu(motion_param_t &param,
                                         int num_pages, int width, int height,
                                         float *in_image, float *out_image)
{
    int gpu_n;
    checkCudaErrors(cudaGetDeviceCount(&gpu_n));
    printf("Motion Correction GPU NUM %d\n", gpu_n);

    //const size_t gpu_mem_size = 2L * 1024 * 1024 * 1024;
    //const size_t data_size_per_frame = width * height * sizeof(float);
    //const size_t num_streams = (gpu_mem_size + data_size_per_frame - 1) / data_size_per_frame;


    gpuMotionCorrect_t gmc;
    gmc.size = (param.search_size << 1) + 1;
    gmc.corr_size = pow(2, ceil(log(gmc.size * gmc.size) / log(2)));
    gmc.w_l1 = width >> 1;
    gmc.h_l1 = height >> 1;
    gmc.w_l0 = gmc.w_l1 >> 1;
    gmc.h_l0 = gmc.h_l1 >> 1;
    gmc.loop23_ct_estimate = int((height / param.patch_offset) * (width / param.patch_offset)); 
    gmc.loop23_ct_rnd = pow(2, ceil(log(gmc.loop23_ct_estimate) / log(2)));

    gmc.gbuf = new gpu_buffer_t[gpu_n];
    gmc.T = num_pages;
    gmc.W = width;
    gmc.H = height;
    gmc.himg = in_image;
    gmc.hout = out_image;
    gmc.mp = &param;

    std::vector<std::thread> threads(gpu_n);
    int *splits_timeframes = get_time_frame_splits(num_pages, gpu_n);
    gmc.gbuf[0].t_start = 0;
    memcpy(out_image, in_image, height * width * sizeof(float)); // copy first frame

    gmc.motion_list.resize(num_pages);
    gmc.motion_list[0].x = 0;
    gmc.motion_list[0].y = 0;
    gmc.motion_list[0].corr = 0;
    gmc.motion_list[0].valid = true;

    for(int i = 0; i < gpu_n; i++)
    {
        gmc.gbuf[i].gpu_device_id = i;

        if(i != 0) {
            gmc.gbuf[i].t_start = gmc.gbuf[i-1].t_end;
        }
        gmc.gbuf[i].t_end = gmc.gbuf[i].t_start + splits_timeframes[i];
        gmc.gbuf[0].t_start = 1;
        gmc.gbuf[i].delta_t = gmc.gbuf[i].t_end - gmc.gbuf[i].t_start;
        allocate_device_buffers(&gmc, i);
    }
        
    for(int i = 0; i < gpu_n; i++)
    {
        threads[i] = std::thread(process_gpu_threads, &gmc, i);
    }
    
    for (auto& th : threads) {
        th.join();
    }

    for(int i = 0; i < gpu_n; i++)
    {
        free_device_buffers(&gmc, i);
    }

    delete [] gmc.gbuf;

    return gmc.motion_list;
}

