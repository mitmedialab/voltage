#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "wrapper.hpp"
#include <thread>
#include <vector>
#include <nppcore.h>
#include <nppi.h>


#define checkCudaErrors(func) {                                        \
    cudaError_t error = func;                                          \
    if (error != 0) {                                                  \
        printf("%s-%s(%d): Cuda failure Error: %s\n", __FILE__, __func__, __LINE__, cudaGetErrorString(error));  \
        fflush(stdout);                                                 \
    }                                                                  \
}

__forceinline__ __device__
float* gloc3D(float *img, int t, int h, int w, int k, int i, int j)
{
    return img + ((k * w * h) + ((i * w) + j)); 
}

__forceinline__ __device__
float* gloc3D_3C(float *img, int t, int h, int w, int k, int i, int j, int c)
{
    return img + (((k * w * h) + ((i * w) + j)) * 3) + c; 
}

__forceinline__ __device__
float* gloc2D(float *img, int h, int w, int i, int j)
{
    return img + ((i * w) + j); 
}

__forceinline__ __device__
double* gloc2D(double *img, int h, int w, int i, int j)
{
    return img + ((i * w) + j); 
}

__global__
void divide_by_C(float *arr, double *val, int t)
{
	float *x = gloc2D(arr, gridDim.x, t, blockIdx.x, 0);
	double sum = val[blockIdx.x];
    for(int i = threadIdx.x; i < (t + blockDim.x); i += blockDim.x) {
	 	if(i < t) {
	 		x[i] = x[i] / sum;
	 	}
    }
}

__global__
void conv_double_float_arr(double *in, float *out, int t)
{
	for(int i = threadIdx.x; i < (t + blockDim.x); i += blockDim.x) {
		if(i < t) {
			*gloc2D(out, gridDim.x, t, blockIdx.x, i) = *gloc2D(in, gridDim.x, t, blockIdx.x, i);
		}
	}
}

int* get_n_splits(int T, int n)
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


typedef struct gpuPostprocess {
	float *himg;
	float *hout;
	float *hmasks;
	int gpu_device_id;
	int t;
	int w;
	int h;
	int total_n;
	int n;
	int n_start;
	int n_end;
} gpuPostprocess;


void process_gpu_threads(gpuPostprocess *gp)
{
	cudaSetDevice(gp->gpu_device_id);
	float *gimg;
	checkCudaErrors(cudaMalloc((void **)&gimg, gp->t * gp->w * gp->h * sizeof(float)));
	checkCudaErrors(cudaMemcpy(gimg, gp->himg, gp->t * gp->h * gp->w * sizeof(float), cudaMemcpyHostToDevice));

	//Outputs
	float *out_frame;
	checkCudaErrors(cudaMalloc((void **) &out_frame, gp->w * gp->h * sizeof(float)));
	double *gout_double;
	checkCudaErrors(cudaMalloc((void **) &gout_double, gp->n * gp->t * sizeof(double)));
	float *gout;
	checkCudaErrors(cudaMalloc((void **) &gout, gp->n * gp->t * sizeof(float)));

	float *in_frame;
    float *gmasks;
    checkCudaErrors(cudaMalloc((void **) &gmasks, gp->n * gp->w * gp->h * sizeof(float)));
    checkCudaErrors(cudaMemcpy(gmasks, loc3D(gp->hmasks, gp->total_n, gp->h, gp->w, gp->n_start, 0, 0), gp->n * gp->w * gp->h * sizeof(float), cudaMemcpyHostToDevice));

    NppiSize size = {.width = gp->w, .height = gp->h};

    double *fsum;
    checkCudaErrors(cudaMalloc((void **) &fsum, gp->n * sizeof(double)));

    Npp8u *pDeviceBuffer;
    int nBufferSize;
    nppiSumGetBufferHostSize_32f_C1R(size, &nBufferSize);
    checkCudaErrors(cudaMalloc((void **)&pDeviceBuffer, nBufferSize));
	int pitch = gp->w * sizeof(float);

    for(int i = 0; i < gp->n; ++i) {
    	float *mask = loc3D(gmasks, gp->n, gp->h, gp->w, i, 0, 0);
    	nppiSum_32f_C1R(mask, pitch, size, pDeviceBuffer, &fsum[i]);	
    }

    for(int i = 0; i < gp->n; ++i) {
    	float *gmask = loc3D(gmasks, gp->n, gp->h, gp->w, i, 0, 0);
    	for(int j = 0; j < gp->t; j++) {
			in_frame = loc3D(gimg, gp->t, gp->h, gp->w, j, 0, 0);
			double *go = loc2D(gout_double, gp->n, gp->t, i, j);
			nppiMul_32f_C1R(in_frame, pitch, gmask, pitch, out_frame, pitch, size);
			nppiSum_32f_C1R(out_frame, pitch, size, pDeviceBuffer, go);
    	}
    }

    conv_double_float_arr<<<gp->n, 1024>>>(gout_double, gout, gp->t);
    divide_by_C<<<gp->n, 1024>>>(gout, fsum, gp->t);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(loc2D(gp->hout, gp->total_n, gp->t, gp->n_start, 0), gout, gp->n * gp->t * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(gimg);
	cudaFree(out_frame);
	cudaFree(gout_double);
	cudaFree(gout);
	cudaFree(gmasks);
	cudaFree(fsum);
	cudaFree(pDeviceBuffer);


}


float* get_signals(float *image, float *masks, int t, int h, int w, int n)
{
	float *out = (float *) malloc(n * t * sizeof(float));
	int gpu_n;
    checkCudaErrors(cudaGetDeviceCount(&gpu_n));
    fprintf(stderr, "========================= GPU Count: %d\n", gpu_n);
    std::vector<std::thread> threads(gpu_n);
    int *splits_n = get_n_splits(n, gpu_n);
    gpuPostprocess *gp = new gpuPostprocess[gpu_n];

	for(int i = 0; i < gpu_n; ++i) {
		if(i != 0) {
            gp[i].n_start = gp[i-1].n_end;
        } else {
        	gp[i].n_start = 0;
        }
        gp[i].gpu_device_id = i;
        gp[i].t = t;
        gp[i].h = h;
        gp[i].w = w;
        gp[i].total_n = n;
        gp[i].n_end = gp[i].n_start + splits_n[i];
        gp[i].n = gp[i].n_end - gp[i].n_start;
        gp[i].himg = image;
        gp[i].hout = out;
        gp[i].hmasks = masks;
		threads[i] = std::thread(process_gpu_threads, &gp[i]);
	}

		
	for (auto& th : threads) {
        th.join();
    }

    delete [] gp;
    free(splits_n);

	return out;
}


__global__ 
void __exp_spread(float *data, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride) {
        if(i < N)
            data[i] = log(1 - (data[i] * 0.99999)) * -1;
    }
}

float* _exp_spread(float *img, int t, int h, int w)
{
    int N = t * h * w;
    float *out = (float *) malloc(N * sizeof(float));
    float *gdata;
    checkCudaErrors(cudaMalloc((void **) &gdata, N * sizeof(float)));
    checkCudaErrors(cudaMemcpy(gdata, img, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 A(h, w);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    __exp_spread<<<numBlocks, blockSize>>>(gdata, N);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(out, gdata, N * sizeof(float), cudaMemcpyDeviceToHost));

    return out;
}
