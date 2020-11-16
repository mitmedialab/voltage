#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "wrapper.hpp"
#include <thread>
#include <nppi.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include <stdexcept>

#define checkCudaErrors(func) {                                        \
    cudaError_t error = func;                                          \
    if (error != 0) {                                                   \
        throw std::runtime_error(std::string("Cuda failure Error") );     \
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
    return img + ((i * w) + j) * BYTESPERPIXEL; 
}


__global__ 
void sub_from_frame(float *img, int height, int width, double *val)
{
	img[blockIdx.x * width + threadIdx.x] = img[blockIdx.x * width + threadIdx.x] - *val;
}

__global__ 
void transpose(float *in, float *out, int t, int h, int w, int i)
{
	*gloc3D(out, h, w, t, threadIdx.x, i, blockIdx.x) = *gloc3D(in, t, h, w, blockIdx.x, threadIdx.x, i);
}

__global__
void median_diff(float *in, float *out, int t, int h, int w, int mw, int ot, int i)
{
	float *im = gloc3D(in, h, w, t, blockIdx.x, i, threadIdx.x*mw);
	thrust::sort(thrust::seq, im, im + mw);
	float *om = gloc3D(out, ot, h, w, threadIdx.x, blockIdx.x, i);
	*om = im[mw-1] - im[(int) (mw/2)];
}

__global__ 
void buffer_arrange(float *in, float *out, int ot, int h, int w)
{
	float *om = gloc3D_3C(out, ot, h, w, threadIdx.x, blockIdx.x, blockIdx.y, 0);
	om[0] = *gloc3D(in, ot, h, w, threadIdx.x, blockIdx.x, blockIdx.y);
	om[1] = *gloc3D(in, ot, h, w, threadIdx.x + 1, blockIdx.x, blockIdx.y);
	om[2] = *gloc3D(in, ot, h, w, threadIdx.x + 2, blockIdx.x, blockIdx.y);
}

__global__
void normalize_frames(float *img, int out_dim, float *min, float *max)
{
	img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 0] = img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 0] - *min;
	img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 0] = img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 0] / *max;
	img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 1] = img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 1] - *min;
	img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 1] = img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 1] / *max;
	img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 2] = img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 2] - *min;
	img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 2] = img[(blockIdx.x * out_dim + threadIdx.x) * 3 + 2] / *max;
}


float * preprocess_unet(motion_buffer_t *mbuf, int magnification, int *t_out, int out_dim)
{
	cudaSetDevice(0);
	
	NppiSize isize = {.width = mbuf->w, .height = mbuf->h};
	NppiRect iroi = {.x = 0, .y = 0, .width = mbuf->w, .height = mbuf->h};
	int ipitch = mbuf->w * sizeof(float);
	NppiPoint ioffset = {.x = 0, .y = 0};
	float* buf1;
	int alloc_w = mbuf->w;
	int alloc_h = mbuf->h;
	if(alloc_h < out_dim)
		alloc_h = out_dim;
	if(alloc_w < out_dim)
		alloc_w = out_dim;

	checkCudaErrors(cudaMalloc((void **)&buf1, mbuf->t * alloc_w * alloc_h * sizeof(float)));
	checkCudaErrors(cudaMemcpy(buf1, mbuf->out, mbuf->t * mbuf->h * mbuf->w * sizeof(float), cudaMemcpyHostToDevice));


	int opitch = mbuf->w * sizeof(float);
	NppiSize osize = {.width = mbuf->w, .height = mbuf->h};
	NppiRect oroi = {.x = 0, .y = 0, .width = mbuf->w, .height = mbuf->h};
	float* buf2;
	checkCudaErrors(cudaMalloc((void **)&buf2, mbuf->t * alloc_w * alloc_h * sizeof(float)));

	printf("T: %d H: %d W: %d\n", mbuf->t, mbuf->h, mbuf->w);

	NppiMaskSize mask;
	int median_window = 0;
	if(magnification == 40) {
		mask = NPP_MASK_SIZE_15_X_15;
		median_window = 40;
	} else if(magnification == 20) {
		mask = NPP_MASK_SIZE_13_X_13;
		median_window = 50;
	} else {
		mask = NPP_MASK_SIZE_13_X_13;
		median_window = 60;
	}
	// CUDA pointer 

	Npp8u *pDeviceBuffer;
	int nBufferSize;
	nppiMeanGetBufferHostSize_32f_C1R(osize, &nBufferSize);
	checkCudaErrors(cudaMalloc((void **)&pDeviceBuffer, nBufferSize));
	double *mean_at_frame;
	checkCudaErrors(cudaMalloc((void **)&mean_at_frame, mbuf->t * sizeof(double)));
	NppStatus result;
	for(int i = 0; i < mbuf->t; ++i) {
		Npp32f *frame1 = (Npp32f *)loc3D(buf1, mbuf->t, mbuf->h, mbuf->w, i, 0, 0);
		Npp32f *frame2 = (Npp32f *)loc3D(buf2, mbuf->t, mbuf->h, mbuf->w, i, 0, 0);
		result = nppiFilterGaussBorder_32f_C1R (frame1, ipitch, isize, ioffset, frame2, opitch, osize, mask, NPP_BORDER_REPLICATE);	
		if(result != 0) {
			printf("[NPP Error] %s-%s(%d) Code: %d\n", __FILE__, __func__, __LINE__, result);
			throw std::runtime_error(std::string("NPP failure Error") );
		}

		result = nppiMean_32f_C1R(frame2, opitch, osize, pDeviceBuffer, (Npp64f *)&mean_at_frame[i]);
		if(result != 0) {
			printf("[NPP Error] %s-%s(%d) Code: %d\n", __FILE__, __func__, __LINE__, result);
			throw std::runtime_error(std::string("NPP failure Error") );
		}

		sub_from_frame<<<mbuf->h, mbuf->w>>>(frame2, mbuf->h, mbuf->w, &mean_at_frame[i]);
	}
	cudaFree(pDeviceBuffer);
	int out_t = mbuf->t/median_window;
	*t_out = out_t - 2;
	float *out = (float *) malloc(*t_out * out_dim * out_dim * 3 * sizeof(float));
	dim3 HW(mbuf->h, mbuf->w);
	for(int i = 0; i < mbuf->w; ++i) {
		transpose<<<mbuf->t, mbuf->h>>>(buf2, buf1, mbuf->t, mbuf->h, mbuf->w, i);
	}
	for(int i = 0; i < mbuf->w; ++i) {
		median_diff<<<mbuf->h, out_t>>>(buf1, buf2, mbuf->t, mbuf->h, mbuf->w, median_window, out_t, i);
	}
	buffer_arrange<<<HW, out_t-2>>>(buf2, buf1, out_t, mbuf->h, mbuf->w);
	ipitch = mbuf->w * 3 * sizeof(float);
	opitch = out_dim * 3 * sizeof(float);
	osize = {.width = out_dim, .height = out_dim};
	oroi = {.x = 0, .y = 0, .width = out_dim, .height = out_dim};
	nppiMeanGetBufferHostSize_32f_C1R(osize, &nBufferSize);
	float *min_at_frame;
	checkCudaErrors(cudaMalloc((void **)&min_at_frame, *t_out * sizeof(float)));
	float *max_at_frame;
	checkCudaErrors(cudaMalloc((void **)&max_at_frame, *t_out * sizeof(float)));

	NppiSize osizeminmax = {.width = out_dim * 3, .height = out_dim};
	Npp8u *pDeviceBuffer1;
	nppiMeanGetBufferHostSize_32f_C1R(osizeminmax, &nBufferSize);
	checkCudaErrors(cudaMalloc((void **)&pDeviceBuffer1, nBufferSize));
	for(int i = 0; i < *t_out; ++i) {
		Npp32f *frame1 = (Npp32f *)loc3D_3C(buf1, *t_out, mbuf->h, mbuf->w, i, 0, 0);
		Npp32f *frame2 = (Npp32f *)loc3D_3C(buf2, *t_out, out_dim, out_dim, i, 0, 0);
		result = nppiResize_32f_C3R(frame1, ipitch, isize, iroi, frame2, opitch, osize, oroi, NPPI_INTER_LANCZOS);
		if(result != 0) {
			printf("[NPP Error] %s-%s(%d) Code: %d\n", __FILE__, __func__, __LINE__, result);
			throw std::runtime_error(std::string("NPP failure Error") );
		}
		result = nppiMinMax_32f_C1R(frame2, opitch, osizeminmax, &min_at_frame[i], &max_at_frame[i], pDeviceBuffer1);
		if(result != 0) {
			printf("[NPP Error] %s-%s(%d) Code: %d\n", __FILE__, __func__, __LINE__, result);
			throw std::runtime_error(std::string("NPP failure Error") );
		}
		normalize_frames<<<out_dim, out_dim>>>(frame2, out_dim, &min_at_frame[i], &max_at_frame[i]);
	}
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(out, buf2, *t_out * out_dim * out_dim * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(mean_at_frame);
	cudaFree(min_at_frame);
	cudaFree(max_at_frame);
	cudaFree(pDeviceBuffer);
	cudaFree(pDeviceBuffer1);
	cudaFree(buf1);
	cudaFree(buf2);
	// cudaDeviceReset();
	return out;
}
