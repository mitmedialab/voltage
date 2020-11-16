#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "wrapper.hpp"
#include <cublas_v2.h>
#include <nppcore.h>
#include <nppi.h>
#include <nppdefs.h>

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


// A = mxk
// B = kxn
void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) 
{
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, n, A, k, beta, C, n);
}


void gpu_blas_mmul_sp(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) 
{
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, alpha, B, n, A, k, beta, C, n);
}


// B = At
void gpu_blas_transpose(cublasHandle_t &handle, float *A, int m, int n, float *B)
{
	float const alpha = 1.0;
	float const beta = 0.0;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, n, &beta, A, m, B, m);
}

// void saxpy(cublasHandle_t &handle, float a, float *x, float *y, int N)
// {
// 	float const alpha = 1.0;
// 	cublasSaxpy(handle, N, &alpha, x, 1, y, 1);
// }


__global__
void cdnmf(float *W, float *XHt, float *HHt, float *violation, int ns, int nc)
{
	float grad = 0;
	float *Wv = gloc2D(W, ns, nc, blockIdx.x, 0);
	float pg;
	float *pgs = violation + blockIdx.x;
	float *Xhtv = gloc2D(XHt, ns, nc, blockIdx.x, 0);
	*pgs = 0;
	float hess;
	float *hhtv;
	for(int t = 0; t < nc; ++t) {

		grad = -Xhtv[t];
		hhtv = gloc2D(HHt, nc, nc, t, 0);
		for(int r = 0; r < nc; ++r) {
			grad += hhtv[r] * Wv[r];
		}
		pg = grad;
		if(Wv[t] == 0) {
			if(pg > 0)
				pg = 0;
		}
		*pgs += fabsf(pg);

		hess = hhtv[t];
		if(hess != 0) {
			Wv[t] -= (grad/hess);
			if(Wv[t] < 0)
				Wv[t] = 0;
		}
	}
}


__global__
void print_matrix(float *X, int m, int n)
{
	for(int i = 0; i < m; ++i) {
		for(int j = 0; j < n; ++j) {
			printf("%f ", X[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
}


double _update_coordinate_descent(cublasHandle_t &handle, float *X, float *W, float *Ht, float *Htt, float *HHt, float *XHt, float *WHHt, float *GW, float *pg, Npp8u *pDeviceBuffer, double *viol_g, int ns, int nf, int nc)
{
	// Compute Htt
	gpu_blas_transpose(handle, Ht, nf, nc, Htt); // No need, use gemm
	gpu_blas_mmul(handle, Htt, Ht, HHt, nc, nf, nc);

	gpu_blas_mmul(handle, X, Ht, XHt, ns, nf, nc);

	// gpu_blas_mmul_sp(handle, W, HHt, WHHt, ns, nc, nc);

	cdnmf<<<ns, 1>>>(W, XHt, HHt, pg, ns, nc);

	NppiSize size = {.width = 1, .height = ns};
	int pitch = 1 * sizeof(float);
    nppiAbs_32f_C1IR(pg, pitch, size);
	nppiSum_32f_C1R(pg, pitch, size, pDeviceBuffer, viol_g);
	cudaDeviceSynchronize();
	return *viol_g;
}

int coordinate_descent(float *X, float *W, float *H, int ns, int nf, int nc, int max_iterations, double tolerance)
{
	float *X_g;
	float *W_g;
	float *H_g;
	float *Ht_g;
	float *Wt_g;
	float *Htt_g;
	float *Xt_g;
	float *HHt_g;
	float *XHt_g;
	float *WHHt_g;
	float *GW_g;
	float *WWt_g;
	float *XtW_g;
	float *HtWWt_g;
	float *GH_g;
	float *pg_W;
	float *pg_H;
	double *viol_g;

	Npp8u *pDeviceBuffer1;
	Npp8u *pDeviceBuffer2;
	int nBufferSize;

	// For pg_W sum
	NppiSize size1 = {.width = 1, .height = ns};
	nppiSumGetBufferHostSize_32f_C1R(size1, &nBufferSize);
	checkCudaErrors(cudaMalloc((void **)&pDeviceBuffer1, nBufferSize));

	NppiSize size2 = {.width = 1, .height = nf};
	nppiSumGetBufferHostSize_32f_C1R(size2, &nBufferSize);
	checkCudaErrors(cudaMalloc((void **)&pDeviceBuffer2, nBufferSize));

	checkCudaErrors(cudaMalloc((void **) &X_g, ns * nf * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &W_g, ns * nc * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &Wt_g, nc * ns * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &H_g, nc * nf * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &Xt_g, nf * ns * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &Ht_g, nf * nc * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &Htt_g, nc * nf * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &HHt_g, nc * nc * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &XHt_g, ns * nc * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &WHHt_g, ns * nc * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &GW_g, ns * nc * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &WWt_g, nc * nc * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &XtW_g, nf * nc * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &HtWWt_g, nf * nc * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &GH_g, nf * nc * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &pg_W, ns * 1 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &pg_H, nf * 1 * sizeof(float)));
	checkCudaErrors(cudaMallocManaged((void **) &viol_g, sizeof(double)));

	checkCudaErrors(cudaMemcpy(X_g, X, ns * nf * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(W_g, W, ns * nc * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(H_g, H, nc * nf * sizeof(float), cudaMemcpyHostToDevice));

	cublasHandle_t handle;
	

	cublasCreate(&handle);
	gpu_blas_transpose(handle, H_g, nc, nf, Ht_g);
	gpu_blas_transpose(handle, X_g, ns, nf, Xt_g);

	double violation = 0;
	double violation_init = 0;
	int n_iter;
	for(n_iter = 0; n_iter < max_iterations; ++n_iter) {
		violation = 0;
		violation += _update_coordinate_descent(handle, X_g, W_g, Ht_g, Htt_g, HHt_g, XHt_g, WHHt_g, GW_g, pg_W, pDeviceBuffer1, viol_g, ns, nf, nc);
		violation += _update_coordinate_descent(handle, Xt_g, Ht_g, W_g, Wt_g, WWt_g, XtW_g, HtWWt_g, GH_g, pg_H, pDeviceBuffer2, viol_g, nf, ns, nc);

		if(n_iter == 0) {
			violation_init = violation;
			if(violation_init == 0)
				break;
		}
		if((violation/violation_init) <= tolerance) {
			break;
		}
	}

	gpu_blas_transpose(handle, Ht_g, nf, nc, H_g);

	// Copy to Host buffers
	checkCudaErrors(cudaMemcpy(W, W_g, ns * nc * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(H, H_g, nc * nf * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(pDeviceBuffer1);
	cudaFree(pDeviceBuffer2);
	cudaFree(X_g);
	cudaFree(W_g);
	cudaFree(Wt_g);
	cudaFree(H_g);
	cudaFree(Xt_g);
	cudaFree(Ht_g);
	cudaFree(Htt_g);
	cudaFree(HHt_g);
	cudaFree(XHt_g);
	cudaFree(WHHt_g);
	cudaFree(GW_g);
	cudaFree(WWt_g);
	cudaFree(XtW_g);
	cudaFree(HtWWt_g);
	cudaFree(GH_g);
	cudaFree(pg_W);
	cudaFree(pg_H);
	cudaFree(viol_g);
	return n_iter;
}