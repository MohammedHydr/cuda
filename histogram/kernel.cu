
#include "common.h"
#include "timer.h"
#define COARSE_FACTOR 32

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

	__shared__ int b_s[NUM_BINS];
	
	if (threadIdx.x <NUM_BINS){
		b_s[threadIdx.x]=0;
	}
	
	__syncthreads();

	unsigned int i= blockIdx.x * blockDim.x + threadIdx.x;

	if (i< width*height)
	{
		unsigned char a= image[i];
		atomicAdd(&b_s[a],1);
	}

	__syncthreads();
	
	if (threadIdx.x <NUM_BINS && (b_s[threadIdx.x]!=0))
	{
		atomicAdd(&bins[threadIdx.x], b_s[threadIdx.x]);	
	}
}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

	unsigned int numThreadsPerBlock= 1024;
	unsigned int numBlocks= (width*height + numThreadsPerBlock -1)/numThreadsPerBlock;
	
	histogram_private_kernel<<<numThreadsPerBlock, numBlocks>>>(image_d, bins_d, width, height);

}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

	__shared__ unsigned int bin_s[NUM_BINS];
	if (threadIdx.x <NUM_BINS)
	{
		bin_s[threadIdx.x]=0;
	}

	__syncthreads();
	
	unsigned int i= blockIdx.x * blockDim.x* COARSE_FACTOR + threadIdx.x;
	
	for (int j=0; j<COARSE_FACTOR; ++j)
	{
		if(j*blockDim.x + i < width*height)
		{
			atomicAdd(&bin_s[image[j*blockDim.x+i]],1);
		}
	}

	__syncthreads();

	if (threadIdx.x <NUM_BINS && (bin_s[threadIdx.x]!=0))
	{
		atomicAdd(&bins[threadIdx.x], bin_s[threadIdx.x]);
	}

}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {
	unsigned int numThreadsPerBlock=1024;
	unsigned int numBlocks= (width*height +numThreadsPerBlock-1)/numThreadsPerBlock;
	histogram_private_coarse_kernel<<<numThreadsPerBlock,numBlocks>>>(image_d,bins_d,width,height);
}

