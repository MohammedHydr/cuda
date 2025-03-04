#include "common.h"

#include "timer.h"

__global__ void rgb2gray_kernel (unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, 
					unsigned int width, unsigned int height) {

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i = row * width + col;
	
	if (row < height && col < width){
		gray[i] = red[i]*3/10 + green[i]*6/10 + blue[i]/10;
	}
}

void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, 
					unsigned int width, unsigned int height) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
	unsigned char *red_d, *green_d, *blue_d, *gray_d;
	cudaMalloc((void **) &red_d, height*width*sizeof(unsigned char));
	cudaMalloc((void **) &green_d, height*width*sizeof(unsigned char));
	cudaMalloc((void **) &blue_d, height*width*sizeof(unsigned char));
	cudaMalloc((void **) &gray_d, height*width*sizeof(unsigned char));

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO
	cudaMemcpy(red_d, red, height*width*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(green_d, green, height*width*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(blue_d, blue, height*width*sizeof(unsigned char), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO					  x, y , z 	by default z is 1					
	dim3 numThreadsPerBlock(32, 32);
	dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
	rgb2gray_kernel <<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, width, height);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO
	cudaMemcpy(gray, gray_d, height*width*sizeof(unsigned char), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO

	cudaFree(red_d);
	cudaFree(green_d);
	cudaFree(blue_d);
	cudaFree(gray_d);




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}