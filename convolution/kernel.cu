
#include "common.h"

#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {
	
	// Shared memory for input tile
    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM];

    // Calculate global thread coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int global_x = blockIdx.x * OUT_TILE_DIM + tx - FILTER_RADIUS;
    int global_y = blockIdx.y * OUT_TILE_DIM + ty - FILTER_RADIUS;

	// Use all threads to collaboratively load the input tile into shared memory
    if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
        tile[ty][tx] = input[global_y * width + global_x];
    } else {
        tile[ty][tx] = 0.0f;  // Zero-padding for out-of-bounds elements
    }
		// Ensure all threads have finished loading before computation starts

    __syncthreads(); 

    // Compute the output for the center threads
    if (tx >= FILTER_RADIUS && tx < IN_TILE_DIM - FILTER_RADIUS &&
        ty >= FILTER_RADIUS && ty < IN_TILE_DIM - FILTER_RADIUS) {
        
        float sum = 0.0f;

        // Apply the convolution filter
        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
                sum += tile[ty + i][tx + j] * filter_c[FILTER_RADIUS + i][FILTER_RADIUS + j];
            }
        }

        // Store the computed value in the output
        int output_x = global_x;
        int output_y = global_y;
        if (output_x < width && output_y < height) {
            output[output_y * width + output_x] = sum;
        }
    }
}

void copyFilterToGPU(float filter[][FILTER_DIM]) {
	// Copy the convolution filter to constant memory (faster access for all threads)
    cudaMemcpyToSymbol(filter_c, filter, sizeof(float) * FILTER_DIM * FILTER_DIM);
}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {
	// confgure the number of threads/block (must be large enough to load the entire input tile)
    dim3 threads(IN_TILE_DIM, IN_TILE_DIM);
    dim3 blocks((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_tiled_kernel<<<blocks, threads>>>(input_d, output_d, width, height);
}

