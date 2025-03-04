#include "common.h"
#include "timer.h"

void rgb2gray_cpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, 
                   unsigned int width, unsigned int height) {
    for (unsigned int i = 0; i < height * width; ++i) {
        gray[i] = red[i] * 3 / 10 + green[i] * 6 / 10 + blue[i] / 10;
    }
}

int main(int argc, char** argv) {
    cudaDeviceSynchronize();

    // Timer initialization
    Timer timer;

    // Image dimensions
    unsigned int width = (argc > 1) ? atoi(argv[1]) : 1920;
    unsigned int height = (argc > 2) ? atoi(argv[2]) : 1080;

    unsigned int size = width * height * sizeof(unsigned char);

    // Allocate host memory
    unsigned char* red = (unsigned char*) malloc(size);
    unsigned char* green = (unsigned char*) malloc(size);
    unsigned char* blue = (unsigned char*) malloc(size);
    unsigned char* gray_cpu = (unsigned char*) malloc(size);
    unsigned char* gray_gpu = (unsigned char*) malloc(size);

    // Initialize image data
    for (unsigned int i = 0; i < width * height; ++i) {
        red[i] = rand() % 256;
        green[i] = rand() % 256;
        blue[i] = rand() % 256;
    }

    // Compute on CPU
    startTime(&timer);
    rgb2gray_cpu(red, green, blue, gray_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Compute on GPU
    startTime(&timer);
    rgb2gray_gpu(red, green, blue, gray_gpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify result
    for (unsigned int i = 0; i < width * height; ++i) {
        double diff = (gray_cpu[i] - gray_gpu[i]) / (double) gray_cpu[i];
        const double tolerance = 1e-9;
        if (diff > tolerance || diff < -tolerance) {
            printf("Mismatch at index %u (CPU result = %u, GPU result = %u)\n", i, gray_cpu[i], gray_gpu[i]);
            exit(0);
        }
    }

    // Free memory
    free(red);
    free(green);
    free(blue);
    free(gray_cpu);
    free(gray_gpu);

    return 0;
}
