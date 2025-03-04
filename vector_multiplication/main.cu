
#include "common.h"
#include "timer.h"

void axpb_cpu(double* x, double* y, double a, double b, unsigned int M) {
    for(unsigned int i = 0; i < M; ++i) {
        y[i] = a*x[i] + b;
    }
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int M = (argc > 1)?(atoi(argv[1])):32000000;
    double a = rand();
    double b = rand();
    double* x = (double*) malloc(M*sizeof(double));
    for (unsigned int i = 0; i < M; ++i) {
        x[i] = rand();
    }
    double* y_cpu = (double*) malloc(M*sizeof(double));
    double* y_gpu = (double*) malloc(M*sizeof(double));

    // Compute on CPU
    startTime(&timer);
    axpb_cpu(x, y_cpu, a, b, M);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Compute on GPU
    startTime(&timer);
    axpb_gpu(x, y_gpu, a, b, M);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify result
    for(unsigned int i = 0; i < M; ++i) {
        double diff = (y_cpu[i] - y_gpu[i])/y_cpu[i];
        const double tolerance = 1e-9;
        if(diff > tolerance || diff < -tolerance) {
            printf("Mismatch at index %u (CPU result = %e, GPU result = %e)\n", i, y_cpu[i], y_gpu[i]);
            exit(0);
        }
    }

    // Free memory
    free(x);
    free(y_cpu);
    free(y_gpu);

    return 0;

}

