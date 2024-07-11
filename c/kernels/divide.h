#include <cuda_runtime.h>

/*
 * CUDA Kernel function: Matrix Divide
 *
 *      c = a / b
 */
__global__ void matrixDivideKernel(double* a, double* b, double* c, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] / b[idx];
    }
}
