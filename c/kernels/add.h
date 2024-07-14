#include <cuda_runtime.h>

/*
 * CUDA Kernel function: Matrix Addition
 *
 *      c = a + b
 *
 *      Note: we do not check dimensions of the matrices for performance
 *            and we expect that the caller would have done this instead.
 */
__global__ void matrixAddKernel(double *a, double *b, double *c, int size, int *gpuError) {
    const int boundCheckError = -1;

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        c[index] = a[index] + b[index];
        return;
    }

    atomicExch(gpuError, boundCheckError);
}
