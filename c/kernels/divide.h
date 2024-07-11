#include <cuda_runtime.h>

/*
 * CUDA Kernel function: Matrix Divide
 *
 *      c = a / b
 *
 *      if the divisor is 0, return DivisionByZero error via gpu_errorFlag
 *
 *      Note: we do not check dimensions of the matrices for performance
 *            and we expect that the caller would have done this instead.
 */
__global__ void matrixDivideKernel(double* a, double* b, double* c, int size, int *gpuError) {
    const int divByZero = -1;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        if (b[idx] == 0) {
            *gpuError = divByZero;
        }else{
            c[idx] = a[idx] / b[idx];
            *gpuError = 0;
        }
    }
}
