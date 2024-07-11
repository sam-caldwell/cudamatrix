#include <cuda_runtime.h>

/*
 * CUDA Kernel function: Matrix Multiply
 *
 *      c = a x b
 */
__global__ void matrixMultiplyKernel(double* a, double* b, double* c, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int row = idx / cols;
    int col = idx % cols;
    if (row < rows && col < cols) {
        double value = 0.0;
        for (int k = 0; k < cols; ++k) {
            value += a[row * cols + k] * b[k * cols + col];
        }
        c[row * cols + col] = value;
    }
}
