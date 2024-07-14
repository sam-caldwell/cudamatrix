#include <cuda_runtime.h>

/*
 * CUDA Kernel function: Matrix Addition
 *
 *      c = a + b
 *
 *      Note: we do not check dimensions of the matrices for performance
 *            and we expect that the caller would have done this instead.
 */
//__global__ void matrixAddKernel(double* a, double* b, double* c, int size, int *gpuError) {
//    printf("matrixAddKernel()\n");
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
//    if (idx < size) {
//        c[idx] = a[idx] + b[idx];
//        printf("Thread %d: (idx:%d) a=%f  b=%f c=%f\n", threadIdx.x, idx, a[idx], b[idx], c[idx]);
//    }
//
//    *gpuError=0;
//}
__global__ void matrixAddKernel(double *a, double *b, double *c, int rows, int cols, int *gpuError) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && column < cols) {
        int index = row * cols + column;
        c[index] = a[index] + b[index];
        *gpuError=0;
        return;
    }
    *gpuError=-1;
    return;
}
