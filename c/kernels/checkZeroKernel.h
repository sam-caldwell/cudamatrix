#include <cuda_runtime.h>

__global__ void checkZeroKernel(double* b, int size, int* errorFlag) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size && b[idx] == 0) {
        *errorFlag = 1;
    }
}