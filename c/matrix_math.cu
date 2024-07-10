// matrix_add.cu
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void matrixAddKernel(double* a, double* b, double* result, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

extern "C" void matrix_add(double* a, double* b, double* result, int rows, int cols) {
    int size = rows * cols;
    double* d_a;
    double* d_b;
    double* d_result;

    cudaMalloc((void**)&d_a, size * sizeof(double));
    cudaMalloc((void**)&d_b, size * sizeof(double));
    cudaMalloc((void**)&d_result, size * sizeof(double));

    cudaMemcpy(d_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(double), cudaMemcpyHostToDevice);

    //ToDo: evaluate whether we really need 256 threadsPerBlock
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, size);

    cudaMemcpy(result, d_result, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

__global__ void checkZeroKernel(double* b, int size, int* errorFlag) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size && b[idx] == 0) {
        *errorFlag = 1;
    }
}

__global__ void matrixDivideKernel(double* a, double* b, double* result, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] / b[idx];
    }
}

extern "C" int matrix_divide(double* a, double* b, double* result, int rows, int cols) {
    int size = rows * cols;
    double* d_a;
    double* d_b;
    double* d_result;
    int* d_errorFlag;
    int h_errorFlag = 0;

    cudaMalloc((void**)&d_a, size * sizeof(double));
    cudaMalloc((void**)&d_b, size * sizeof(double));
    cudaMalloc((void**)&d_result, size * sizeof(double));
    cudaMalloc((void**)&d_errorFlag, sizeof(int));

    cudaMemcpy(d_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_errorFlag, &h_errorFlag, sizeof(int), cudaMemcpyHostToDevice);

    // Check for zero values in divisor
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    checkZeroKernel<<<blocksPerGrid, threadsPerBlock>>>(d_b, size, d_errorFlag);

    cudaMemcpy(&h_errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_errorFlag != 0) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        cudaFree(d_errorFlag);
        return h_errorFlag;  // Division by zero detected
    }

    // Perform the division if no zero values are found
    matrixDivideKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, size);

    cudaMemcpy(result, d_result, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cudaFree(d_errorFlag);

    return h_errorFlag;
}

__global__ void matrixMultiplyKernel(double* a, double* b, double* result, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int row = idx / cols;
    int col = idx % cols;
    if (row < rows && col < cols) {
        double value = 0.0;
        for (int k = 0; k < cols; ++k) {
            value += a[row * cols + k] * b[k * cols + col];
        }
        result[row * cols + col] = value;
    }
}

extern "C" void matrix_multiply(double* a, double* b, double* result, int rows, int cols) {
    int size = rows * cols;
    double* d_a;
    double* d_b;
    double* d_result;

    cudaMalloc((void**)&d_a, size * sizeof(double));
    cudaMalloc((void**)&d_b, size * sizeof(double));
    cudaMalloc((void**)&d_result, size * sizeof(double));

    cudaMemcpy(d_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, rows, cols);

    cudaMemcpy(result, d_result, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}