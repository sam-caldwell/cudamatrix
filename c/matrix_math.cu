// matrix_add.cu
#include <cuda_runtime.h>
#include <iostream>
#include "exceptions.h"
#include "kernels/checkZeroKernel.h"
#include "kernels/add.h"
#include "kernels/divide.h"
#include "kernels/multiply.h"

#define CUDA_FREE_ALL(a,b,c) \
    cudaFree(a); \
    cudaFree(b); \
    cudaFree(c);
/*
 * CUDA interface function: Add two Matrices
 *
 *      c = a + b, return error_code (-1) or success (0)
 */
extern "C" int matrix_add(double* a, double* b, double* c, int rows, int cols) {
    int size = rows * cols;
    double* gpu_a = nullptr;
    double* gpu_b = nullptr;
    double* gpu_c = nullptr;

    try {
        cudaError_t err;
        // Allocate device memory
        err = cudaMalloc((void**)&gpu_a, size * sizeof(double));
        if (err != cudaSuccess) throw CudaMallocExceptionGpuA();

        err = cudaMalloc((void**)&gpu_b, size * sizeof(double));
        if (err != cudaSuccess) throw CudaMallocExceptionGpuB();

        err = cudaMalloc((void**)&gpu_c, size * sizeof(double));
        if (err != cudaSuccess) throw CudaMallocExceptionGpuC();

        // Copy data to device
        err = cudaMemcpy(gpu_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw CudaMemcpyExceptionA();

        err = cudaMemcpy(gpu_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw CudaMemcpyExceptionB();

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_a, gpu_b, gpu_c, size);

        err = cudaGetLastError();
        if (err != cudaSuccess) throw KernelLaunchException();

        // Copy result back to host
        err = cudaMemcpy(c, gpu_c, size * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw CudaMemcpyExceptionC();

    } catch (const CudaMallocExceptionGpuA& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -1;
    } catch (const CudaMallocExceptionGpuB& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -2;
    } catch (const CudaMallocExceptionGpuC& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -3;
    } catch (const CudaMallocExceptionErrorFlag& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -4;
    } catch (const CudaMemcpyExceptionA& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -5;
    } catch (const CudaMemcpyExceptionB& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -6;
    } catch (const CudaMemcpyExceptionC& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -7;
    } catch (const KernelLaunchException& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -8;
    } catch (const DivisionByZeroException& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -9;
    } catch (const std::runtime_error& e) {
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
         return -1;
     }
    CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
    return 0;  // Return success code
}

/*
 * CUDA interface function: Divide two Matrices
 *
 *      c = a / b, return error_code (-1) or success (0)
 */
extern "C" int matrix_divide(double* a, double* b, double* c, int rows, int cols) {
    int size = rows * cols;
    double* gpu_a = nullptr;
    double* gpu_b = nullptr;
    double* gpu_c = nullptr;

    try {
        int* d_errorFlag = 0;
        cudaError_t err;
        // Allocate device memory
        err = cudaMalloc((void**)&gpu_a, size * sizeof(double));
        if (err != cudaSuccess) throw CudaMallocExceptionGpuA();

        err = cudaMalloc((void**)&gpu_b, size * sizeof(double));
        if (err != cudaSuccess) throw CudaMallocExceptionGpuB();

        err = cudaMalloc((void**)&gpu_c, size * sizeof(double));
        if (err != cudaSuccess) throw CudaMallocExceptionGpuC();

        err = cudaMalloc((void**)&d_errorFlag, sizeof(int));
        if (err != cudaSuccess) throw CudaMallocExceptionErrorFlag();

        // Copy data to device
        err = cudaMemcpy(gpu_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw CudaMemcpyExceptionA();

        err = cudaMemcpy(gpu_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw CudaMemcpyExceptionB();

        int h_errorFlag = 0;
        err = cudaMemcpy(d_errorFlag, &h_errorFlag, sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw CudaMemcpyExceptionErrorFlag();

        // Check for zero values in divisor
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        checkZeroKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_b, size, d_errorFlag);

        err = cudaMemcpy(&h_errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw CudaMemcpyExceptionErrorFlag();

        if (h_errorFlag != 0) {
            throw DivisionByZeroException();
        }

        // Perform the division if no zero values are found
        matrixDivideKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_a, gpu_b, gpu_c, size);

        err = cudaGetLastError();
        if (err != cudaSuccess) throw KernelLaunchException();

        // Copy result back to host
        err = cudaMemcpy(c, gpu_c, size * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw CudaMemcpyExceptionC();

    } catch (const CudaMallocExceptionGpuA& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -1;
     } catch (const CudaMallocExceptionGpuB& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -2;
     } catch (const CudaMallocExceptionGpuC& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -3;
     } catch (const CudaMallocExceptionErrorFlag& e){
         CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
         return -4;
     } catch (const CudaMemcpyExceptionA& e){
         CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
         return -5;
     } catch (const CudaMemcpyExceptionB& e){
         CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
         return -6;
     } catch (const CudaMemcpyExceptionC& e){
         CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
         return -7;
     } catch (const KernelLaunchException& e){
         CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
         return -8;
     } catch (const DivisionByZeroException& e){
         CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
         return -9;
     } catch (const std::runtime_error& e) {
         CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
         return -1;
    }

    // Free memory
    CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
    return 0;  // Return success code
}

/*
 * CUDA interface function: Multiply two Matrices
 *
 *      c = a * b, return error_code (-1) or success (0)
 */
extern "C" int matrix_multiply(double* a, double* b, double* c, int rows, int cols) {
    int size = rows * cols;
    double* gpu_a = nullptr;
    double* gpu_b = nullptr;
    double* gpu_c = nullptr;
    int* d_errorFlag = 0;

    try {
        cudaError_t err;
        // Allocate device memory
        err = cudaMalloc((void**)&gpu_a, size * sizeof(double));
        if (err != cudaSuccess) throw CudaMallocExceptionGpuA();

        err = cudaMalloc((void**)&gpu_b, size * sizeof(double));
        if (err != cudaSuccess) throw CudaMallocExceptionGpuB();

        err = cudaMalloc((void**)&gpu_c, size * sizeof(double));
        if (err != cudaSuccess) throw CudaMallocExceptionGpuC();

        err = cudaMalloc((void**)&d_errorFlag, sizeof(int));
        if (err != cudaSuccess) throw CudaMallocExceptionErrorFlag();

        // Copy data to device
        err = cudaMemcpy(gpu_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw CudaMemcpyExceptionA();

        err = cudaMemcpy(gpu_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw CudaMemcpyExceptionA();

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_a, gpu_b, gpu_c, rows, cols);

        err = cudaGetLastError();
        if (err != cudaSuccess) throw KernelLaunchException();

        // Copy result back to host
        err = cudaMemcpy(c, gpu_c, size * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw CudaMemcpyExceptionC();

    } catch (const CudaMallocExceptionGpuA& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -1;
    } catch (const CudaMallocExceptionGpuB& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -2;
    } catch (const CudaMallocExceptionGpuC& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -3;
    } catch (const CudaMallocExceptionErrorFlag& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -4;
    } catch (const CudaMemcpyExceptionA& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -5;
    } catch (const CudaMemcpyExceptionB& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -6;
    } catch (const CudaMemcpyExceptionC& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -7;
    } catch (const KernelLaunchException& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -8;
    } catch (const DivisionByZeroException& e){
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -9;
    } catch (const std::runtime_error& e) {
        CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);
        return -1;
    }
    // Free memory
    CUDA_FREE_ALL(gpu_a,gpu_b,gpu_c);

    return 0;  // Return success code
}