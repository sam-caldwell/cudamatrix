#ifndef INITIALIZE_GPU_MATRIX
#define INITIALIZE_GPU_MATRIX

#include <cuda_runtime.h>
#include "exceptions.h"
#include <iostream>

/*
 * GPU Matrix Initializer
 *
 *   - Allocate memory for the GPU.
 *   - If copyData is true, copy the HostMatrix into the GPU.
 */
void initializeGpuMatrix(double* hostMatrix, double*& gpuMatrix, int size, bool copyData){
    cudaError_t err;

    cudaDeviceSynchronize();

    std::cout << "memory allocation start" << std::endl;
    err = cudaMallocManaged((void**)&gpuMatrix, size *sizeof(double), cudaMemAttachGlobal);
    if (err != cudaSuccess) throw CudaException(err);

    if (copyData){
        std::cout << "memory copy start" << std::endl;
        err = cudaMemcpy(gpuMatrix, hostMatrix, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw CudaException(err);
    }

    err = cudaGetLastError(); //capture last error now before we mess it up accidentally.
    if (err != cudaSuccess) {
        std::cout << "initializeGpuMatrix detected error in cudaGetLastError()" << std::endl;
        throw CudaException(err);
    }

    std::cout << "initializeGpuMatrix() done" << std::endl;
}
#endif