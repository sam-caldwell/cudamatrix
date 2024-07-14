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
    err = cudaMallocManaged((void**)&gpuMatrix, size *sizeof(double), cudaMemAttachGlobal);
    if (err != cudaSuccess) throw CudaException(err);

    if (copyData){
        err = cudaMemcpy(gpuMatrix, hostMatrix, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw CudaException(err);
    }
}
#endif