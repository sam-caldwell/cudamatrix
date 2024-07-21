#ifndef INITIALIZE_GPU_ERROR_FLAG
#define INITIALIZE_GPU_ERROR_FLAG

#include <iostream>
#include <cuda_runtime.h>
#include "exceptions.h"

/*
 * initialize the GPU Error flag variable.
 *
 *   This will be used to capture errors during runtime
 *   and return them to the host via captureGpuErrors().
 */
void initializeGpuErrorFlag(int* gpuError){
    cudaError_t err;

    std::cout << "initializeGpuErrorFlag() start" << std::endl;

    if (gpuError == nullptr) throw CudaException(cudaErrorUnknown);

    err = cudaMallocManaged((void**)&gpuError, sizeof(int), cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(err) << std::endl;
        throw CudaException(err);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cudaGetLastError after cudaMallocManaged: " << cudaGetErrorString(err) << std::endl;
        throw CudaException(err);
    }

    err = cudaMemset(gpuError, 0, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << std::endl;
        throw CudaException(err);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cudaGetLastError after cudaMemset: " << cudaGetErrorString(err) << std::endl;
        throw CudaException(err);
    }

    std::cout << "initializeGpuErrorFlag() end" << std::endl;
}



#endif
