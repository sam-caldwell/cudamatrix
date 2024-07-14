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
    std::cout << "initializeGpuErrorFlag() start" << std::endl;
    cudaError_t err;

    if (gpuError == nullptr) throw CudaException(cudaErrorUnknown);

    err = cudaMallocManaged((void**)&gpuError, sizeof(int), cudaMemAttachGlobal);

    if (err != cudaSuccess) throw CudaException(err);

    err = cudaMemset(gpuError, 0, sizeof(int));

    if (err != cudaSuccess) throw CudaException(err);

    std::cout << "initializeGpuErrorFlag() end" << std::endl;

}

#endif
