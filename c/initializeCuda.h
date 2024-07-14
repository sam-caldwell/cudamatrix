#ifndef INITIALIZE_CUDA_H
#define INITIALIZE_CUDA_H

#include <iostream>
#include <cuda_runtime.h>
#include "exceptions.h"
/*
 * Select the Cuda device or throw an exception if there is none.
 */
void initializeCuda(int desiredDevice) {
    int deviceCount;
    cudaError_t err;

    err = cudaGetDeviceCount(&deviceCount); // Get number of CUDA devices
    if (err != cudaSuccess) {
        throw CudaException(err);
    }

    if (deviceCount == 0) {
        throw CudaException(cudaErrorInitializationError);
    }

    if (desiredDevice >= deviceCount) {
        throw CudaException(cudaErrorNoDevice);
    }

    err = cudaSetDevice(desiredDevice);
    if (err != cudaSuccess) {
        throw CudaException(err);
    }

    std::cout << "CUDA successfully initialized" << std::endl;

}

#endif
