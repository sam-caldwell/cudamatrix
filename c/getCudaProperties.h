#ifndef GET_CUDA_PROPERTIES
#define GET_CUDA_PROPERTIES

#include <cuda_runtime.h>
#include <iostream>
#include "exceptions.h"

void getCudaProperties(){
    cudaError_t err;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        throw CudaException(err);
    }
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
}

#endif