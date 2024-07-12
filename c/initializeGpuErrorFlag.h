#ifndef INITIALIZE_GPU_ERROR_FLAG
#define INITIALIZE_GPU_ERROR_FLAG

#include <cuda_runtime.h>
#include "exceptions.h"

/*
 * initialize the GPU Error flag variable.
 *
 *   This will be used to capture errors during runtime
 *   and return them to the host via captureGpuErrors().
 */
void initializeGpuErrorFlag(int*& gpuError){

    cudaError_t err;

    if (gpuError == nullptr) {

        err = cudaMalloc((void**)&gpuError, sizeof(int));

        if (err != cudaSuccess) throw CudaException(err);

        // Initialize the error flag to zero
        err = cudaMemset(gpuError, 0, sizeof(int));

        if (err != cudaSuccess) throw CudaException(err);

    }

}

#endif
