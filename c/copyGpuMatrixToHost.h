#ifndef INITIALIZE_COPY_GPU_MATRIX
#define INITIALIZE_COPY_GPU_MATRIX

#include <iostream>
#include <cuda_runtime.h>
#include "exceptions.h"

/*
 * Copy GPU Matrix Back To Host
 *
 *   - copy GPU Matrix back to the host
 *   - throw exception on error
 */
void copyGpuMatrixToHost(double* hostMatrix, double* gpuMatrix, int size){

    cudaError_t err;

    std::cout << "copyGpuMatrixToHost() starting" << std::endl;
    err = cudaMemcpy(hostMatrix, gpuMatrix, size * sizeof(double), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) throw CudaException(err);
    std::cout << "copyGpuMatrixToHost() cudaSuccess" << std::endl;

    std::cout << "copyGpuMatrixToHost() hostMatrix" << std::endl;

    for (int i = 0; i < size; ++i){
        std::cout << hostMatrix[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "copyGpuMatrixToHost() done" << std::endl;
}

#endif