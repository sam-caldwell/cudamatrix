
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
