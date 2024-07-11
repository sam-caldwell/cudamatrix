void captureGpuErrors(int *gpuErrorFlag){

    int hostErrorFlag = 0;
    const int divByZero = -1;
    cudaError_t err = cudaGetLastError();

    // Get the last CUDA error state
    err = cudaGetLastError();
    if (err != cudaSuccess) throw CudaException(err);

    if (gpuErrorFlag){
        // Copy the CUDA kernel error state (things we raised in our own programming)
        err = cudaMemcpy(&hostErrorFlag, gpuErrorFlag, sizeof(int), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) throw CudaException(err);
        if (hostErrorFlag == divByZero) throw DivisionByZeroException();
    }
}