
void initializeGpuMatrix(double* HostMatrix, double*& GpuMatrix, int size, bool copyData){
    cudaError_t err;
    err = cudaMalloc((void**)&GpuMatrix, size * sizeof(double));
    if (err != cudaSuccess) throw CudaException(err);

    if (copyData){
        err = cudaMemcpy(GpuMatrix, HostMatrix, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw CudaException(err);
    }
}