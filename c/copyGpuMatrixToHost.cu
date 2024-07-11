
void copyGpuMatrixToHost(double* HostMatrix, double* GpuMatrix, int size){

    cudaError_t err;

    err = cudaMemcpy(HostMatrix, GpuMatrix, size * sizeof(double), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) throw CudaException(err);

}