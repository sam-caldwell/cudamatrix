
void freeMatrixMemory(double* gpuMatrixA, double* gpuMatrixB, double* gpuMatrixC, int *error){

    if (gpuMatrixA) cudaFree(gpuMatrixA);

    if (gpuMatrixB) cudaFree(gpuMatrixB);

    if (gpuMatrixC) cudaFree(gpuMatrixC);

    if (error) cudaFree(error);

}
