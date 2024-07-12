#ifndef PRINT_MATRIX_H
#define PRINT_MATRIX_H

#include <iostream>

void printMatrix(double* matrix, int rows, int cols) {
    std::cout << "printMatrix() ...in c... starting." << std::endl;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << matrix[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "---" << std::endl;
}

// Function to print GPU matrix data
void printGpuMatrix(double* gpuMatrix, int size){
    double* hostMatrix = new double[size]; // Allocate memory on the host to store GPU data
    cudaError_t err = cudaMemcpy(hostMatrix, gpuMatrix, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy matrix from GPU to CPU: " << cudaGetErrorString(err) << std::endl;
        delete[] hostMatrix;
        return;
    }

    // Print the matrix elements
    std::cout << "GPU Matrix:" << std::endl;
    for (int i = 0; i < size; ++i){
        std::cout << hostMatrix[i] << " ";
    }
    std::cout << std::endl;

    delete[] hostMatrix; // Free host memory after use
}

#endif