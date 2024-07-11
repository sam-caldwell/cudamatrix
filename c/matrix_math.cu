#include <cuda_runtime.h>
#include <iostream>
#include "exceptions.h"
#include "freeMatrixMemory.h"
#include "initializeGpuMatrix.h"
#include "initializeGpuErrorFlag.h"
#include "copyGpuMatrixToHost.h"
#include "captureGpuErrors.h"
#include "kernels/add.h"
#include "kernels/divide.h"
#include "kernels/multiply.h"

/*
 * CUDA interface function: Add two Matrices
 *
 *      c = a + b, return error_code (-1) or success (0)
 */
extern "C" int matrix_add(double* matrixA, double* matrixB, double* matrixC, int rows, int cols) {
    int size = rows * cols;
    double* gpuMatrixA = nullptr;
    double* gpuMatrixB = nullptr;
    double* gpuMatrixC = nullptr;
    int *gpuError = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    try {
        initializeGpuMatrix(matrixA, gpuMatrixA, size, true);
        initializeGpuMatrix(matrixB, gpuMatrixB, size, true);
        initializeGpuMatrix(matrixC, gpuMatrixC, size, false);
        initializeGpuErrorFlag(gpuError);

        matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(gpuMatrixA, gpuMatrixB, gpuMatrixC, size, gpuError);

        captureGpuErrors(gpuError);
        copyGpuMatrixToHost(matrixC, gpuMatrixC, size);
    } catch (const CudaException& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, gpuError);
        return e.error();
    } catch (const std::runtime_error& e) {
        const int unhandledException = -65535;
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, gpuError);
        return unhandledException;
    }
    freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, gpuError);
    return static_cast<int>(cudaSuccess);  // Return success code
}

/*
 * CUDA interface function: Divide two Matrices
 *
 *      c = a / b, return error_code (-1) or success (0)
 */
extern "C" int matrix_divide(double* matrixA, double* matrixB, double* matrixC, int rows, int cols) {
    int size = rows * cols;
    double* gpuMatrixA = nullptr;
    double* gpuMatrixB = nullptr;
    double* gpuMatrixC = nullptr;
    int* gpuError = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    try {
        initializeGpuMatrix(matrixA, gpuMatrixA, size, true);
        initializeGpuMatrix(matrixB, gpuMatrixB, size, true);
        initializeGpuMatrix(matrixC, gpuMatrixC, size, false);
        initializeGpuErrorFlag(gpuError);

        matrixDivideKernel<<<blocksPerGrid, threadsPerBlock>>>(gpuMatrixA, gpuMatrixB, gpuMatrixC, size, gpuError);

        captureGpuErrors(gpuError);
        copyGpuMatrixToHost(matrixC, gpuMatrixC, size);
    } catch (const CudaException& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, gpuError);
        return e.error();
    } catch (const DivisionByZeroException& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, gpuError);
        return e.error();
    } catch (const std::runtime_error& e) {
        const int unhandledException = -65535;
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, gpuError);
        return unhandledException;
    }
    freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, gpuError);
    return static_cast<int>(cudaSuccess);  // Return success code
}

/*
 * CUDA interface function: Multiply two Matrices
 *
 *      c = a * b, return error_code (-1) or success (0)
 */
extern "C" int matrix_multiply(double* matrixA, double* matrixB, double* matrixC, int rows, int cols) {
    int size = rows * cols;
    double* gpuMatrixA = nullptr;
    double* gpuMatrixB = nullptr;
    double* gpuMatrixC = nullptr;
    int* gpuError = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    try {
        initializeGpuMatrix(matrixA, gpuMatrixA, size, true);
        initializeGpuMatrix(matrixB, gpuMatrixB, size, true);
        initializeGpuMatrix(matrixC, gpuMatrixC, size, false);
        initializeGpuErrorFlag(gpuError);

        matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(gpuMatrixA, gpuMatrixB, gpuMatrixC, rows, cols, gpuError);

        captureGpuErrors(gpuError);
        copyGpuMatrixToHost(matrixC, gpuMatrixC, size);
    } catch (const CudaException& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, gpuError);
        return e.error();
    } catch (const std::runtime_error& e) {
        const int unhandledException = -65535;
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, gpuError);
        return unhandledException;
    }
    freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, gpuError);
    return static_cast<int>(cudaSuccess);  // Return success code
}
