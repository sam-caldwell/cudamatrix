#include <cuda_runtime.h>
#include <iostream>
#include "exceptions.h"
#include "initializeCuda.h"
#include "printMatrix.h"
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
extern "C" int matrixAdd(double *matrixA, double *matrixB, double *matrixC, int rows, int cols) {
    const int size = rows * cols;
    double* gpuMatrixA = nullptr;
    double* gpuMatrixB = nullptr;
    double* gpuMatrixC = nullptr;
    int gpuError = 0;
    try{
        initializeCuda(0);
        initializeGpuErrorFlag(&gpuError);
        initializeGpuMatrix(matrixA, gpuMatrixA, size, true);
        initializeGpuMatrix(matrixB, gpuMatrixB, size, true);
        initializeGpuMatrix(matrixC, gpuMatrixC, size, false);

        std::cout << "launch kernel:" << std::endl;

        // Launch kernel
        int threadsPerBlock = 16;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(gpuMatrixA, gpuMatrixB, gpuMatrixC, size, &gpuError);
        cudaDeviceSynchronize();

        captureGpuErrors(&gpuError);

        std::cout << "end state:" << std::endl;
        print1dMatrix(gpuMatrixA, size);
        print1dMatrix(gpuMatrixB, size);
        print1dMatrix(gpuMatrixC, size);

        copyGpuMatrixToHost(matrixC, gpuMatrixC, size);

    } catch (const CudaException& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
        return e.error();
    } catch (const ProgramError& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
        return e.error();
    } catch (const std::runtime_error& e) {
        const int unhandledException = -65535;
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
        return unhandledException;
    }
    freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
    return static_cast<int>(cudaSuccess);
}

/*
 * CUDA interface function: Divide two Matrices
 *
 *      c = a / b, return error_code (-1) or success (0)
 *
 * BAD: NEEDS WORK STILL
 *
 */
extern "C" int matrixDivide(double* matrixA, double* matrixB, double* matrixC, int rows, int cols) {
    int size = rows * cols;
    double* gpuMatrixA = nullptr;
    double* gpuMatrixB = nullptr;
    double* gpuMatrixC = nullptr;
    int gpuError = 0;

    try {
        initializeCuda(0);
        initializeGpuErrorFlag(&gpuError);
        initializeGpuMatrix(matrixA, gpuMatrixA, size, true);
        initializeGpuMatrix(matrixB, gpuMatrixB, size, true);
        initializeGpuMatrix(matrixC, gpuMatrixC, size, false);

        std::cout << "launch kernel:" << std::endl;

        int threadsPerBlock = 16;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
//        matrixDivideKernel<<<blocksPerGrid, threadsPerBlock>>>(gpuMatrixA, gpuMatrixB, gpuMatrixC, size, &gpuError);
        cudaDeviceSynchronize();

        captureGpuErrors(&gpuError);

        std::cout << "end state:" << std::endl;
        print1dMatrix(gpuMatrixA, size);
        print1dMatrix(gpuMatrixB, size);
        print1dMatrix(gpuMatrixC, size);

        copyGpuMatrixToHost(matrixC, gpuMatrixC, size);

    } catch (const CudaException& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
        return e.error();
    } catch (const ProgramError& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
        return e.error();
    } catch (const DivisionByZeroException& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
        return e.error();
    } catch (const std::runtime_error& e) {
        const int unhandledException = -65535;
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
        return unhandledException;
    }
    freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
    return static_cast<int>(cudaSuccess);  // Return success code
}

/*
 * CUDA interface function: Multiply two Matrices
 *
 *      c = a * b, return error_code (-1) or success (0)
 *
 * BAD: NEEDS WORK STILL
 *
 */
extern "C" int matrixMultiply(double* matrixA, double* matrixB, double* matrixC, int rows, int cols) {
    int size = rows * cols;
    double* gpuMatrixA = nullptr;
    double* gpuMatrixB = nullptr;
    double* gpuMatrixC = nullptr;
    int gpuError = 0;

    try {
        initializeCuda(0);
        initializeGpuErrorFlag(&gpuError);
        initializeGpuMatrix(matrixA, gpuMatrixA, size, true);
        initializeGpuMatrix(matrixB, gpuMatrixB, size, true);
        initializeGpuMatrix(matrixC, gpuMatrixC, size, false);

        int threadsPerBlock = 16;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(gpuMatrixA, gpuMatrixB, gpuMatrixC, rows, cols, &gpuError);
        cudaDeviceSynchronize();

        captureGpuErrors(&gpuError);

        std::cout << "end state:" << std::endl;
        print1dMatrix(gpuMatrixA, size);
        print1dMatrix(gpuMatrixB, size);
        print1dMatrix(gpuMatrixC, size);

        copyGpuMatrixToHost(matrixC, gpuMatrixC, size);

    } catch (const CudaException& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
        return e.error();
    } catch (const ProgramError& e){
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
        return e.error();
    } catch (const std::runtime_error& e) {
        const int unhandledException = -65535;
        freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
        return unhandledException;
    }
    freeMatrixMemory(gpuMatrixA,gpuMatrixB,gpuMatrixC, &gpuError);
    return static_cast<int>(cudaSuccess);  // Return success code
}
