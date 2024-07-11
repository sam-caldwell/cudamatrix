#ifndef INITIALIZE_GPU_MATRIX
#define INITIALIZE_GPU_MATRIX

#include <cuda_runtime.h>
#include "exceptions.h"

/*
 * GPU Matrix Initializer
 *
 *   - Allocate memory for the GPU.
 *   - If copyData is true, copy the HostMatrix into the GPU.
 */
void initializeGpuMatrix(double* HostMatrix, double*& GpuMatrix, int size, bool copyData);

#endif