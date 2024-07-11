#ifndef INITIALIZE_COPY_GPU_MATRIX
#define INITIALIZE_COPY_GPU_MATRIX

#include <cuda_runtime.h>
#include "exceptions.h"

/*
 * Copy GPU Matrix Back To Host
 *
 *   - copy GPU Matrix back to the host
 *   - throw exception on error
 */
void copyGpuMatrixToHost(double* HostMatrix, double* GpuMatrix, int size);

#endif