#ifndef INITIALIZE_GPU_ERROR_FLAG
#define INITIALIZE_GPU_ERROR_FLAG

#include <cuda_runtime.h>
#include "exceptions.h"

/*
 * initialize the GPU Error flag variable.
 *
 *   This will be used to capture errors during runtime
 *   and return them to the host via captureGpuErrors().
 */
void initializeGpuErrorFlag(int*& gpuError);

#endif
