#ifndef FREE_MATRIX_MEMORY
#define FREE_MATRIX_MEMORY

#include <cuda_runtime.h>
#include "exceptions.h"

/*
 * Free the Matrix Memory Objects (A,B,C,err)
 *
 * Free any non-null memory objects
 */
void freeMatrixMemory(double* gpuMatrixA, double* gpuMatrixB, double* gpuMatrixC, int *error);

#endif