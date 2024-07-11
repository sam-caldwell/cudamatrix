#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <cuda_runtime.h>

class CudaException {
private:
    int errorState;
public:
    int error() const {
        return errorState;
    }
    explicit CudaException(cudaError_t err) : errorState(static_cast<int>(err)) {}
};

class DivisionByZeroException {
public:
    int error() const {
        const int divByZero = -1;
        return divByZero;
    }
    DivisionByZeroException() = default; // Default constructor
};

#endif // EXCEPTIONS_H
