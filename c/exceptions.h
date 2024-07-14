#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <cuda_runtime.h>

/*
 * General Cuda exception to be
 * thrown so we can return a numeric
 * error state to our golang caller.
 */
class CudaException {
private:
    int errorState;
public:
    int error() const {
        return errorState;
    }
    explicit CudaException(cudaError_t err) : errorState(static_cast<int>(err)) {}
};

/*
 * General Program exception to be
 * thrown so we can return a numeric
 * error state to our golang caller.
 */
class ProgramError {
private:
    int errorState;
public:
    int error() const {
        return errorState;
    }
    explicit ProgramError(int err) : errorState(err){}
};

/*
 * A DivisionByZeroException our C code
 * can call on signal from CUDA when the
 * divisor is zero.
 *
 * To the mathematician I dated years ago,
 * every DivisionByZeroException reminds
 * me of my cheesy joke...and your hour-
 * long lecture of how mathematically dumb
 * it was.  Cheers!
 */
class DivisionByZeroException {
public:
    int error() const {
        const int divByZero = -1;
        return divByZero;
    }
    DivisionByZeroException() = default; // Default constructor
};

#endif // EXCEPTIONS_H
