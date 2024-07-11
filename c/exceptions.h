#include <stdexcept>


class CudaMallocExceptionGpuA: public std::runtime_error {
public:
    explicit CudaMallocExceptionGpuA() : std::runtime_error("") {}
};

class CudaMallocExceptionGpuB: public std::runtime_error {
public:
    explicit CudaMallocExceptionGpuB() : std::runtime_error("") {}
};

class CudaMallocExceptionGpuC: public std::runtime_error {
public:
    explicit CudaMallocExceptionGpuC() : std::runtime_error("") {}
};

class CudaMallocExceptionErrorFlag : public std::runtime_error {
public:
    explicit CudaMallocExceptionErrorFlag() : std::runtime_error("") {}
};

class CudaMemcpyExceptionA : public std::runtime_error {
public:
    explicit CudaMemcpyExceptionA() : std::runtime_error("") {}
};

class CudaMemcpyExceptionB : public std::runtime_error {
public:
    explicit CudaMemcpyExceptionB() : std::runtime_error("") {}
};

class CudaMemcpyExceptionC : public std::runtime_error {
public:
    explicit CudaMemcpyExceptionC() : std::runtime_error("") {}
};

class CudaMemcpyExceptionErrorFlag : public std::runtime_error {
public:
    explicit CudaMemcpyExceptionErrorFlag() : std::runtime_error("") {}
};

class KernelLaunchException : public std::runtime_error {
public:
    explicit KernelLaunchException() : std::runtime_error("") {}
};

class DivisionByZeroException : public std::runtime_error {
public:
    explicit DivisionByZeroException() : std::runtime_error("") {}
};
