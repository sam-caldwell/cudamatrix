.PHONY: all clean build build/cuda build/linux build/darwin build/windows

LD_LIBRARY_PATH=$(shell pwd)/build

all: clean build

build: build/linux build/darwin build/windows
	@echo "build complete"

clean:
	@echo "cleaning..."
	rm -rf build
	mkdir -p build

build/linux: build/cuda
build/darwin: build/cuda
build/windows: build/cuda

build/cuda:
	echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
	# Build for Linux
	nvcc -Xcompiler -fPIC -c -o build/matrix_math_linux_amd64.o c/matrix_math.cu
	nvcc -shared -o build/libmatrix_math_linux_amd64.so build/matrix_math_linux_amd64.o

	# Build for macOS
	nvcc -Xcompiler -fPIC -c -o build/matrix_math_darwin_amd64.o c/matrix_math.cu
	nvcc -shared -o build/libmatrix_math_darwin_amd64.dylib build/matrix_math_darwin_amd64.o

	# Build for Windows
	nvcc -Xcompiler -fPIC -c -o build/matrix_math_windows_amd64.o c/matrix_math.cu
	nvcc -shared -o build/libmatrix_math_windows_amd64.dll build/matrix_math_windows_amd64.o

# Use the following target to run the example
run: build
	echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH} go run examples/add2matrices/main.go

test: clean build
	@echo "testing..."
	echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
	CUDA_DEVICE_DEBUG=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH} go test -v -failfast .
