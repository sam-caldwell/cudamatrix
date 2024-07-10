# Makefile

.PHONY: all clean build build/cuda build/linux build/darwin build/windows

all: clean build

build: build/linux build/darwin build/windows

clean:
	rm -rf build
	mkdir -p build

build/linux: build/cuda
build/darwin: build/cuda
build/windows: build/cuda

build/cuda:
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
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(pwd)/build go run examples/add2matrices/main.go

test: build
	go test -v .
