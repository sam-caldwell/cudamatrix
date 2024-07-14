package cudamatrix

/*
#cgo CFLAGS: -I.
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/build -lmatrix_math_linux_amd64
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/build -lmatrix_math_darwin_amd64
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/build -lmatrix_math_windows_amd64
#include "c/matrix_math.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"github.com/sam-caldwell/errors"
	"unsafe"
)

// Multiply multiplies two matrices and returns the result matrix.  (return nil on error state)
func (lhs *Matrix) Multiply(rhs *Matrix) (result *Matrix, err error) {

	lhs.lock.RLock()
	defer lhs.lock.RUnlock()

	rhs.lock.RLock()
	defer rhs.lock.RUnlock()

	if lhs.data.cols() != rhs.data.rows() {
		return nil, fmt.Errorf(errors.MatrixDimensionMismatch)
	}

	if result, err = NewMatrix(lhs.data.rows(), lhs.data.cols()); err != nil {
		return nil, err
	}

	lhsData := *lhs.flatten()
	rhsData := *rhs.flatten()
	resultData := make([]float64, lhs.data.rows()*lhs.data.cols())

	C.matrixMultiply(
		(*C.double)(unsafe.Pointer(&lhsData[0])),
		(*C.double)(unsafe.Pointer(&rhsData[0])),
		(*C.double)(unsafe.Pointer(&resultData[0])),
		C.int(int(lhs.data.rows())),
		C.int(int(lhs.data.cols())),
	)

	result.from1dArray(&resultData)

	return result, nil

}
