package cudamatrix

/*
#cgo CFLAGS: -I.
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/build -lmatrix_math_linux_amd64
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/build -lmatrix_math_darwin_amd64
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/build -lmatrix_math_windows_amd64
#include "c/matrix_math.h"
*/
import "C"
import (
	"fmt"
	"github.com/sam-caldwell/errors"
	"unsafe"
)

// Add adds multiple matrices and returns the result matrix.  (return nil on error state)
func (lhs *Matrix) Add(rhs *Matrix) (result *Matrix, err error) {

	if rhs == nil {
		return nil, fmt.Errorf(errors.NilPointer)
	}

	lhs.lock.RLock()
	defer lhs.lock.RUnlock()

	rhs.lock.RLock()
	defer rhs.lock.RUnlock()

	if !lhs.HasSameDimensions(rhs) {
		return nil, fmt.Errorf(errors.MatrixDimensionMismatch)
	}

	rows := int(lhs.data.rows())
	cols := int(lhs.data.cols())

	if rows == 0 || cols == 0 {
		return nil, fmt.Errorf(errors.BoundsCheckError)
	}

	lhsData := *lhs.flatten()
	rhsData := *rhs.flatten()
	resultData := make([]float64, rows*cols)

	var errorCode C.int
	errorCode = C.matrixAdd(
		(*C.double)(unsafe.Pointer(&lhsData[0])),
		(*C.double)(unsafe.Pointer(&rhsData[0])),
		(*C.double)(unsafe.Pointer(&resultData[0])),
		C.int(rows),
		C.int(cols),
	)

	if errorCode != CudaSuccess {
		if err = CintToError(errorCode); err != nil {
			return nil, err
		}
	}

	result, err = NewMatrix(uint(rows), uint(cols))
	if err != nil {
		return nil, err
	}
	err = result.from1dArray(&resultData)

	return result, err
}
