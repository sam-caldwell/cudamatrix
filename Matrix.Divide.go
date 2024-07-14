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

// Divide performs element-wise division of two matrices and returns the result matrix.  (return nil on error state)
func (lhs *Matrix) Divide(rhs *Matrix) (result *Matrix, err error) {

	lhs.lock.RLock()
	defer lhs.lock.RUnlock()

	rhs.lock.RLock()
	defer rhs.lock.RUnlock()

	if !lhs.HasSameDimensions(rhs) {
		return nil, fmt.Errorf(errors.MatrixDimensionMismatch)
	}

	if result, err = NewMatrix(lhs.data.rows(), lhs.data.cols()); err != nil {
		return nil, err
	}

	rows := int(lhs.data.rows())
	cols := int(lhs.data.cols())

	lhsData := *lhs.flatten()
	rhsData := *rhs.flatten()
	resultData := make([]float64, rows*cols)

	errorCode := C.matrixDivide(
		(*C.double)(unsafe.Pointer(&lhsData[0])),
		(*C.double)(unsafe.Pointer(&rhsData[0])),
		(*C.double)(unsafe.Pointer(&resultData[0])),
		C.int(rows),
		C.int(cols),
	)

	if errorCode != 0 {
		return nil, fmt.Errorf(errors.DivisionByZero)
	}

	err = result.from1dArray(&resultData)

	return result, err
}
