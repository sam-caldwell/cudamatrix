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

// Add adds multiple matrices and returns the result matrix.  (return nil on error state)
func (lhs *Matrix) Add(rhs *Matrix) (result *Matrix, err error) {

	lhs.lock.RLock()
	defer lhs.lock.RUnlock()

	rhs.lock.RLock()
	defer rhs.lock.RUnlock()

	if !lhs.HasSameDimensions(rhs) {
		return nil, fmt.Errorf(errors.MatrixDimensionMismatch)
	}

	result, err = NewMatrix(lhs.data.rows(), lhs.data.cols())
	if err != nil {
		return nil, err
	}

	lhsData := *lhs.flatten()
	rhsData := *rhs.flatten()
	resultData := make([]float64, lhs.data.rows()*lhs.data.cols())

	C.matrix_add(
		(*C.double)(unsafe.Pointer(&lhsData[0])),
		(*C.double)(unsafe.Pointer(&rhsData[0])),
		(*C.double)(unsafe.Pointer(&resultData[0])),
		C.int(int(lhs.data.rows())),
		C.int(int(lhs.data.cols())),
	)

	result.from1dArray(&resultData)

	return result, nil
}
