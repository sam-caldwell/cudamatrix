package cudamatrix

import "C"
import "fmt"

// CintToError - render golang error from C.int
func CintToError(e C.int) error {
	if e == CudaSuccess {
		return nil
	}
	return fmt.Errorf("CudaException(%d)", int(e))
}
