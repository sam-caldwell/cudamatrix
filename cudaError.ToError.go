package cudamatrix

import "fmt"

// ToError - render golang error
func (e *CudaError) ToError() error {
	if *e == CudaSuccess {
		return nil
	}
	return fmt.Errorf("CudaException(%d)", int(*e))
}
