package cudamatrix

import (
	"sync"
)

// Matrix - represents a 2D floating-point matrix (for CUDA)
type Matrix struct {
	lock sync.RWMutex
	data MatrixElements
}
