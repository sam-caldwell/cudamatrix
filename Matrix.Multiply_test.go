package cudamatrix

import (
	"github.com/sam-caldwell/errors"
	"testing"
)

func TestMatrix_Multiply(t *testing.T) {

	//dumpMatrix := func(M *Matrix) {
	//	var v float64
	//	fmt.Print("----\n")
	//	for r := uint(0); r < M.Rows(); r++ {
	//		fmt.Printf("| ")
	//		for c := uint(0); c < M.Cols(); c++ {
	//			v, _ = M.Get(r, c)
	//			fmt.Printf("%04.1f ", v)
	//		}
	//		fmt.Print(" |\n")
	//	}
	//	fmt.Print("----\n")
	//}

	t.Run("Multiply lhs against nil rhs matrix", func(t *testing.T) {
		var A Matrix  // empty
		var B *Matrix // nil
		if C, err := A.Multiply(B); err == nil {
			t.Fatalf("error expected.  none found")
		} else {
			if err.Error() != errors.NilPointer {
				t.Fatalf("expected NilPointer error. Got: %v", err)
			}
			if C != nil {
				t.Fatalf("expect nil result on error")
			}
		}
	})
	t.Run("Multiply two empty matrices", func(t *testing.T) {
		var A Matrix  // empty
		var B *Matrix // nil
	})
	t.Run("Multiply two matrices where lhs cols != rhs rows. expect error", func(t *testing.T) {

	})
	t.Run("Multiply two matrices of same dimensions", func(t *testing.T) {

	})
	t.Run("Multiply two matrices of different dimensions where lhs cols == rhs rows. ok", func(t *testing.T) {

	})
}
