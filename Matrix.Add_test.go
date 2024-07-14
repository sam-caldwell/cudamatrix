package cudamatrix

import (
	"github.com/sam-caldwell/errors"
	"testing"
)

func TestMatrix_Add(t *testing.T) {

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

	t.Run("Add nil rhs to Matrix", func(t *testing.T) {
		var A Matrix  //(empty)
		var B *Matrix //(nil)
		if C, err := A.Add(B); err == nil {
			t.Fatalf("err expected.  none found")
		} else {
			if err.Error() != errors.NilPointer {
				t.Fatalf("expected NilPointer error.  Got: %v", err)
			}
			if C != nil {
				t.Fatalf("expect nil result on error")
			}
		}
	})

	t.Run("Add two empty matrices", func(t *testing.T) {
		A := &(Matrix{})
		B := &(Matrix{})
		var (
			C   *Matrix
			err error
		)
		if C, err = A.Add(B); err == nil {
			t.Fatal("expected an error. got none")
		} else {
			if err.Error() != errors.BoundsCheckError {
				t.Fatalf("unexpected error.  Got %v", err)
			}
			if C != nil {
				t.Fatal("expect nil result on error")
			}
		}
	})
	t.Run("Add two (non-empty) matrices", func(t *testing.T) {
		var (
			A   *Matrix
			B   *Matrix
			C   *Matrix
			err error
		)
		t.Run("Setup the test", func(t *testing.T) {
			t.Run("Matrix A", func(t *testing.T) {
				A = &(Matrix{
					data: MatrixElements{
						{0, 1, 2, 3},
						{4, 5, 6, 7},
						{8, 9, 10, 11},
						{12, 13, 14, 15},
					},
				})
			})
			t.Run("Matrix B", func(t *testing.T) {
				B = &(Matrix{
					data: MatrixElements{
						{0, 1, 2, 3},
						{4, 5, 6, 7},
						{8, 9, 10, 11},
						{12, 13, 14, 15},
					},
				})
			})
			t.Run("Matrix C", func(t *testing.T) {
				C, err = A.Add(B)
				if err != nil {
					t.Fatal(err)
				}
				var v float64
				//t.Log("Addition has been performed...")
				//dumpMatrix(A)
				//dumpMatrix(B)
				//dumpMatrix(C)
				{
					if v, err = C.Get(0, 0); err != nil || v != 0 {
						t.Fatalf("value mismatch at (0,0):%v | %v", v, err)
					}
					if v, err = C.Get(0, 1); err != nil || v != 2 {
						t.Fatalf("value mismatch at (0,1):%v | %v", v, err)
					}
					if v, err = C.Get(0, 2); err != nil || v != 4 {
						t.Fatalf("value mismatch at (0,2):%v | %v", v, err)
					}
					if v, err = C.Get(0, 3); err != nil || v != 6 {
						t.Fatalf("value mismatch at (0,3):%v | %v", v, err)
					}
				}
				{
					if v, err = C.Get(1, 0); err != nil || v != 8 {
						t.Fatalf("value mismatch at (1,0):%v | %v", v, err)
					}
					if v, err = C.Get(1, 1); err != nil || v != 10 {
						t.Fatalf("value mismatch at (1,1):%v | %v", v, err)
					}
					if v, err = C.Get(1, 2); err != nil || v != 12 {
						t.Fatalf("value mismatch at (1,2):%v | %v", v, err)
					}
					if v, err = C.Get(1, 3); err != nil || v != 14 {
						t.Fatalf("value mismatch at (1,3):%v | %v", v, err)
					}
				}
				{
					if v, err = C.Get(2, 0); err != nil || v != 16 {
						t.Fatalf("value mismatch at (2,0):%v | %v", v, err)
					}
					if v, err = C.Get(2, 1); err != nil || v != 18 {
						t.Fatalf("value mismatch at (2,1):%v | %v", v, err)
					}
					if v, err = C.Get(2, 2); err != nil || v != 20 {
						t.Fatalf("value mismatch at (2,2):%v | %v", v, err)
					}
					if v, err = C.Get(2, 3); err != nil || v != 22 {
						t.Fatalf("value mismatch at (2,3):%v | %v", v, err)
					}
				}
				{
					if v, err = C.Get(3, 0); err != nil || v != 24 {
						t.Fatalf("value mismatch at (3,0):%v | %v", v, err)
					}
					if v, err = C.Get(3, 1); err != nil || v != 26 {
						t.Fatalf("value mismatch at (3,1):%v | %v", v, err)
					}
					if v, err = C.Get(3, 2); err != nil || v != 28 {
						t.Fatalf("value mismatch at (3,2):%v | %v", v, err)
					}
					if v, err = C.Get(3, 3); err != nil || v != 30 {
						t.Fatalf("value mismatch at (3,3):%v | %v", v, err)
					}
				}
			})
		})
	})
}
