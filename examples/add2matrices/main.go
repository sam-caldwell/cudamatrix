package main

import (
	"fmt"
	"github.com/sam-caldwell/cudamatrix"
	"os"
)

func main() {
	a, _ := cudamatrix.NewMatrix(5, 5)
	b, _ := cudamatrix.NewMatrix(5, 5)
	c, err := a.Add(b)
	if err != nil {
		fmt.Print(err)
		os.Exit(1)
	}
	fmt.Printf("%v\n\n", *c)
}
