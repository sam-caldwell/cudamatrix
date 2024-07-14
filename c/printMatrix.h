#ifndef PRINT_MATRIX_H
#define PRINT_MATRIX_H

#include <iostream>

// Function to print 2d matrix data
void print2dMatrix(double* matrix, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << matrix[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Function to print 1d matrix data
void print1dMatrix(double* gpuMatrix, int size){
    std::cout << "  |";
    for (int i = 0; i < size; ++i){
        std::cout << gpuMatrix[i] << " ";
    }
    std::cout << std::endl;
}

#endif