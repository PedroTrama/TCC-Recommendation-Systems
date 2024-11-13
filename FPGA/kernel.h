#ifndef KERNEL_H
#define KERNEL_H

extern "C" void krnl_matrix_factorization(float* P, float* Q, float* R, int users, int movies, int factors);

#endif