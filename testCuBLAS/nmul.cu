#include <cublas_v2.h>
#include <cstdlib>
#include <curand.h>
#include<iostream>
#include "cublas.h"

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

     // Destroy the handle
    cublasDestroy(handle);
}

void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {


    // Allocate 3 arrays on CPU
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

    // for simplicity we are going to use square arrays
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;
    
    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

    // Allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A,nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc((void**)&d_B,nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc((void**)&d_C,nr_rows_C * nr_cols_C * sizeof(float));

    // Fill the arrays A and B on GPU with random numbers
    //GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
    //GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
    for (int index = 0 ; index < 9 ; index++)
    {
        h_A[index] = 3.0;
        h_B[index] = 5.0;
        std::cout<<h_A[index]<<" "<<h_B[index]<<"\n";
    }
    std::cout << "A =" << std::endl;
    print_matrix(h_A, nr_rows_A, nr_cols_A);
    std::cout << "B =" << std::endl;
    print_matrix(h_B, nr_rows_B, nr_cols_B);

    // Optionally we can copy the data back on CPU and print the arrays
    cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
    
    float *hh_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *hh_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *hh_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
    cudaMemcpy(d_A,hh_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(d_B,hh_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "A =" << std::endl;
    print_matrix(hh_A, nr_rows_A, nr_cols_A);
    std::cout << "B =" << std::endl;
    print_matrix(hh_B, nr_rows_B, nr_cols_B);

    // Multiply A and B on GPU
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

    // Copy (and print) the result on host memory
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);

     

    //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);  

    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}