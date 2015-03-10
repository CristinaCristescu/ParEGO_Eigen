#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cublas.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define index(i,j,ld) (((j)*(ld))+(i))


using namespace std;

#include "Cuda_Utilities.h"

namespace Cuda_Utilities
{

}

extern "C" void matrixMul(int HA, int WA, int HB, int WB, int HC, int WC,
                          float* A, float* B, float* C)
{
    
    cublasStatus status;

    cudaDeviceProp deviceProp;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

    // use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;
    block_size = 8;
    //cerr << "block: " << block_size << "\n";

    float* AA; float* BB; float* CC;
    /*ALLOCATE ON THE DEVICE*/
    checkCudaErrors(cudaMalloc((void **) &AA, HA*WA*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &BB, HB*WB*sizeof(float)));
    checkCudaErrors(cudaMemcpy(AA, A, HA*WA*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(BB, B, HB*WB*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &CC, HC*WC*sizeof(float)));
   /* cerr << "Matrix A:";
     for (int i=0;i<HA*WA;i++)
        fprintf(stderr, "%lg ", A[i]);
        fprintf(stderr,"\n");
    cerr <<"Matrix B:" ;   
    for (int i=0;i<HB*WB;i++)
        fprintf(stderr, "%lg ", B[i]);
     fprintf(stderr,"\n");

     float* checkA = (float*)malloc(HA*WA*sizeof(float));
     cublasGetMatrix(HA,WA,sizeof(float),AA,HA,checkA,HA);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device read error (A)\n");
    }
    cerr << "Matrix A checkA";
     for (int i=0;i<HA*WA;i++)
        fprintf(stderr, "%lg ", checkA[i]);
        fprintf(stderr, "\n");

        float* checkB = (float*)malloc(HB*WB*sizeof(float));
     cublasGetMatrix(HB,WB,sizeof(float),BB,HB,checkB,HB);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device read error (A)\n");
    }
    cerr << "Matrix B checkB:";
     for (int i=0;i<HB*WB;i++)
        fprintf(stderr, "%lg ", checkB[i]);
        fprintf(stderr,"\n");

             fprintf (stderr, "%d %d %d %d %d %d", HA, WA, HB, WB, HC, WC);*/
    //fprintf(stderr,"hihi\n");
    /*KERNEL*/
    cudaDeviceSynchronize();
    cudaThreadSynchronize();

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(HC / threads.x, WC / threads.y);

    // CUBLAS version 2.0
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasHandle_t handle;

        checkCudaErrors(cublasCreate(&handle));

    

        checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HA,WB,WA, &alpha, AA, HA, BB, HB, &beta, CC, HC));

        // copy result from device to host
        checkCudaErrors(cudaMemcpy(C, CC, HC*WC*sizeof(float), cudaMemcpyDeviceToHost));

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));
    }

    
    // cerr << "Matrix C:";
    // for (int i=0;i<HC*WC;i++)
    //     fprintf(stderr, "%lg ", C[i]);
    //     fprintf(stderr,"\n");

    checkCudaErrors(cudaFree(AA));
    checkCudaErrors(cudaFree(BB));
    checkCudaErrors(cudaFree(CC));
        /* Shutdown */


}