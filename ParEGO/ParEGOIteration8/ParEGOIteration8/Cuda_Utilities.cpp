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

float* myAA; float* myBB; float* myCC;

void myalloc(int iter)
{
    /*ALLOCATE ON THE DEVICE*/
    /*ALLOCATE ON THE DEVICE*/
    checkCudaErrors(cublasAlloc(iter*iter,sizeof(float),(void**)&myAA));
    checkCudaErrors(cublasAlloc(iter*iter,sizeof(float),(void**)&myBB));
    checkCudaErrors(cublasAlloc(iter*iter,sizeof(float),(void**)&myCC));

}

void mydealloc()
{
    
    checkCudaErrors(cudaFree(myAA));
    checkCudaErrors(cudaFree(myBB));
    checkCudaErrors(cudaFree(myCC));
   

}

extern "C" void matrixMul(int HA, int WA, int HB, int WB, int HC, int WC,
                          float* A, float* B, float* C)
{
    
    clock_t time1, time2, start, end;
    start = clock();

    cublasStatus status;

    cudaDeviceProp deviceProp;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

    // use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;
    //cerr << "block: " << block_size << "\n";

    
   
    
    time1 = clock();
    
    cublasSetMatrix(HA,WA,sizeof(float),A,HA,myAA,HA);
    cublasSetMatrix(HB,WB,sizeof(float),B,HB,myBB,HB);
    
    time2 = clock();
    float diff1 =(float)time2-(float)time1;
    float seconds1 = diff1 / CLOCKS_PER_SEC;
    //fprintf(stdout, "Allocation: %lg\n", seconds1);

   /* cerr << "Matrix A:";
     for (int i=0;i<HA*WA;i++)
        fprintf(stderr, "%lg ", A[i]);
        fprintf(stderr,"\n");
    cerr <<"Matrix B:" ;   
    for (int i=0;i<HB*WB;i++)
        fprintf(stderr, "%lg ", B[i]);
     fprintf(stderr,"\n");

     float* checkA = (float*)malloc(HA*WA*sizeof(float));
     cublasGetMatrix(HA,WA,sizeof(float),myAA,HA,checkA,HA);
    if (status != CUBLAS_STATUS_SUmyCCESS) 
    {
        fprintf (stderr, "!!!! device read error (A)\n");
    }
    cerr << "Matrix A checkA";
     for (int i=0;i<HA*WA;i++)
        fprintf(stderr, "%lg ", checkA[i]);
        fprintf(stderr, "\n");

        float* checkB = (float*)malloc(HB*WB*sizeof(float));
     cublasGetMatrix(HB,WB,sizeof(float),myBB,HB,checkB,HB);
    if (status != CUBLAS_STATUS_SUmyCCESS) 
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

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(HC / threads.x, WC / threads.y);

    // CUBLAS version 2.0
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasHandle_t handle;

        checkCudaErrors(cublasCreate(&handle));

         time1 = clock();

        checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HA,WB,WA, &alpha, myAA, HA, myBB, HB, &beta, myCC, HC));

        time2 = clock();
        diff1 = ((float)time2-(float)time1);
        float seconds1 = diff1 / CLOCKS_PER_SEC;
        //fprintf(stdout, "Multiplication: %lg\n", seconds1);

        // copy result from device to host
        checkCudaErrors(cublasGetMatrix(HC,WC,sizeof(float),myCC,HC,C,HC));

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));
    }

    
    // cerr << "Matrix C:";
    // for (int i=0;i<HC*WC;i++)
    //     fprintf(stderr, "%lg ", C[i]);
    //     fprintf(stderr,"\n");

    
        /* Shutdown */
    end = clock();
    diff1 = ((float)time2-(float)time1);
    seconds1 = diff1 / CLOCKS_PER_SEC;
        //fprintf(stdout, "Overall: %lg\n", seconds1);
    

}

