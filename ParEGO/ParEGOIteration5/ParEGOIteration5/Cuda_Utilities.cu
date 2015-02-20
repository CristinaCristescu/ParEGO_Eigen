#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"


namespace Cuda_Utilities
{

__device__ __host__ static matrixMul(int HA, int WA, int HB, int WB, int HC, int WC,
                                   double* A, double* B, double* C)
{
    float* AA; float* BB; float* CC;
    /*ALLOCATE ON THE DEVICE*/
    status=cublasAlloc(HA*WA,sizeof(float),(void**)&AA);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
    }

    status=cublasAlloc(HB*WB,sizeof(float),(void**)&BB);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
    }

    status=cublasAlloc(HC*WC,sizeof(float),(void**)&CC);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
    }

    /*SET MATRIX*/
    status=cublasSetMatrix(HA,WA,sizeof(float),A,HA,AA,HA);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
    }

    status=cublasSetMatrix(HB,WB,sizeof(float),B,HB,BB,HB);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
    }

    /*KERNEL*/
    cublasSgemm('n','n',HA,WB,WA,1,AA,HA,BB,HB,0,CC,HC);

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! kernel execution error.\n");
    }
    cublasGetMatrix(HC,WC,sizeof(float),CC,HC,C,HC);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device read error (A)\n");
    }

    free( A );  free( B );  free ( C );
    status = cublasFree(AA);
    free( A );  free( B );  free ( C );
    status = cublasFree(AA);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! memory free error (A)\n");
        if (status != CUBLAS_STATUS_SUCCESS) 
        {
            fprintf (stderr, "!!!! memory free error (A)\n");
        }
        status = cublasFree(BB);
        if (status != CUBLAS_STATUS_SUCCESS) 
        {
            fprintf (stderr, "!!!! memory free error (B)\n");
        }
        status = cublasFree(CC);
        if (status != CUBLAS_STATUS_SUCCESS) 
        {
            fprintf (stderr, "!!!! memory free error (C)\n");
        }

        /* Shutdown */
        status = cublasShutdown();
        if (status != CUBLAS_STATUS_SUCCESS) 
        {
            fprintf (stderr, "!!!! shutdown error (A)\n");
        }
    }
}
}