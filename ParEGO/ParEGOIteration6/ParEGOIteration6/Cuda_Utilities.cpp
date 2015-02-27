#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>

#define index(i,j,ld) (((j)*(ld))+(i))


#include "Cuda_Utilities.h"

namespace Cuda_Utilities
{

void printMat(float*P,int uWP,int uHP){
//printf("\n %f",P[1]);
int i,j;
for(i=0;i<uHP;i++){

    printf("\n");

    for(j=0;j<uWP;j++)
        printf("%lg ",P[index(i,j,uHP)]);
        //printf("%lg ",P[i*uWP+j]);
}
}

extern "C" void matrixMul(int HA, int WA, int HB, int WB, int HC, int WC,
                          float* A, float* B, float* C)
{
    // Not sure it should be here!
    cublasInit();

    cublasStatus status;
    float* AA; float* BB; float* CC;
    /*ALLOCATE ON THE DEVICE*/
    status=cublasAlloc(HA*WA,sizeof(float),(void**)&AA);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED)
            fprintf(stderr, "CUBLAS_STATUS_NOT_INITIALIZED\n");
        else if (status == CUBLAS_STATUS_INVALID_VALUE)
            fprintf(stderr, "CUBLAS_STATUS_INVALID_VALUE\n");
        else if (status == CUBLAS_STATUS_ALLOC_FAILED)
            fprintf(stderr, "CUBLAS_STATUS_ALLOC_FAILED\n");
        fprintf (stderr, "!!!! device memory allocation error (AA)\n");
    }

    status=cublasAlloc(HB*WB,sizeof(float),(void**)&BB);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED)
            fprintf(stderr, "CUBLAS_STATUS_NOT_INITIALIZED\n");
        else if (status == CUBLAS_STATUS_INVALID_VALUE)
            fprintf(stderr, "CUBLAS_STATUS_INVALID_VALUE\n");
        else if (status == CUBLAS_STATUS_ALLOC_FAILED)
            fprintf(stderr, "CUBLAS_STATUS_ALLOC_FAILED\n");
        fprintf (stderr, "!!!! device memory allocation error (BB)\n");
    }

    status=cublasAlloc(HC*WC,sizeof(float),(void**)&CC);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED)
            fprintf(stderr, "CUBLAS_STATUS_NOT_INITIALIZED\n");
        else if (status == CUBLAS_STATUS_INVALID_VALUE)
            fprintf(stderr, "CUBLAS_STATUS_INVALID_VALUE\n");
        else if (status == CUBLAS_STATUS_ALLOC_FAILED)
            fprintf(stderr, "CUBLAS_STATUS_ALLOC_FAILED\n");
        fprintf (stderr, "!!!! device memory allocation error (CC)\n");
    }

    /*SET MATRIX*/
    status=cublasSetMatrix(HA,WA,sizeof(float),A,HA,AA,HA);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device memory copy error (A)\n");
    }

    status=cublasSetMatrix(HB,WB,sizeof(float),B,HB,BB,HB);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device memory copy error (B)\n");
    }

     /*for (int i=0;i<HA*WA;i++)
        fprintf(stderr, "%lg ", A[i]);
        fprintf(stderr,"\n");
    for (int i=0;i<HB*WB;i++)
        fprintf(stderr, "%lg ", B[i]);
     fprintf(stderr,"\n");
     float* checkA = (float*)malloc(HA*WA*sizeof(float));
     cublasGetMatrix(HA,WA,sizeof(float),AA,HA,checkA,HA);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device read error (A)\n");
    }
     for (int i=0;i<HA*WA;i++)
        fprintf(stderr, "%lg ", checkA[i]);
        fprintf(stderr, "\n");

        float* checkB = (float*)malloc(HB*WB*sizeof(float));
     cublasGetMatrix(HB,WB,sizeof(float),BB,HB,checkB,HB);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device read error (A)\n");
    }
    /* for (int i=0;i<HB*WB;i++)
        fprintf(stderr, "%lg ", checkB[i]);
        fprintf(stderr,"\n");

     fprintf (stderr, "%d %d %d %d %d %d", HA, WA, HB, WB, HC, WC);
    fprintf(stderr,"hihi\n");*/
    /*KERNEL*/
    cublasSgemm('n','n',HA,WB,WA,1,AA,HA,BB,HB,0,C,HC);

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! kernel execution error.\n");
        if (status == CUBLAS_STATUS_NOT_INITIALIZED)
            fprintf (stderr, "CUBLAS_STATUS_NOT_INITIALIZED\n");
        else if (status == CUBLAS_STATUS_INVALID_VALUE)
            fprintf(stderr, "CUBLAS_STATUS_INVALID_VALUE\n");  
        else if (status == CUBLAS_STATUS_ARCH_MISMATCH) 
            fprintf(stderr, "CUBLAS_STATUS_ARCH_MISMATCH\n");
        else if (status == CUBLAS_STATUS_EXECUTION_FAILED)
            fprintf (stderr, "CUBLAS_STATUS_EXECUTION_FAILED\n");        
    }

    

    cublasGetMatrix(HC,WC,sizeof(float),CC,HC,C,HC);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! device read error (Res)\n");
    }
    /*for (int i=0;i<HRes*WRes;i++)
        fprintf(stderr, "%lg ", Res[i]);
        fprintf(stderr,"\n");*/

    
    status = cublasFree(AA);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf (stderr, "!!!! memory free error (AA)\n");
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