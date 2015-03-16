//
//  Vector.cpp
//  ParEGOIteration6
//
//  Created by Bianca Cristina Cristescu on 24/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#include "Matrix.h"
#include "Vector.h"

#include "Cuda_Utilities.h"

#include <stdio.h>
#include <iostream>

#define Debug_Matrix = false
#define GPU true
#define GPU_THRESHOLD 70
#define GPU_EXISTS

MyMatrix::MyMatrix(size_t m, size_t n)
{
    fRows = n;
    fColumns = m;
    fMatrix = MatrixXd(fRows,fColumns);
}

MyMatrix::MyMatrix(const MyMatrix& matrix)
{
    fRows = matrix.fRows;
    fColumns = matrix.fColumns;
    fMatrix = matrix.fMatrix;
}

MyMatrix::~MyMatrix()
{}

void MyMatrix::insert(int i, int j, double val)
{
    fMatrix(i,j) = val;
}

MyMatrix& MyMatrix::operator=(const MyMatrix& A)
{
    if (this == &A)
        return *this;
    fRows = A.fRows;
    fColumns = A.fColumns;
    fMatrix = A.fMatrix;
    return *this;
}

const MyMatrix MyMatrix::operator*(const MyMatrix& A) const
{
    MyMatrix result;
    result.fRows = fRows;
    result.fColumns = A.fColumns;

    assert(fColumns == A.fRows);
#if defined GPU_EXISTS
    if (fRows >= GPU_THRESHOLD || fColumns >= GPU_THRESHOLD || 
        A.fRows >= GPU_THRESHOLD || A.fColumns >= GPU_THRESHOLD)
    {
        // Create the raw array like represantion for the GPU
        float* this_matrix_gpu = (float*)malloc(fRows*fColumns*sizeof(float));
        float* A_gpu = (float*)malloc(A.fRows*A.fColumns*sizeof(float));

        for (int i = 0; i< fRows; i++)
            for (int j = 0; j < fColumns; j++)
            {
                this_matrix_gpu[j*fRows+i] = fMatrix(i,j);
            }
        for (int i = 0; i< A.fRows; i++)
            for (int j = 0; j < A.fColumns; j++)
            {
                A_gpu[j*A.fRows+i] = A.fMatrix(i,j);
            }
         

        //Result from the GPU.
        float* result_gpu = (float*)malloc(fRows*A.fColumns*sizeof(float));

        matrixMul(fRows, fColumns, A.fRows, A.fColumns, A.fColumns, fRows,
                  this_matrix_gpu, A_gpu, result_gpu);

        result.fMatrix = MatrixXd(result.fRows, result.fColumns);
        for (int i = 0; i < result.fRows; i++)
            for (int j = 0; j < result.fColumns; j++)
            {
                result.fMatrix(i,j) = result_gpu[i*result.fColumns+j];
            }
    }
    else {
        result.fMatrix = fMatrix*A.fMatrix;     
    }
#else
    result.fMatrix = fMatrix*A.fMatrix;     
#endif

    return result;
}

const MyMatrix MyMatrix::operator*(const MyVector& A) const
{
    MyMatrix result;
    result.fRows = fRows;
    //UF
  result.fColumns = 1;

    assert(fColumns == A.fN);
#if defined GPU_EXISTS
    if (fRows >= GPU_THRESHOLD || A.fN >= GPU_THRESHOLD)
    {
        // Create the raw array like represantion for the GPU
        float* this_matrix_gpu = (float*)malloc(fRows*fColumns*sizeof(float));
        float* A_gpu = (float*)malloc(A.fN*1*sizeof(float));

        for (int i = 0; i< fRows; i++)
            for (int j = 0; j < fColumns; j++)
            {
                this_matrix_gpu[j*fRows+i] = fMatrix(i,j);
                //fprintf(stderr, "%lg ", this_matrix_gpu[j*fRows+i]);
            }
            //fprintf(stderr, "\n");
        for (int index = 0; index < A.fN; index++)
        {
            A_gpu[index] = A.fVector(index);
            //fprintf(stderr, "%lg ", A_gpu[index]);
        }
        //fprintf(stderr, "\n");
        //Result from the GPU.
        float* result_gpu = (float*)malloc(fRows*1*sizeof(float));

      matrixMul(fRows, fColumns, A.fN, 1, fRows, 1,

                  this_matrix_gpu, A_gpu, result_gpu);

        result.fMatrix = MatrixXd(fRows, A.fN);
        for (int i = 0; i < result.fRows; i++)
            for (int j = 0; j < result.fColumns; j++)
            {
                result.fMatrix(i,j) = result_gpu[j*result.fColumns+i];
                //fprintf(stderr, "%lg\n", result.fMatrix(i,j));
            }
            //fprintf(stderr, "\n");
    }
    else 
    {
        result.fMatrix = fMatrix*A.fVector;
    } 
#else
        result.fMatrix = fMatrix*A.fVector;
#endif

    return result;
}

const MyMatrix MyMatrix::operator+(const MyMatrix& A) const
{
    MyMatrix result;
    result.setRows(fRows);
    result.setColumns(fColumns);
    result.fMatrix = fMatrix+A.fMatrix;
    return result;
}

double MyMatrix::posdet()
{
    // *** NB: this function changes R !!! ***
    // computes the determinant of R. If it is non-positive, then it adjusts R and recomputes.
    // returns the final determinant and changes R.
    double detR;
    detR = fMatrix.determinant();
    double diag=1.01;
    unsigned int nr = 0;
    while(detR<=0.0)
    {
        for(int i=0;i<fRows;i++)
            fMatrix(i,i)=diag;
        detR = fMatrix.determinant();
        nr++;
        diag+=0.01;
    }
    if(isnan(detR))
        assert(!isnan(detR));
    return(detR);
}

MyMatrix MyMatrix::inverse()
{
    MyMatrix result(fRows, fColumns);
    //std::cerr << fMatrix.determinant();
    //std::cerr << fMatrix.inverse();
    result.fMatrix = fMatrix.inverse();
    //std::cerr << result.fMatrix;
    result.fRows = result.fMatrix.rows();
    result.fColumns = result.fMatrix.cols();
    return result;
}

ostream& operator<<(ostream& os, const MyMatrix& dt)
{
    os << dt.fMatrix;
    return os;
}

MyVector::MyVector(size_t n)
{
    fN = n;
    fVector = VectorXd(fN);
}

MyVector::MyVector(size_t n, double init_value)
{
    fN = n;
    fVector = VectorXd(fN);
    for (int i = 0; i < fN; i++)
        fVector(i) = init_value;
}

//Construct a Vector object from an array.
//Start from index 1 in the array.
MyVector::MyVector(size_t n, double* vector)
{
    fN = n;
    fVector = VectorXd(fN);
    for(int i=0;i<fN;i++){
        //cerr << vector[i+1] << " ";
        fVector(i)=vector[i+1];
    }
}

MyVector::MyVector(const MyVector& V)
{
    fN = V.fN;
    fVector = V.fVector;
}

void MyVector::insert(size_t i, double val)
{
    fVector(i) = val;
}

MyVector& MyVector::operator=(const MyVector& A)
{
    if (this == &A)
        return *this;
    fN = A.fN;
    fVector = A.fVector;
    return *this;
}

const MyVector MyVector::operator*(const MyVector& A) const
{
    MyVector result;
    result.setN(fN);
    result.fVector = fVector*A.fVector;
    return result;
}

const MyMatrix MyVector::operator*(const MyMatrix& A) const
{
    MyMatrix result;
    result.fRows = 1;
    result.fColumns = A.fColumns;

    assert(fN == A.fRows);

#if defined GPU_EXISTS


    if (A.fRows >= GPU_THRESHOLD || fN >= GPU_THRESHOLD 
        || A.fColumns >= GPU_THRESHOLD)
    {
    //     cerr << "Before A:";
    // cerr << fVector;
    // cerr << A.fMatrix;
    // cerr << "Expected: " <<fVector.transpose()*A.fMatrix;

        // Create the raw array like represantion for the GPU
        float* this_vector_gpu = (float*)malloc(1*fN*sizeof(float));
        float* A_gpu = (float*)malloc(A.fRows*A.fColumns*sizeof(float));

        for (int index = 0; index < fN; index++){
            this_vector_gpu[index] = fVector(index);
            //fprintf(stderr, "%lg ", this_vector_gpu[index]);

        }
        //fprintf(stderr, "\n");
        for (int i = 0; i< A.fRows; i++)
            for (int j = 0; j < A.fColumns; j++)
            {
                A_gpu[j*A.fRows+i] = A.fMatrix(i,j);
                //fprintf(stderr, "%lg ", A_gpu[j*A.fRows+i]);
            }
            //fprintf(stderr, "\n");


        //Result from the GPU.
        float* result_gpu = (float*)malloc(1*A.fColumns*sizeof(float));

        matrixMul(1, fN, A.fRows, A.fColumns, 1, A.fColumns,
                  this_vector_gpu, A_gpu, result_gpu);

        result.fMatrix = MatrixXd(result.fRows, result.fColumns);
        for (int i = 0; i < result.fRows; i++)
            for (int j = 0; j < result.fColumns; j++)
            {
                result.fMatrix(i,j) = result_gpu[i*result.fColumns+j];
                //fprintf(stderr, "%lg ", result.fMatrix(i,j));
            }
            //fprintf(stderr, "\n");
            //cerr << result.fMatrix;
    }
    else 
    {
        result.fMatrix = fVector.transpose()*A.fMatrix;
        //cerr << result.fMatrix;
    }
#else
        result.fMatrix = fMatrix*A.fVector;
#endif

    return result;
}

const MyVector MyVector::operator*(double coef) const
{
    MyVector result;
    result.fVector = fVector*coef;
    result.fN = fN;
    return result;
}


const MyVector MyVector::operator+(const MyVector& A) const
{
    MyVector result;
    result.setN(fN);
    result.fVector = fVector+A.fVector;
    return result;
}

const MyVector MyVector::operator-(const MyVector& A) const
{
    MyVector result;
    result.setN(fN);
    result.fVector = fVector-A.fVector;
    return result;
}

ostream& operator<<(ostream& os, const MyVector& dt)
{
    os << dt.fVector;
    return os;
}

MyVector MyVector::transpose()
{
    MyVector result;
    result.fN = fN;
    result.fVector = fVector.transpose();
    return result;
}



