/**
 * \class MyMatrix
 *
 *
 * \brief The class is a wrapper around a matrix object.
 *
 * This class enables the code to be flexibles and maintanable.
 *
 * \note Copyright (c) 2015 Bianca-Cristina Cristescu. All rights reserved.
 *
 * \author (last to touch it) Bianca-Cristina Cristescu
 *
 * \version $Revision: 13
 *
 * \date $Date: 25/01/15.
 *
 */

#include "Matrix.h"
#include "Vector.h"

#include <stdio.h>
#include <iostream>
#include <vector>

/// Matrix default Constructor.
MyMatrix::MyMatrix(size_t m, size_t n)
{
    fRows = n;
    fColumns = m;
    fMatrix = MatrixXd(fRows,fColumns);
}

/// Copy-constructor.
MyMatrix::MyMatrix(const MyMatrix& matrix)
{
    fRows = matrix.fRows;
    fColumns = matrix.fColumns;
    fMatrix = matrix.fMatrix;
}

/// Destructor.
MyMatrix::~MyMatrix()
{}

/// Insert an element.
void MyMatrix::insert(int i, int j, double val)
{
    fMatrix(i,j) = val;
}

/// Operator=
MyMatrix& MyMatrix::operator=(const MyMatrix& A)
{
    if (this == &A)
        return *this;
    fRows = A.fRows;
    fColumns = A.fColumns;
    fMatrix = A.fMatrix;
    return *this;
}

/// Operator*
const MyMatrix MyMatrix::operator*(const MyMatrix& A) const
{
    MyMatrix result;
    result.fMatrix = fMatrix*A.fMatrix;
    result.fRows = result.fMatrix.rows();
    result.fColumns = result.fMatrix.cols();
    return result;
}

/// Operator*
const MyMatrix MyMatrix::operator*(const MyVector& A) const
{
    MyMatrix result;
    result.fMatrix = fMatrix*A.fVector;
    result.fRows = result.fMatrix.rows();
    result.fColumns = result.fMatrix.cols();
    return result;
}

/// Operator+
const MyMatrix MyMatrix::operator+(const MyMatrix& A) const
{
    MyMatrix result;
    result.setRows(fRows);
    result.setColumns(fColumns);
    result.fMatrix = fMatrix+A.fMatrix;
    return result;
}

/// Compute the determinant of teh matrix.
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

/// Compute teh inverse of the matrix.
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

/// Stream operator.
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
MyVector::MyVector(size_t n, std::vector<double> vector)
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
    MyMatrix result(1,A.fColumns);
    //std::cerr << result.fMatrix;
    //std::cerr << A.fMatrix;
    result.fMatrix = fVector.transpose()*A.fMatrix;
    //cerr << result.fMatrix;
    result.fRows = result.fMatrix.rows();
    result.fColumns = result.fMatrix.cols();
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



