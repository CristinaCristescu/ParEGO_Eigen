//
//  Matrix.cpp
//  MatrixTest
//
//  Created by Bianca Cristina Cristescu on 28/10/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//
#include <stdio.h>
#include "Matrix.h"
#include </Users/cristina/developer/eigen/eigen-eigen-1306d75b4a21/Eigen/Dense>

MyMatrix::MyMatrix(size_t m, size_t n)
{
    fN = n;
    fM = m;
    fMatrix = MatrixXd(fN,fM);
}

void MyMatrix::insert(size_t i, size_t j, double val)
{
    fMatrix(i,j) = val;
}

MyMatrix& MyMatrix::operator=(const MyMatrix& A)
{
    fN = A.fN;
    fM = A.fM;
    fMatrix = A.fMatrix;
    return *this;
}

const MyMatrix MyMatrix::operator*(const MyMatrix& A)
{
    MyMatrix result;
    result.setN(fN);
    result.setM(A.fM);
    result.fMatrix = fMatrix*A.fMatrix;
    return result;
}

MyMatrix MyMatrix::operator+(const MyMatrix& A)
{
    MyMatrix result;
    result.setN(fN);
    result.setM(fM);
    result.fMatrix = fMatrix+A.fMatrix;
    return result;
}

ostream& operator<<(ostream& os, const MyMatrix& dt)
{
    os << dt.fMatrix;
    return os;
}


