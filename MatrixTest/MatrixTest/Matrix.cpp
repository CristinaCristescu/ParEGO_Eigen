//
//  Matrix.cpp
//  MatrixTest
//
//  Created by Bianca Cristina Cristescu on 28/10/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//
#include <stdio.h>
#include <iostream>
#include "Matrix.h"

MyMatrix::MyMatrix(size_t m, size_t n)
{
    fN = n;
    fM = m;
    fMatrix = MatrixXd(fN,fM);
}

MyMatrix::MyMatrix(MyMatrix& matrix)
{
    fN = matrix.fN;
    fM = matrix.fM;
    fMatrix = matrix.fMatrix;
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
    MyMatrix result(0,0);
    result.fMatrix = MatrixXd::Zero(fN, A.fM);
    cerr << result.fMatrix << "\n";
    result.setN(fN);
    result.setM(A.fM);
    result.fMatrix = fMatrix*A.fMatrix;
    cerr << result.fMatrix;
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


