//
//  Matrix.cpp
//  ParEGOIteration6
//
//  Created by Bianca Cristina Cristescu on 24/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#include "Matrix.h"

#include <stdio.h>

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


