//
//  Matrix.h
//  ParEGOIteration6
//
//  Created by Bianca Cristina Cristescu on 24/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef __ParEGOIteration6__Matrix__
#define __ParEGOIteration6__Matrix__

#include <stdio.h>
#include <Dense>

using namespace std;
using namespace Eigen;

class MyMatrix {
private:
    size_t fN;
    size_t fM;
    MatrixXd fMatrix;
public:
    MyMatrix() : fM(0), fN(0) {}
    MyMatrix(size_t m, size_t n);
    double operator()(size_t i, size_t j)
    {
        return fMatrix(i,j);
    }
    const double operator()(size_t i, size_t j) const
    {
        return fMatrix(i,j);
    }
    MyMatrix& operator=(const MyMatrix& A);
    MyMatrix operator+(const MyMatrix& A);
    const MyMatrix operator*(const MyMatrix& A);
    friend ostream& operator<<(ostream& os, const MyMatrix& dt);
    void insert(size_t i, size_t j, double val);
    void setN(size_t n) { fN = n;}
    void setM(size_t m) { fM = m;}
    const size_t getN() { return fN; }
    const size_t getM() { return fM; }
};



#endif /* defined(__ParEGOIteration6__Matrix__) */

