//
//  Matrix.h
//  MatrixTest
//
//  Created by Bianca Cristina Cristescu on 28/10/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef __MatrixTest__Matrix__
#define __MatrixTest__Matrix__

#include <stdio.h>
#include </Users/cristina/developer/eigen/eigen-eigen-1306d75b4a21/Eigen/Dense>

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

#endif /* defined(__MatrixTest__Matrix__) */
