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

class MyVector;

class MyMatrix {
    
    friend class MyVector;
    
private:
    size_t fRows;
    size_t fColumns;
    MatrixXd fMatrix;
public:
    MyMatrix() : fRows(0), fColumns(0) { fMatrix = MatrixXd(); }
    MyMatrix(size_t m, size_t n);
    MyMatrix(const MyMatrix& matrix);
    MyMatrix& operator=(const MyMatrix& A);
    ~MyMatrix();
    //TO DO: Move Constructor
    //TO DO: Move assignement
    
    //TO DO : Equality
    //TO DO : Non-equality
    
    double operator()(size_t i, size_t j)
    {
        return fMatrix(i,j);
    }
    const double operator()(size_t i, size_t j) const
    {
        return fMatrix(i,j);
    }
    
    const MyMatrix operator+(const MyMatrix& A) const;
    const MyMatrix operator*(const MyMatrix& A) const;
    const MyMatrix operator*(const MyVector& A) const;
    friend ostream& operator<<(ostream& os, const MyMatrix& dt);
    void insert(int i, int j, double val);
    void setRows(size_t n) { fRows = n;}
    void setColumns(size_t m) { fColumns = m;}
    const size_t getRows() { return fMatrix.rows(); }
    const size_t getColumns() { return fMatrix.cols();}
    
    double posdet();
    MyMatrix inverse();
};



#endif /* defined(__ParEGOIteration6__Matrix__) */

