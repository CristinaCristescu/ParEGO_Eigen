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

#ifndef __ParEGOIteration13__Matrix__
#define __ParEGOIteration13__Matrix__

#include <stdio.h>
#include <Dense>

using namespace std;
using namespace Eigen;

class MyVector;

class MyMatrix {
    
friend class MyVector;
    
private:
    size_t fRows; ///< Number of rows in the matrix.
    size_t fColumns; ///< Number of columns in the matrix.
    MatrixXd fMatrix; ///< The container for the matrix.
public:
    MyMatrix() : fRows(0), fColumns(0) { fMatrix = MatrixXd(); }
    MyMatrix(size_t m, size_t n);
    MyMatrix(const MyMatrix& matrix);
    MyMatrix& operator=(const MyMatrix& A);
    ~MyMatrix();
    
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



#endif /* defined(__ParEGOIteration13__Matrix__) */

