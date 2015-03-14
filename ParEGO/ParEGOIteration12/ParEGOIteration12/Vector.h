//
//  Vector.h
//  ParEGOIteration7
//
//  Created by Bianca Cristina Cristescu on 25/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef ParEGOIteration7_Vector_h
#define ParEGOIteration7_Vector_h

#include <stdio.h>
#include <Dense>

using namespace std;
using namespace Eigen;

class MyVector {
    
    friend class MyMatrix;

private:
    size_t fN;
    VectorXd fVector;
public:
    MyVector() : fN(0) {}
    MyVector(size_t n);
    MyVector(size_t n, double init_value);
    MyVector(size_t n, double* vector);
    MyVector(const MyVector& V);
    MyVector& operator=(const MyVector& A);
    
    //TO DO: Move Constructor
    //TO DO: Move assignement
    
    //TO DO : Equality
    //TO DO : Non-equality

    double operator()(size_t i)
    {
        return fVector(i);
    }
    const double operator()(size_t i) const
    {
        return fVector(i);
    }
    
    const MyVector operator+(const MyVector& A) const;
    const MyVector operator-(const MyVector& A) const;
    const MyVector operator*(const MyVector& A) const;
    const MyMatrix operator*(const MyMatrix& A) const;
    const MyVector operator*(double coef) const;
    friend ostream& operator<<(ostream& os, const MyVector& dt);
    void insert(size_t i, double val);
    void setN(size_t n) { fN = n;}
    const size_t size() { return fN; }
    
    MyVector transpose();
};



#endif
