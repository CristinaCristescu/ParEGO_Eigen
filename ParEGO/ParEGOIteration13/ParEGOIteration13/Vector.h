/**
 * \class MyVector
 *
 *
 * \brief The class is a wrapper around the vector class.
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

#ifndef ParEGOIteration13_Vector_h
#define ParEGOIteration13_Vector_h

#include <vector>
#include <Dense>

using namespace Eigen;

class MyVector {
    
friend class MyMatrix;

private:
    size_t fN; ///< Size of the vector.
    VectorXd fVector; ///< Container for the vector.
public:
    MyVector() : fN(0) {}
    MyVector(size_t n);
    MyVector(size_t n, double init_value);
    MyVector(size_t n, std::vector<double> vector);
    MyVector(const MyVector& V);
    MyVector& operator=(const MyVector& A);

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
